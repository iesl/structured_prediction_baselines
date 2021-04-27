import datetime
import logging
import math
import os
import time
import traceback
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from allennlp.common import Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataLoader
from allennlp.models.model import Model
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training import util as training_util
from torch.cuda import amp

from structured_prediction_baselines.training import utils

logger = logging.getLogger(__name__)


@Trainer.register("gradient_descent_custom", constructor="from_partial_objects")
class GradientDescentCustomTrainer(GradientDescentTrainer):
    """

    """

    def __init__(self, model: Model, optimizer: torch.optim.Optimizer,
                 data_loader: DataLoader, **kwargs):
        super().__init__(model, optimizer, data_loader, **kwargs)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self.optimizer.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for param_group in self.optimizer.param_groups:
                for p in param_group["params"]:
                    p.grad = None

            batch_loss = 0.0
            batch_group_outputs = []
            for batch in batch_group:
                with amp.autocast(self._use_amp):
                    batch_outputs = self.batch_outputs(batch, for_training=True)
                    batch_group_outputs.append(batch_outputs)
                    loss = batch_outputs["loss"]
                    reg_loss = batch_outputs.get("reg_loss")
                    if torch.isnan(loss):
                        raise ValueError("nan loss encountered")
                    loss = loss / len(batch_group)

                    batch_loss += loss.item()
                    if reg_loss is not None:
                        reg_loss = reg_loss / len(batch_group)
                        batch_reg_loss = reg_loss.item()
                        train_reg_loss += batch_reg_loss  # type: ignore

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

            train_loss += batch_loss

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = utils.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss,
                batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            if self._primary:
                # Updating tqdm only for the primary as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_outputs,
                    metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=batch_grad_norm,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = utils.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batch_loss=None,
            batch_reg_loss=None,
            num_batches=batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, Optional[float], int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_data_loader is not None:
            validation_data_loader = self._validation_data_loader
        else:
            raise ConfigurationError(
                "Validation results cannot be calculated without a validation_data_loader"
            )

        regularization_penalty = self.model.get_regularization_penalty()

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
        else:
            val_generator_tqdm = validation_data_loader

        batches_this_epoch = 0
        val_loss = 0.0
        val_batch_loss = 0.0
        val_reg_loss = None if regularization_penalty is None else 0.0
        val_batch_reg_loss = None if regularization_penalty is None else 0.0
        done_early = False
        for batch in val_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing validation early! "
                        "This implies that there is an imbalance in your validation "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            with amp.autocast(self._use_amp):
                batch_outputs = self.batch_outputs(batch, for_training=False)
                loss = batch_outputs.get("loss")
                reg_loss = batch_outputs.get("reg_loss")
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_batch_loss = loss.item()
                    val_loss += val_batch_loss
                    if reg_loss is not None:
                        val_batch_reg_loss = reg_loss.item()
                        val_reg_loss += val_batch_reg_loss  # type: ignore

            # Update the description with the latest metrics
            val_metrics = utils.get_metrics(
                self.model,
                val_loss,
                val_reg_loss,
                val_batch_loss,
                val_batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            description = training_util.description_from_metrics(val_metrics)
            if self._primary:
                val_generator_tqdm.set_description(description, refresh=False)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    [batch],
                    [batch_outputs],
                    val_metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=False,
                    is_primary=self._primary,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (validation)."
            )
            # Indicate that we're done so that any workers that have remaining data stop validation early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, val_reg_loss, batches_this_epoch

    def _try_train(self) -> Tuple[Dict[str, Any], int]:
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._primary and self._checkpointer is not None:
                self._checkpointer.save_checkpoint(epoch, self, save_model_only=True)

            # Wait for the primary process to finish saving the model checkpoint
            if self._distributed:
                dist.barrier()

            # get peak of memory usage
            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            this_epoch_val_metric: float = 0.0
            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, val_reg_loss, num_batches = self._validation_loss(epoch)

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.
                    if self._distributed:
                        dist.barrier()

                    val_metrics = utils.get_metrics(
                        self.model,
                        val_loss,
                        val_reg_loss,
                        batch_loss=None,
                        batch_reg_loss=None,
                        num_batches=num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=self.cuda_device,
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = self._metric_tracker.combined_score(val_metrics)
                    self._metric_tracker.add_metrics(val_metrics)
                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._primary:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"),
                    metrics,
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)

            if self._primary and self._checkpointer is not None:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            # Wait for the primary process to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            for callback in self._callbacks:
                callback.on_epoch(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1
        else:
            epoch = self._num_epochs - 1

        # Load the best model state before returning
        best_model_state = (
            None if self._checkpointer is None else self._checkpointer.best_model_state()
        )
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics, epoch