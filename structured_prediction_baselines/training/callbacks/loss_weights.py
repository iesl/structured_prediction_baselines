from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from allennlp.common.lazy import Lazy
from allennlp.common.checks import ConfigurationError
# from allennlp.training.callbacks import TrainerCallback, TrackEpochCallback

from allennlp.training.trainer import (
    GradientDescentTrainer,
    TrainerCallback,
    TrackEpochCallback,
)

import warnings
import logging

logger = logging.getLogger(__name__)

@TrainerCallback.register("lossweight-set-callback")
class TurnOnLossAfterEpochs(TrackEpochCallback):
    """
    This callback sets provided loss index (losses in the loss_idx_list) 
    to be turned on/off if `model.epoch`> self.epoch_to_turn_on which can be read inside `forward()`. 
    This callback lets you pass to the `GradientDescentTrainer` to access the current epoch number in your model during training. 
    The losses in loss_idx_list will be initially set to 0 and turned on after trainig few epochs (self.epoch_to_turn_on).
    """
    def __init__(
        self,
        serialization_dir: str,
        loss_idx_list: Optional[List[int]] = None,
        epoch_to_turn_on: Optional[List[int]]=None,
        initial_weight_list: Optional[List[int]]=None,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
        )
        self.loss_idx_list = loss_idx_list
        self.epoch_to_turn_on = epoch_to_turn_on
        if loss_idx_list is not None:
            if epoch_to_turn_on is not None: # both provided.
                if len(loss_idx_list) != len(epoch_to_turn_on):
                    raise ConfigurationError(
                        "`epoch_to_turn_on` (List) should have the same length with `loss_idx_list`."
                    )
                else: 
                    self.initial_weight_list = [0.0]* len(loss_idx_list)
            else: # just loss_idx_list provided.
                raise ConfigurationError(
                    "`epoch_to_turn_on` (List) should be specified when `loss_idx_list` is specified."
                )
        elif epoch_to_turn_on is not None: # just epoch_to_turn_on provided.
            raise ConfigurationError(
                "`loss_idx_list` (List) should be specified when `epoch_to_turn_on` is specified."
            )

    def get_loss_weights_then_set0(self, trainer: "GradientDescentTrainer"):
        for loss_idx in self.loss_idx_list:
            self.initial_weight_list[loss_idx] = trainer.model.sampler.loss_fn.loss_weights[loss_idx] 
            trainer.model.sampler.loss_fn.loss_weights[loss_idx]  = 0

    def set_loss_weights(self, trainer: "GradientDescentTrainer", loss_idx):
        trainer.model.sampler.loss_fn.loss_weights[loss_idx] = self.initial_weight_list[loss_idx]

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary,**kwargs) # --> trainer.model.epoch = 0  # type: ignore[assignment]
        self.get_loss_weights_then_set0(trainer)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        Overriding on_epoch to control the weights.
        """
        super().on_epoch(trainer, metrics, epoch, is_primary,**kwargs) # --> trainer.model.epoch = epoch + 1  # type: ignore[assignment]
        if self.loss_idx_list is not None and self.epoch_to_turn_on is not None:
            for i, epoch_thresh in enumerate(self.epoch_to_turn_on):
                if trainer.model.epoch > epoch_thresh:
                   self.set_loss_weights(trainer, self.loss_idx_list[i]) #trainer.model.sampler.loss_fn.loss_weights[loss_idx] = self.initial_weight_list[loss_idx]




# @TrainerCallback.register("c")
# class CustomTensorBoardCallback(TensorBoardCallback):
#     def __init__(
#         self,
#         serialization_dir: str,
#         tensorboard_writer: Lazy[TensorBoardWriter] = Lazy(TensorBoardWriter),
#         model_outputs_to_log: List[str] = None,
#     ) -> None:
#         super().__init__(
#             serialization_dir=serialization_dir,
#             tensorboard_writer=tensorboard_writer,
#         )
#         self._model_outputs_to_log = model_outputs_to_log or []
#         self._warned_about_missing_keys = False

#     def _warn_about_missing_keys(
#         self, model_outputs: List[Dict[str, Any]]
#     ) -> None:

#         if not self._warned_about_missing_keys:
#             for key in self._model_outputs_to_log:
#                 if key not in model_outputs[0]:
#                     logger.warning(f"Key {key} missing in model outputs.")
#             self._warned_about_missing_keys = True

#         return

#     def on_batch(
#         self,
#         trainer: "GradientDescentTrainer",
#         batch_inputs: List[List[TensorDict]],
#         batch_outputs: List[Dict[str, Any]],
#         batch_metrics: Dict[str, Any],
#         epoch: int,
#         batch_number: int,
#         is_training: bool,
#         is_primary: bool = True,
#         batch_grad_norm: Optional[float] = None,
#         **kwargs: Any,
#     ) -> None:
#         # do everything as the parent does
#         super().on_batch(
#             trainer,
#             batch_inputs,
#             batch_outputs,
#             batch_metrics,
#             epoch,
#             batch_number,
#             is_training,
#             is_primary=is_primary,
#             batch_grad_norm=batch_grad_norm,
#             **kwargs,
#         )
#         assert len(batch_outputs) == 1, "Gradient accumulation not supported"
#         self._warn_about_missing_keys(batch_outputs)

#         for key in self._model_outputs_to_log:
#             value = batch_outputs[0].get(key, None)

#             if value is not None:
#                 if is_training:
#                     self._tensorboard.add_train_histogram(  # type: ignore
#                         "model_outputs/" + key, value
#                     )
# # @TrainerCallback.register("track_epoch_callback")
# # class TrackEpochCallback(TrainerCallback):
# #     """
# #     A callback that you can pass to the `GradientDescentTrainer` to access the current epoch number
# #     in your model during training. This callback sets `model.epoch`, which can be read inside of
# #     `model.forward()`. We set `model.epoch = epoch + 1` which now denotes the number of
# #     completed epochs at a given training state.
# #     """
# #     def on_start(
# #         self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
# #     ) -> None:
# #         super().on_start(trainer, is_primary)
# #         trainer.model.apply(lambda module: module.epoch = 0)
# #         trainer.model.epoch = 0
# #     def on_epoch(
# #         self,
# #         trainer: "GradientDescentTrainer",
# #         metrics: Dict[str, Any],
# #         epoch: int,
# #         is_primary: bool = True,
# #         **kwargs,
# #     ) -> None:
# #         trainer.model.apply(lambda module: module.epoch = module.epoch+1)
# #         trainer.model.epoch = epoch + 1