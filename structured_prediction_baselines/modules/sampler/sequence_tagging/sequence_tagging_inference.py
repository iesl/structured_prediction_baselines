import contextlib
import warnings
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    Generator,
)

import torch
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.stopping_criteria import StoppingCriteria, StopAfterNumberOfSteps
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import TaskNN


@Sampler.register(
    "sequence-tagging-inference", constructor="from_partial_objects"
)
class SequenceTaggingInference(Sampler):
    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Loss,  #: This loss can be different from the main loss
        inference_nn: Optional[TaskNN] = None,
        cost_augmented_layer: Optional[torch.nn.Module] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn
        assert self.loss_fn.reduction == "none", "We do reduction or our own"
        self.optimizer = optimizer
        self.stopping_criteria = stopping_criteria
        if isinstance(stopping_criteria, int):
            self.stopping_criteria = StopAfterNumberOfSteps(stopping_criteria)
        self.stopping_criteria = stopping_criteria
        self._different_training_and_eval = True

    @classmethod
    def from_partial_objects(
        cls,
        optimizer: Lazy[Optimizer],
        loss_fn: Lazy[Loss],  #: This loss can be different from the main loss
        inference_nn: Optional[TaskNN] = None,
        cost_augmented_layer: Optional[torch.nn.Module] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
    ) -> "SequenceTaggingInference":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        trainable_parameters = {}
        trainable_parameters.update(inference_nn.named_parameters())
        trainable_parameters.update(cost_augmented_layer.named_parameters())
        optimizer_ = optimizer.construct(model_parameters=list(trainable_parameters.items()))

        return cls(
            optimizer_,
            loss_fn_,
            inference_nn=inference_nn,
            score_nn=score_nn,
            cost_augmented_layer=cost_augmented_layer,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
        )

    def get_loss_fn(
        self, x: Any, labels: Optional[torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        # Sampler gets labels of shape (batch, ...), hence this
        # function will get labels of shape (batch*num_init_samples, ...)
        # but Loss expect y or shape (batch, num_samples or 1, ...)
        # Also during eval the loss is different. We inform the loss function
        # using None

        if self.training and (labels is None):
            warnings.warn("Labels should not be None in training mode!")

        def loss_fn(y_hat: torch.Tensor, y_cost_aug: torch.Tensor, buffer: Dict) -> torch.Tensor:
            return self.loss_fn(
                x,
                (
                    labels  # E:labels.unsqueeze(1)
                    if (self.training and labels is not None)
                    else None
                ),
                y_hat,
                y_cost_aug,
                buffer,
                None,
            )

        return loss_fn

    def get_dtype_device(self) -> Tuple[torch.dtype, torch.device]:
        for param in self.loss_fn.parameters():
            dtype = param.dtype
            device = param.device

            break

        return dtype, device

    def get_batch_size(self, x: Any) -> int:
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        else:
            raise NotImplementedError

    @contextlib.contextmanager
    def param_grad(self) -> Generator[None, None, None]:
        if self.inference_nn is not None and self.cost_augmented_layer is not None:
            trainable_modules = [self.inference_nn, self.cost_augmented_layer]
            try:
                for module in trainable_modules:
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = True
                        yield
            finally:
                # set the requires_grad back to false
                for module in trainable_modules:
                    if module is not None:
                        for param in module.parameters():
                            param.requires_grad = False
        else:
            warnings.warn(
                (
                    "You are using inference network-based model but either one or both of cost_augmented_layer "
                    "and inference_nn are None. Either you are not using the right Sampler or you have not "
                    "constructed it correctly"
                )
            )
            try:
                yield
            finally:
                pass

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ] = None,  #: If given will have shape (batch, ...)
        buffer: Dict = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if labels is None:
            y_inf: torch.Tensor = self.inference_nn(x, buffer)  # (batch_size, ...)
            return y_inf, None

        if buffer is None:
            buffer = {}
        # switch on gradients on parameters using context manager
        with self.param_grad():

            loss_fn = self.get_loss_fn(
                x, labels
            )  #: Loss function will expect labels in form (batch, num_samples or 1, ...)

            loss_values: List[float] = []
            loss_values_tensors: List[torch.Tensor] = []
            step_number = 0
            loss_value: Union[torch.Tensor, float] = float("inf")

            # we need to enable grad because if the top-level model
            # was being called in a validation loop, the training
            # flag will be False for all modules. This will not allow
            # gradient based inference to progress.
            with torch.enable_grad():
                while not self.stopping_criteria(step_number, float(loss_value)):
                    self.optimizer.zero_grad(set_to_none=True)
                    y_inf: torch.Tensor = self.inference_nn(x, buffer)  # (batch_size, ...)

                    labels = labels.unsqueeze(1)
                    y_inf = y_inf.unsqueeze(1)

                    y_cost_aug = self.cost_augmented_layer(torch.cat((y_inf, labels), dim=2))  # (batch_size, ...)
                    loss_value = self.update(
                        y_inf, y_cost_aug, buffer, loss_fn
                    )

                    loss_values.append(float(loss_value))

                    step_number += 1

                # buffer['y_inf'] = y_inf

        return y_cost_aug, y_inf

    def update(
        self,
        y_hat: torch.Tensor,
        y_cost_aug: torch.Tensor,
        buffer: Dict,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """
        loss = loss_fn(y_hat, y_cost_aug, buffer)

        loss.backward()  # type:ignore
        self.optimizer.step()

        return loss
