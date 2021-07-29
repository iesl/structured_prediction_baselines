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

import numpy as np
import torch
import itertools
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.stopping_criteria import (
    StoppingCriteria,
    StopAfterNumberOfSteps,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)


@Sampler.register("sg-spen", constructor="from_partial_objects")
class SGSpenSampler(Sampler):
    def __init__(
        self,
        loss_fn: Loss,
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        optimizer: Optimizer = None,
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        n_samples: int = 10,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self._different_training_and_eval = True

        if isinstance(stopping_criteria, int):
            self.stopping_criteria: StoppingCriteria = StopAfterNumberOfSteps(
                stopping_criteria
            )
        else:
            self.stopping_criteria = stopping_criteria
        self.n_samples = n_samples

    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return False

    @classmethod
    def from_partial_objects(
        cls,
        loss_fn: Lazy[Loss],  #: This loss can be different from the main loss
        inference_nn: TaskNN,  #: inference_nn cannot be None for this sampler
        score_nn: ScoreNN,
        optimizer: Lazy[Optimizer] = None,
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        n_samples: int = 10,
    ) -> "SGSpenSampler":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        trainable_parameters: Dict[str, torch.Tensor] = {}

        trainable_parameters.update(inference_nn.named_parameters())

        if cost_augmented_layer is not None:
            trainable_parameters.update(
                cost_augmented_layer.named_parameters()
            )

        optimizer_ = None
        if optimizer is not None:
            optimizer_ = optimizer.construct(
                model_parameters=list(trainable_parameters.items())
            )

        return cls(
            loss_fn_,
            inference_nn=inference_nn,
            score_nn=score_nn,
            optimizer=optimizer_,
            cost_augmented_layer=cost_augmented_layer,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
            n_samples=n_samples
        )

    def get_loss_fn(
        self, x: Any, labels: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor]:

        if self.training and (labels is None):
            warnings.warn("Labels should not be None in training mode!")

        def loss_fn(
            y_hat: torch.Tensor, y_cost_aug: torch.Tensor, buffer: Dict
        ) -> torch.Tensor:
            return self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
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
    def only_inference_nn_grad_on(self) -> Generator[None, None, None]:
        # switch off the gradients for score_nn but first cache their requires grad
        assert self.score_nn is not None
        score_nn_requires_grad_map = {
            name: param.requires_grad
            for name, param in self.score_nn.named_parameters()
        }
        try:
            # first switch off all.

            for p in self.score_nn.parameters():
                p.requires_grad_(False)
            # then switch on inf net params
            if self.optimizer:
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        p.requires_grad_(True)
            yield
        finally:
            # set the requires_grad back to false for inf net

            for n, p in self.score_nn.named_parameters():  # type: ignore
                p.requires_grad_(score_nn_requires_grad_map[n])

            if self.optimizer:
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        p.requires_grad_(False)

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if labels is None or (not self.training):
            y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
                1
            )  # (batch_size, 1, ...)

            return y_inf, None
        else:
            # switch on gradients on the parameters of inference network using context manager
            with self.only_inference_nn_grad_on():
                labels = labels.unsqueeze(1)
                loss_fn = self.get_loss_fn(
                    x, labels
                )  #: Loss function will expect labels in form (batch, num_samples or 1, ...)

                loss_values: List[float] = []
                step_number = 0
                loss_value: Union[torch.Tensor, float] = float("inf")

                while not self.stopping_criteria(
                    step_number, float(loss_value)
                ):
                    if self.optimizer:
                        self.optimizer.zero_grad(set_to_none=True)

                    y_inf, y_cost_aug = self._get_values(x, labels, buffer)
                    samples = self._get_samples(y_inf, self.n_samples)
                    sample_pairs = list(itertools.permutations(samples.transpose(0, 1), 2))
                    y_hat, y_hat_extra = tuple(map(torch.stack, zip(*sample_pairs)))
                    y_hat = torch.transpose(y_hat, 0, 1)
                    y_hat_extra = torch.transpose(y_hat_extra, 0, 1)
                    loss_value = self.update(
                        y_hat, y_hat_extra, buffer, loss_fn
                    )

                    loss_value = torch.mean(loss_value)
                    loss_values.append(float(loss_value))

                    step_number += 1
                self._metrics[self.name + '_loss'] = np.mean(loss_values)
                self._total_loss += np.mean(loss_values)
                self._num_batches += 1

            return (
                y_hat.detach().clone(),
                y_hat_extra.detach().clone()
            )

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

        if self.optimizer:
            loss.backward()  # type:ignore
            self.optimizer.step()

        return loss

    def _get_samples(
        self,
        y_pred: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        if not self.is_normalized:
            y_pred = torch.sigmoid(y_pred)

        p = y_pred.squeeze(1)  # (batch, num_labels)
        distribution = torch.distributions.Bernoulli(probs=p)
        samples = torch.transpose(distribution.sample([n_samples]), 0, 1)
        return samples

    def _get_values(
        self,
        x: Any,
        labels: torch.Tensor,
        buffer: Dict,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        y_inf: torch.Tensor = self.inference_nn(
            x, buffer
        ).unsqueeze(
            1
        )  # (batch_size, 1, ...) unormalized
        # inference_nn is TaskNN so it will output tensor of shape (batch, ...)
        # hence the unsqueeze

        if self.cost_augmented_layer is not None:
            y_cost_aug = self.cost_augmented_layer(
                torch.cat(
                    (
                        y_inf.squeeze(1),
                        labels.to(dtype=y_inf.dtype).squeeze(1),
                    ),
                    dim=-1,
                ),
                buffer,
            ).unsqueeze(
                1
            )  # (batch_size,1, ...)
        else:
            y_cost_aug = None

        return y_inf, y_cost_aug