from typing import List, Tuple, Union, Dict, Any, Optional, Generator, Iterable
from allennlp.common.registrable import Registrable
import torch


class OracleValueFunction(Registrable):
    """
    Either a differentiable (w.r.t y) or non-differentiable function that takes in true label
    and an set of arbitrary y's(either discrete in case of non-differentiable value) or
    a continuous relaxations. The shape of input y will be (batch, num_samples or 1, ...).

    This will not be an instance of torch.nn.Module because we do not expect it to carry any parameters.
    """

    def __init__(self, differentiable: bool = True, **kwargs: Any):
        super().__init__()
        self.differentiable = differentiable

    def flatten_y(
        self, labels: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        num_samples = y_hat.shape[1]

        return (
            labels.expand_as(y_hat).flatten(0, 1),
            y_hat.flatten(0, 1),
        ), num_samples

    def unflatten_metric(
        self, metric: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        return metric.reshape(-1, num_samples, *metric.shape[2:])

    def compute(
        self, labels: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.

        return (
            x.detach() if isinstance(x, torch.Tensor) else x for x in tensors
        )

    def __call__(
        self,
        labels: torch.Tensor,  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.differentiable:
            labels, y_hat = self.detach_tensors(labels, y_hat)

        (labels, y_hat), num_samples = self.flatten_y(labels, y_hat)
        value = self.compute(labels, y_hat)
        value = self.unflatten_metric(value, num_samples)

        return value
