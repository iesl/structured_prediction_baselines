from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from allennlp.common.lazy import Lazy


class Loss(torch.nn.Module, Registrable):
    """Base class for all the loss functions.

    In some cases, this will only act as a wrapper around loss modules from pytorch.
    """

    allowed_reductions = ["sum", "mean", "none"]

    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        reduction: Optional[str] = "sum",
        **kwargs: Any,
    ):
        """
        Args:
            score_nn: Needed if we are train a SPEN or DVN and need to compute v(x,y), i.e. the value of an input-output instance.
            oracle_value_function: Needed if we are doing DVN or SPEN.
        """
        super().__init__()  # type: ignore
        self.score_nn = score_nn
        self.oracle_value_function = oracle_value_function

        if reduction not in self.allowed_reductions:
            raise ValueError(
                f"reduction should be one of {self.allowed_reductions}"
            )
        self.reduction = reduction

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        y_hat_probabilities: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        loss_unreduced = self._forward(
            x, labels, y_hat, y_hat_probabilities, **kwargs
        )

        return self.reduce(loss_unreduced)

    def reduce(self, loss_unreduced: torch.Tensor):
        if self.reduction == "sum":
            return torch.sum(loss_unreduced)
        elif self.reduction == "mean":
            return torch.mean(loss_unreduced)
        elif self.reduction == "none":
            return loss_unreduced
        else:
            raise ValueError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        y_hat_probabilities: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: Any input tensor (MLC task) or dictionary of tensors (sequence tagging task) based on the task.
        Returns:
            loss of shape
                1. (batch, num_samples) if reduction is None
                2. (,), ie a scaler loss if reduction is sum or mean
        """
        raise NotImplementedError
