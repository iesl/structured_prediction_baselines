from typing import Any, Optional, Dict, Tuple, cast

import torch
from allennlp.common.checks import ConfigurationError

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import OracleValueFunction
from structured_prediction_baselines.modules.score_nn import ScoreNN


class SGSpenLoss(Loss):
    def __init__(
        self,
        score_nn: ScoreNN,
        oracle_value_function: OracleValueFunction,
        oracle_cost_weight: float = 1.0,
        reduction: Optional[str] = "none",
        normalize_y: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            reduction=reduction,
            normalize_y=normalize_y,
        )

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None")

        if self.oracle_value_function is None:
            raise ConfigurationError("oracle_value_function cannot be None")

        if oracle_cost_weight == 0:
            raise ConfigurationError("oracle_cost_weight must be non zero")

        self.oracle_cost_weight = oracle_cost_weight
        self._oracle_cost_values = []
        self._y_hat_extra_score_values = []
        self._sample_score_values = []
        self._ground_truth_score_values = []

    def _forward(self, x: Any, labels: Optional[torch.Tensor], y_hat: torch.Tensor, y_hat_extra: Optional[
        torch.Tensor
    ], buffer: Dict, **kwargs: Any) -> torch.Tensor:
        assert buffer is not None
        (
            y_hat_oracle_cost,
            y_hat_score,
            y_hat_extra_oracle_cost,
            y_hat_extra_score
        ) = self._get_values(x, labels, y_hat, y_hat_extra, buffer)

        loss_unreduced = (
            self.oracle_cost_weight * (y_hat_oracle_cost - y_hat_extra_oracle_cost)
            + y_hat_score - y_hat_extra_score
        )
        return loss_unreduced

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,  # Assumed to be unnormalized
        y_hat_extra: Optional[torch.Tensor],  # assumed to be unnormalized
        buffer: dict,
        **kwargs: Any,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        assert (
            labels is not None
        )  # if you call this loss, labels cannot be None

        if y_hat_extra is None:
            y_hat_extra = (
                y_hat if not self.normalize_y else self.normalize(y_hat)
            )
        elif self.normalize_y:  # y_cost_aug is not None
            y_hat_extra = self.normalize(y_hat_extra)

        if self.normalize_y:
            y_hat = self.normalize(y_hat)

        y_hat_score = self.score_nn(x, y_hat, buffer)
        y_hat_extra_score = self.score_nn(x, y_hat_extra, buffer)

        y_hat_oracle_cost: torch.Tensor = self.oracle_value_function.compute_as_cost(
            labels, y_hat, mask=buffer.get("mask")
        )
        y_hat_extra_oracle_cost: torch.Tensor = self.oracle_value_function.compute_as_cost(
            labels, y_hat_extra, mask=buffer.get("mask")
        )  # (batch, num_samples)

        # self._oracle_cost_values.append(float(torch.mean(oracle_cost)))
        # self._y_hat_extra_score_values.append(float(torch.mean(y_hat_extra_score)))
        # self._inference_score_values.append(float(torch.mean(inference_score)))
        # self._ground_truth_score_values.append(float(torch.mean(ground_truth_score)))

        return (
            y_hat_oracle_cost,
            y_hat_score,
            y_hat_extra_oracle_cost,
            y_hat_extra_score,
        )

    def get_metrics(self, reset: bool = False):
        pass
