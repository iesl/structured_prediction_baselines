import itertools
from typing import Any, Optional, Dict, Tuple, cast

import torch
from allennlp.common.checks import ConfigurationError
import numpy as np

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
        n_samples: int = 20,
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

        self.n_samples = n_samples
        self.oracle_cost_weight = oracle_cost_weight
        self._n_valid_samples = []
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
            y_hat_n_oracle_cost,
            y_hat_n_score
        ) = self._get_values(x, labels, y_hat, y_hat_extra, buffer)

        loss_unreduced = (
            self.oracle_cost_weight * (y_hat_oracle_cost - y_hat_n_oracle_cost)
            + y_hat_score - y_hat_n_score
        )
        loss_value_filtered = torch.gt(loss_unreduced, 0)
        filtered_idx = torch.nonzero(loss_value_filtered, as_tuple=True)
        samples_mask = torch.zeros_like(loss_unreduced)
        samples_mask[filtered_idx] = 1
        loss_unreduced *= samples_mask
        # print(samples_mask.sum(dim=1))
        self._n_valid_samples.append(float(torch.mean(samples_mask.sum(dim=1))))
        loss_unreduced = loss_unreduced.sum(dim=1)/(samples_mask.sum(dim=1) + 1)
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

        if self.normalize_y:
            y_hat = self.normalize(y_hat)

        samples = self._get_samples(y_hat, self.n_samples)
        sample_pairs = list(itertools.permutations(samples.transpose(0, 1), 2))
        y_hat, y_hat_n = tuple(map(torch.stack, zip(*sample_pairs)))
        y_hat = torch.transpose(y_hat, 0, 1)
        y_hat_n = torch.transpose(y_hat_n, 0, 1)

        y_hat_score = self.score_nn(x, y_hat, buffer)
        y_hat_n_score = self.score_nn(x, y_hat_n, buffer)

        y_hat_oracle_cost: torch.Tensor = self.oracle_value_function.compute_as_cost(
            labels, y_hat, mask=buffer.get("mask")
        )
        y_hat_n_oracle_cost: torch.Tensor = self.oracle_value_function.compute_as_cost(
            labels, y_hat_n, mask=buffer.get("mask")
        )  # (batch, num_samples)

        # self._oracle_cost_values.append(float(torch.mean(oracle_cost)))
        # self._y_hat_extra_score_values.append(float(torch.mean(y_hat_n_score)))
        # self._inference_score_values.append(float(torch.mean(inference_score)))
        # self._ground_truth_score_values.append(float(torch.mean(ground_truth_score)))

        return (
            y_hat_oracle_cost,
            y_hat_score,
            y_hat_n_oracle_cost,
            y_hat_n_score,
        )

    def _get_samples(
        self,
        y_pred: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        p = y_pred.squeeze(1)  # (batch, num_labels)
        distribution = torch.distributions.Bernoulli(probs=p)
        samples = torch.transpose(distribution.sample([n_samples]), 0, 1)
        return samples

    def get_metrics(self, reset: bool = False):
        metrics = {}

        if self._n_valid_samples:
            metrics = {
                'n_valid_samples': np.mean(self._n_valid_samples),
            }

        if reset:
            self._n_valid_samples = []
        return metrics
