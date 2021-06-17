from typing import List, Tuple, Union, Dict, Any, Optional, cast
from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.loss.inference_net_loss import MarginBasedLoss
from allennlp.common.checks import ConfigurationError
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
import torch

# DVNLoss* are loss functions to train DVN,
# DVNScoreLoss* are loss functions to train infrence network with DVN.


class NCERankingLoss(Loss):
    """
    Loss function to train DVN, typically soft BCE loss.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for NCERankingLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # I want to get individual Energy scores & probability scores.
        predicted_score, oracle_value = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        return self.compute_loss(predicted_score, oracle_value)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, 1, ...)
        # samples shape (batch, num_samples, num_labels)
        samples = buffer['samples']
        num_samples = samples.shape[1] 

        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task

        if self.normalize_y:
            y_hat = self.normalize(y_hat)
        predicted_score_yhat = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, 1)

        if labels is not None:
            # For dvn we do not take gradient of oracle_score, so we detach y_hat
            noise_prob: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat.detach().clone(), **kwargs
            )  # (batch, num_samples)

        return predicted_score, noise_prob
    
    def get_metrics(self, reset: bool = False):
        metrics = {}
        return metrics
