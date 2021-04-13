from typing import List, Tuple, Union, Dict, Any, Optional, cast
from structured_prediction_baselines.modules.loss import Loss
from allennlp.common.checks import ConfigurationError
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
import torch


class DVNLoss(Loss):
    """
    Loss function to train DVN, typically soft BCE loss.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

        if self.oracle_value_function is None:
            raise ConfigurationError(
                "oracle_value_function cannot be None for DVNLoss"
            )

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
        # y_hat shape (batch, num_samples, ...)
        num_samples = y_hat[1]
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task

        if self.normalize_y:
            y_hat = self.normalize(y_hat)
        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        if labels is not None:
            # For dvn we do not take gradient of oracle_score, so we detach y_hat
            oracle_score: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat.detach().clone(), **kwargs
            )  # (batch, num_samples)
        else:
            oracle_score = None

        return predicted_score, oracle_score


class DVNScoreLoss(Loss):
    """
    Just uses score from the score network as the objective.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
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

        predicted_score = self._get_predicted_score(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        return self.compute_loss(predicted_score)

    def _get_predicted_score(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect

        # score_nn always expects y to be normalized
        # do the normalization based on the task

        if self.normalize_y:
            y_hat = self.normalize(y_hat)

        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        return predicted_score


class DVNCostAugLoss(Loss):
    """
    Loss function to train DVN, typically soft BCE loss.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

        if self.oracle_value_function is None:
            raise ConfigurationError(
                "oracle_value_function cannot be None for DVNLoss"
            )

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

        predicted_score_list, oracle_value_list = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        loss = (  self.compute_loss(predicted_score_list[0], oracle_value_list[0]) 
                + self.compute_loss(predicted_score_list[1], oracle_value_list[1]))
        
        return loss

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
        # y_hat shape (batch, num_samples, ...)
        num_samples = y_hat[1]
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task

        if self.normalize_y:
            y_hat = self.normalize(y_hat)
            y_hat_extra = self.normalize(y_hat_extra)
        
        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)
        predicted_score_extra = self.score_nn(
            x, y_hat_extra, buffer, **kwargs
        )  # (batch, num_samples)

        if labels is not None:
            # For dvn we do not take gradient of oracle_score, so we detach y_hat
            oracle_score: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat.detach().clone(), **kwargs
            )  # (batch, num_samples)
            oracle_score_extra: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat_extra.detach().clone(), **kwargs
            )  # (batch, num_samples)
        else:
            oracle_score = None
            oracle_score_extra = None
        
        predicted_score_list = [predicted_score, predicted_score_extra]
        oracle_score_list = [oracle_score, oracle_score_extra]
        return predicted_score_list, oracle_score_list


class DVNScoreCostAugLoss(Loss):
    """
    Just uses score from the score network as the objective, 
    but also train CostAug network (i.e. Cost-Augmented Network).
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for DVNLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
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

        pred_score_infnet, pred_score_costaugnet = self._get_predicted_score(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )
        
        loss = (  self.compute_loss(pred_score_infnet) 
                + self.compute_loss(pred_score_costaugnet))
        return loss

    def _get_predicted_score(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect

        # score_nn always expects y to be normalized
        # do the normalization based on the task

        if self.normalize_y:
            y_hat = self.normalize(y_hat)

        predicted_score_infnet = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        predicted_score_costaug = self.score_nn(
            x, y_hat_extra, buffer, **kwargs
        )  # (batch, num_samples)

        return [predicted_score_infnet, predicted_score_costaug]



