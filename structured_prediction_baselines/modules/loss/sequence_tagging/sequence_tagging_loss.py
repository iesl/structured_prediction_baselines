from typing import Any, Optional, Tuple, cast, Union, Dict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import OracleValueFunction
from structured_prediction_baselines.modules.score_nn import ScoreNN


def flatten_y(
    labels: torch.Tensor, y_hat: torch.Tensor
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int, int]:
    _, num_samples, seq_length, _ = y_hat.shape

    return (
               labels.expand_as(y_hat).flatten(0, 2),
               y_hat.flatten(0, 2),
           ), num_samples, seq_length


def unflatten_metric(
    metric: torch.Tensor, num_samples: int, seq_length: int
) -> torch.Tensor:
    return metric.reshape(-1, num_samples, seq_length, *metric.shape[1:])


@Loss.register("sequence-tagging-loss")
class SequenceTaggingLoss(Loss):
    eps = 1e-9

    def __init__(self,
                 margin_type: int = 0,
                 beta: float = 1.0,
                 ce_weight: float = 1.0,
                 cross_entropy: bool = True,
                 zero_truncation: bool = False,
                 **kwargs):
        """
        margin_type: can be 0: margin-rescaled (default)
                            1: slack-rescaled
                            2: perceptron (delta = 0)
                            3: contrastive (delta = 1),
                            as described here https://arxiv.org/pdf/1803.03376.pdf
        cross_entropy: set True when training inference network and false for energy function, default True
        zero_truncation: set True when training for energy function, default False

        """
        super().__init__(**kwargs)
        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None")

        if self.oracle_value_function is None:
            raise ConfigurationError(
                "oracle_value_function cannot be None"
            )
        self.loss_fn = torch.nn.NLLLoss(reduction="none")
        if margin_type < 0 or margin_type > 3:
            raise ConfigurationError(
                "margin_type must be between 0 and 3"
            )
        self.margin_type = margin_type
        self.beta = beta
        self.ce_weight = ce_weight
        self.cross_entropy = cross_entropy
        self.zero_truncation = zero_truncation

    def forward(self, x: Any,
                labels: Optional[torch.Tensor],  # (batch, 1, ...)
                y_inf: torch.Tensor,  # (batch, num_samples, ...)
                y_cost_aug: torch.Tensor = None,  # (batch, num_samples, ...),
                buffer: Dict = None,
                y_hat_probabilities: Optional[torch.Tensor] = None,  # (batch, num_samples, ...)
                **kwargs: Any) -> torch.Tensor:
        if buffer is None:
            buffer = {}
            mask = util.get_text_field_mask(x)
            mask = mask.unsqueeze(dim=1)  # (batch_size, 1, ...)
            buffer["mask"] = mask

        ground_truth_score, inf_score, cost_aug_score, oracle_score = self._get_values(x, labels, y_inf, y_cost_aug,
                                                                                       buffer)

        hinge_score = cost_aug_score - ground_truth_score
        hinge_inference_score = inf_score - ground_truth_score
        if self.margin_type == 0:  # margin-rescaled
            hinge_score += oracle_score
        elif self.margin_type == 1:  # slack-rescaled
            hinge_score = oracle_score * (1.0 + hinge_score)
        elif self.margin_type == 2:  # perceptron
            pass
        elif self.margin_type == 3:  # contrastive
            hinge_score += 1

        if self.zero_truncation:
            hinge_score = F.relu(hinge_score)
            hinge_inference_score = F.relu(hinge_inference_score)

        cross_entropy_loss: torch.Tensor = torch.zeros_like(hinge_score)
        if self.cross_entropy:
            (labels, y_inf), num_samples, seq_length = flatten_y(labels, y_inf)
            # print(labels.shape)
            _, label_ids = labels.max(dim=1)
            cross_entropy_loss = self.loss_fn(torch.log(self.eps + y_inf), label_ids)
            cross_entropy_loss = unflatten_metric(cross_entropy_loss, num_samples, seq_length)
            cross_entropy_loss *= buffer["mask"]
            cross_entropy_loss = torch.sum(cross_entropy_loss, dim=-1)  # (batch_size, num_samples)

        loss_unreduced = -hinge_score - self.beta * hinge_inference_score + self.ce_weight * cross_entropy_loss

        return self.reduce(loss_unreduced)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_cost_aug: Optional[torch.Tensor],
        buffer: dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.oracle_value_function = cast(
            OracleValueFunction, self.oracle_value_function
        )  # purely for typing, no runtime effect
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        ground_truth_score = self.score_nn(x, labels, buffer)
        inference_score = self.score_nn(x, y_hat, buffer)
        cost_aug_score = self.score_nn(x, y_cost_aug, buffer)

        if labels is not None:
            oracle_score: Optional[torch.Tensor] = self.oracle_value_function(
                labels, y_hat, mask=buffer["mask"]
            )  # (batch, num_samples)
        else:
            oracle_score = None

        return ground_truth_score, inference_score, cost_aug_score, oracle_score
