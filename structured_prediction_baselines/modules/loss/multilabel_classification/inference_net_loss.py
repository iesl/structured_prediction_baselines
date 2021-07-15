from typing import Any, Optional, Tuple, cast, Union, Dict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.loss.sg_spen_loss import SGSpenLoss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN

from structured_prediction_baselines.modules.loss.inference_net_loss import (
    MarginBasedLoss,
    InferenceLoss,
    InferenceScoreLoss,
)


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)


@Loss.register("multi-label-margin-based")
class MultiLabelMarginBasedLoss(MarginBasedLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)


@Loss.register("multi-label-inference")
class MultiLabelInferenceLoss(InferenceLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)


@Loss.register("multi-label-inference-score")
class MultiLabelInferenceScoreLoss(InferenceScoreLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)


@Loss.register("multi-label-sg-spen")
class MultiLabelSGSpenLoss(SGSpenLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

@Loss.register("zero-task-nn-loss")
class ZeroTaskNNLoss(InferenceScoreLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        return 0 * super().forward(
                x, labels, y_hat, y_hat_extra, buffer, **kwargs
                )

