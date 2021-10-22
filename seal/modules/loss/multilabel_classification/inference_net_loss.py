from typing import Any, Optional, Tuple, cast, Union, Dict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from seal.modules.loss import Loss
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.score_nn import ScoreNN

from seal.modules.loss.inference_net_loss import (
    MarginBasedLoss,
    InferenceLoss,
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
