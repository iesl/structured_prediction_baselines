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
    return torch.softmax(y, dim=-1)


@Loss.register("sequence-tagging-margin-based")
class SequenceTaggingMarginBasedLoss(MarginBasedLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)


@Loss.register("sequence-tagging-inference")
class SequenceTaggingInferenceLoss(InferenceLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
