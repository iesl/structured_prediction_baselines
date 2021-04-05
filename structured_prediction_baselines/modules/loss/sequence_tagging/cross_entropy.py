from typing import Any, Optional, Tuple, cast, Union, Dict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN


@Loss.register("sequence-tagging-masked-cross-entropy")
class SequenceTaggingMaskedCrossEntropyWithLogitsLoss(Loss):
    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, 1, ...)
        y_hat_extra: Optional[
            torch.Tensor
        ] = None,  # (batch, num_samples, ...),
        buffer: Dict = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert self.normalize_y, "This loss expects logits"
        assert labels is not None
        mask = buffer.get("mask") if buffer is not None else None

        if mask is None:
            mask = util.get_text_field_mask(x)  # (batch, seq_len, num_tags)

        assert y_hat.dim() == 4
        y_hat = y_hat.squeeze(1)  # (batch, seq_len, num_tags)
        labels = labels.squeeze(1)  # (batch, seq_len, num_tags)
        labels_indices = torch.argmax(labels, dim=-1)  # (batch, seq_len)

        return util.sequence_cross_entropy_with_logits(
            y_hat,  # type: ignore
            labels_indices,  # type:ignore
            mask,
            average=None,  # type: ignore
        ).unsqueeze(
            1
        )  # (batch, 1)
