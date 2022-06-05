from typing import Any, Optional, Tuple, cast, Union, Dict

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors
from allennlp.nn import util
from torch.nn.functional import relu
import torch.nn.functional as F

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y, dim=-1)


@Loss.register("machine-translation-masked-cross-entropy")
class MachineTranslationMaskedCrossEntropyWithLogitsLoss(Loss):
    def _forward(
        self,
        x: Any,
        labels: Optional[TextFieldTensors],  # (batch, 1, ...), target_tokens here
        y_hat: torch.Tensor,  # (batch, 1, ...)
        y_hat_extra: Optional[
            torch.Tensor
        ] = None,  # (batch, num_samples, ...),
        buffer: Dict = None,
        **kwargs: Any,
    ) -> torch.Tensor:

        assert labels is not None
        mask = util.get_text_field_mask(labels)
        targets = util.get_token_ids_from_text_field_tensors(labels)

        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = mask[:, 1:].contiguous()

        assert y_hat.dim() == 4
        y_hat = y_hat.squeeze(1)  # (batch, seq_len, num_tags)

        return util.sequence_cross_entropy_with_logits(
            y_hat,  # type: ignore
            relevant_targets,  # type:ignore
            relevant_mask,
            average=None,  # type: ignore
        ).unsqueeze(
            1
        )  # (batch, 1)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
