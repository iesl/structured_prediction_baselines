import logging
from typing import Any, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import RegularizerApplicator, InitializerApplicator, util
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.score_nn import ScoreNN
from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register(
    "sequence-tagging-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@Model.register("sequence-tagging", constructor="from_partial_objects")
class SequenceTagging(ScoreBasedLearningModel):
    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        label_encoding: str,
        label_namespace: str = "labels",
        **kwargs: Any,
    ):
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        if not label_encoding:
            raise ConfigurationError("label_encoding was not specified.")
        self._f1_metric = SpanBasedF1Measure(
            vocab, tag_namespace=label_namespace, label_encoding=label_encoding
        )

    def convert_to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Converts the labels to one-hot if not already"""
        labels = F.one_hot(labels, num_classes=self.num_tags)

        return labels

    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze and turn the labels into one-hot if required"""
        # for mlc the labels already are in shape (batch, num_labels)
        # we just need to unsqueeze

        return labels.unsqueeze(1)

    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(1)

    def initialize_buffer(
        self,
        **kwargs: Any,
    ) -> Dict:
        return {"mask": util.get_text_field_mask(kwargs.get("tokens"))}

    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("tokens")
        _forward_args["labels"] = kwargs.pop("tags")
        _forward_args["meta"] = kwargs.pop("metadata")

        return {**_forward_args, **kwargs}

    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> None:
        mask = buffer.get("mask")
        mask = self.squeeze_y(mask)
        assert mask is not None
        # y_hat: (batch, seq_len, num_labels)
        # labels: (batch, seq_len, num_labels) ie one-hot
        # mask: (batch, seq_len)
        labels_indices = torch.argmax(labels, dim=-1)
        self._f1_metric(y_hat, labels_indices, mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_metric.get_metric(reset=reset)

        return {x: y for x, y in f1_dict.items() if "overall" in x}
