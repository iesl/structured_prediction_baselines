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
from allennlp.nn.util import viterbi_decode, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import allowed_transitions

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
        self.label_encoding = label_encoding
        self._f1_metric = SpanBasedF1Measure(
            vocab, tag_namespace=label_namespace, label_encoding=label_encoding
        )
        self._accuracy = CategoricalAccuracy()

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
        mask = util.get_text_field_mask(kwargs.get("tokens"))
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)
        return {"mask": mask}

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
        # y_hat: (batch, seq_len, num_labels)
        # labels: (batch, seq_len, num_labels) ie one-hot
        # mask: (batch, seq_len)
        mask = buffer.get("mask")
        assert mask is not None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask_length = mask.shape[1]
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        end_transitions = self.get_end_transitions()

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask).data.tolist()
        if y_hat.dim() == 3:
            predictions_list = [
                y_hat[i].detach().cpu() for i in range(y_hat.size(0))
            ]
        else:
            predictions_list = [y_hat]
        y_pred = []
        for predictions, length in zip(predictions_list, sequence_lengths):
            pred_indices, _ = viterbi_decode(
                    predictions[:length], transition_matrix, allowed_start_transitions=start_transitions, allowed_end_transitions=end_transitions
                )
            pred_indices = F.pad(torch.Tensor(pred_indices, device=device), (0, mask_length-len(pred_indices)))
            pred_indices = pred_indices.reshape(1, -1)
            pred_indices = self.convert_to_one_hot(pred_indices.to(torch.int64))
            y_pred.append(pred_indices)

        y_pred = torch.cat(y_pred, dim=0)
        labels_indices = torch.argmax(labels, dim=-1)
        self._f1_metric(y_pred, labels_indices, mask)
        self._accuracy(y_pred, labels_indices, mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_metric.get_metric(reset=reset)
        # metrics = {x: y for x, y in f1_dict.items() if "overall" in x}
        metrics = {x: y for x, y in f1_dict.items()}
        metrics["accuracy"] = self._accuracy.get_metric(reset=reset)
        return metrics

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIOUL labels.

        # Returns

        transition_matrix : `torch.Tensor`
            A `(num_labels, num_labels)` matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.ones([num_labels, num_labels])
        transition_matrix *= float("-inf")
        transitions = allowed_transitions(self.label_encoding, all_labels)
        for from_label, to_label in transitions:
            if from_label < num_labels and to_label < num_labels:
                transition_matrix[from_label][to_label] = 0
        # transition_matrix = transition_matrix[:num_labels, :num_labels]
        # for i, previous_label in all_labels.items():
        #     for j, label in all_labels.items():
        #         # B and L labels can not be preceded or followed by themselves
        #         if label[0] == "B" or label[0] == "L":
        #             transition_matrix[j, j] = float("-inf")
        #         # B labels can only be preceded by a L, U or an O tag.
        #         if i != j and label[0] == "B" and not (previous_label[0] in ["L", "U", "O"]):
        #             transition_matrix[i, j] = float("-inf")
        #         # I labels can only be preceded by themselves or their corresponding B tag.
        #         if i != j and label[0] == "I" and not previous_label == "B" + label[1:]:
        #             transition_matrix[i, j] = float("-inf")
        #         # L labels can only be preceded by their corresponding B or I tag
        #         if i != j and label[0] == "L" and not (previous_label == "I" + label[1:] or previous_label == "B" + label[1:]):
        #             transition_matrix[i, j] = float("-inf")
        #         # U labels can only be preceded by themselves, a L or an O tag.
        #         if i != j and label[0] == "U" and not (previous_label[0] == "L" or previous_label[0] == "O"):
        #             transition_matrix[i, j] = float("-inf")
        #         # O labels can only be preceded by themselves, a L or an U tag.
        #         if i != j and label[0] == "O" and not (previous_label[0] == "L" or previous_label[0] == "U"):
        #             transition_matrix[i, j] = float("-inf")

        return transition_matrix

    def get_start_transitions(self):
        """
        In the BIOUL sequence, we cannot start the sequence with an I-XXX or L-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        # Returns

        start_transitions : `torch.Tensor`
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I" or label[0] == "L":
                start_transitions[i] = float("-inf")

        return start_transitions

    def get_end_transitions(self):
        """
        In the BIOUL sequence, we cannot end the sequence with an I-XXX or B-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        # Returns

        end_transitions : `torch.Tensor`
            The pairwise potentials between a END token and
            the last token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        end_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I" or label[0] == "B":
                end_transitions[i] = float("-inf")

        return end_transitions
