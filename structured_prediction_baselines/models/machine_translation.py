import logging
from typing import Dict, Any, Iterable

import torch
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.nn import util
from allennlp.training.metrics import BLEU
from overrides import overrides

from .base import ScoreBasedLearningModel

START_SYMBOL = '[CLS]'
END_SYMBOL = '[SEP]'
PAD_TOKEN = '[PAD]'

logger = logging.getLogger(__name__)


@Model.register(
    "seal-machine-translation",
    constructor="from_partial_objects_with_inference_module_as_sampler",
)
@Model.register(
    "machine-translation", constructor="from_partial_objects"
)
class MachineTranslation(ScoreBasedLearningModel):
    def __init__(
        self,
        target_namespace: str = "labels",
        bleu_ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # metrics
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        pad_index = self.vocab.get_token_index(
            PAD_TOKEN, target_namespace
        )
        self._bleu = BLEU(
            bleu_ngram_weights,
            exclude_indices={pad_index, self._end_index, self._start_index},
        )

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("source_tokens")
        # _forward_args["buffer"]["mask"] = util.get_text_field_mask(
        #     _forward_args["x"]
        # )
        _forward_args["labels"] = kwargs.pop("target_tokens")
        # _forward_args["meta"] = kwargs.pop("metadata")
        # _forward_args["buffer"]["meta"] = _forward_args["meta"]

        return {**_forward_args, **kwargs}

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @torch.no_grad()
    @overrides
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: TextFieldTensors,
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:

        # shape: (batch_size, beam_size, max_sequence_length)
        top_k_predictions = y_hat
        # shape: (batch_size, max_predicted_sequence_length)
        best_predictions = top_k_predictions[:, 0, :]
        targets = util.get_token_ids_from_text_field_tensors(labels)

        self._bleu(best_predictions, targets)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        metrics.update(self._bleu.get_metric(reset=reset))
        return metrics
