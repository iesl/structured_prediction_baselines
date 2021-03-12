from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from .base import ScoreBasedLearningModel
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.cost_function import CostFunction
from structured_prediction_baselines.modules.loss import Loss
from allennlp.data.vocabulary import Vocabulary
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.metrics import (
    MultilabelClassificationF1,
    MultilabelClassificationMeanAvgPrecision,
    MultilabelClassificationMicroAvgPrecision,
)
from allennlp.models import Model
import logging

logger = logging.getLogger(__name__)


@Model.register("multi-label-classification")
class MultilabelClassification(ScoreBasedLearningModel):
    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        # metrics
        self.f1 = MultilabelClassificationF1()
        self.map = MultilabelClassificationMeanAvgPrecision()
        self.micro_map = MultilabelClassificationMicroAvgPrecision()

    def calculate_metrics(  # type: ignore
        self, labels: torch.Tensor, y_hat: torch.Tensor
    ) -> None:

        if y_hat.dim() == 3:  # (batch, num_samples or 1, num_labels)
            # While calculating metrics, we assume that
            # there aren't multiple samples.
            assert (
                y_hat.shape[1] == 1
            ), f"Incorrect size ({y_hat.shape[1]}) of samples dimension. Expected (1)"
            y_hat = y_hat.squeeze(1)
        # At this point we assume that y_hat is of shape (batch, num_labels)
        # where each entry in [0,1]
        self.f1(y_hat, labels)
        self.map(y_hat, labels)
        self.micro_map(y_hat, labels)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        return {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
        }
