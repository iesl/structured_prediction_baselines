from typing import List, Tuple, Union, Dict, Any, Optional
import torch
import numpy as np
from .base import ScoreBasedLearningModel
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.loss import Loss
from allennlp.data.vocabulary import Vocabulary
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.metrics import (
    MultilabelClassificationF1,
    MultilabelClassificationMeanAvgPrecision,
    MultilabelClassificationMicroAvgPrecision,
    MultilabelClassificationRelaxedF1,
    MultilabelClassificationAvgRank,
    MultilabelClassificationMeanReciprocalRank
)
from allennlp.models import Model
import logging

logger = logging.getLogger(__name__)


@Model.register(
    "multi-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@Model.register(
    "multi-label-classification", constructor="from_partial_objects"
)
class MultilabelClassification(ScoreBasedLearningModel):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # metrics
        self.f1 = MultilabelClassificationF1()
        self.map = MultilabelClassificationMeanAvgPrecision()
        self.micro_map = MultilabelClassificationMicroAvgPrecision()
        self.relaxed_f1 = MultilabelClassificationRelaxedF1()
        self.average_rank = MultilabelClassificationAvgRank()
        self.mrr = MultilabelClassificationMeanReciprocalRank()

    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze and turn the labels into one-hot if required"""
        # for mlc the labels already are in shape (batch, num_labels)
        # we just need to unsqueeze

        return labels.unsqueeze(1)

    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(1)

    def calculate_metrics(  # type: ignore
        self,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
    ) -> None:

        self.map(y_hat, labels)
        self.micro_map(y_hat, labels)

        if not self.inference_module.is_normalized:
            y_hat_n = torch.sigmoid(y_hat)
        else:
            y_hat_n = y_hat

        self.relaxed_f1(y_hat_n, labels)
        self.f1(y_hat_n, labels)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
            "relaxed_f1": self.relaxed_f1.get_metric(reset),
            "average_rank": self.average_rank.get_metric(reset),
            "MRR": self.mrr.get_metric(reset)
        }

        return metrics

    @torch.no_grad()
    def on_epoch(self, x: Any, labels: torch.Tensor, y_pred: torch.Tensor, buffer: Dict, num_samples: int, **kwargs: Any):
        if not self.inference_module.is_normalized:
            y_pred = torch.sigmoid(y_pred)

        p = y_pred.squeeze(1)  # (batch, num_labels)
        distribution = torch.distributions.Bernoulli(probs=p)

        # (batch, num_samples, num_labels)
        distribution_samples = torch.transpose(distribution.sample([num_samples]), 0, 1)
        random_samples = torch.transpose(
            torch.randint(low=0, high=2, size=(num_samples,) + p.shape, dtype=p.dtype, device=p.device), 0, 1
        )

        # (batch, num_samples+1, num_labels)
        ranking_samples = torch.hstack([self.unsqueeze_labels(labels), distribution_samples])

        ranking_scores = self.score_nn(x, ranking_samples, buffer)  # (batch, num_samples+1)
        ranking_labels = torch.zeros_like(ranking_scores)  # (batch, num_samples+1)
        ranking_labels[:, 0] = 1  # set true label index to 1

        # calculate evaluation only metrics
        self.average_rank(ranking_scores, ranking_labels)
        self.mrr(ranking_scores, ranking_labels)

        # call evaluation_module on distribution and random samples
        if self.evaluation_module:
            self.evaluation_module(x, labels, buffer, init_samples=distribution_samples, index=0)
            self.evaluation_module(x, labels, buffer, init_samples=random_samples, index=1)
