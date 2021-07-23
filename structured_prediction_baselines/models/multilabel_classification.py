import logging
from typing import Dict, Any

import torch
from allennlp.models import Model

from structured_prediction_baselines.metrics import (
    MultilabelClassificationF1,
    MultilabelClassificationMeanAvgPrecision,
    MultilabelClassificationMicroAvgPrecision,
    MultilabelClassificationRelaxedF1,
    MultilabelClassificationAvgRank,
    MultilabelClassificationMeanReciprocalRank,
    MultilabelClassificationNormalizedDiscountedCumulativeGain,
    MultilabelClassificationRankBiasedOverlap
)
from .base import ScoreBasedLearningModel

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
        self.ndcg = MultilabelClassificationNormalizedDiscountedCumulativeGain()
        self.rbo = MultilabelClassificationRankBiasedOverlap()

    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze and turn the labels into one-hot if required"""
        # for mlc the labels already are in shape (batch, num_labels)
        # we just need to unsqueeze

        return labels.unsqueeze(1)

    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(1)

    @torch.no_grad()
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
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

        distribution_samples = self.get_samples(y_hat_n, labels=labels)
        sample_scores = self.score_nn(x, distribution_samples, buffer)  # (batch, num_samples+1)
        true_scores = self.oracle_value_function(self.unsqueeze_labels(labels), distribution_samples)
        sample_labels = torch.zeros_like(sample_scores)  # (batch, num_samples+1)
        sample_labels[:, 0] = 1  # set true label index to 1

        # calculate score_nn metrics
        self.average_rank(sample_scores, sample_labels)
        self.mrr(sample_scores, sample_labels)
        self.ndcg(sample_scores, true_scores)
        self.rbo(sample_scores, true_scores)

        # call evaluation_module on distribution and random samples
        if self.evaluation_module:
            random_samples = self.get_samples(y_hat_n, random=True)
            self.evaluation_module(x, labels, buffer, init_samples=distribution_samples, index=0)
            self.evaluation_module(x, labels, buffer, init_samples=random_samples, index=1)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
            "relaxed_f1": self.relaxed_f1.get_metric(reset),
            "average_rank": self.average_rank.get_metric(reset),
            "MRR": self.mrr.get_metric(reset),
            "NDCG": self.ndcg.get_metric(reset),
            "RBO": self.rbo.get_metric(reset)
        }

        return metrics

    def get_samples(self, p, random=False, labels=None):
        num_samples = self.num_eval_samples
        if random:
            samples = torch.transpose(
                torch.randint(low=0, high=2, size=(num_samples,) + p.shape, dtype=p.dtype, device=p.device), 0, 1
            )
        else:
            distribution = torch.distributions.Bernoulli(probs=p)
            samples = torch.transpose(distribution.sample([num_samples]), 0, 1)  # (batch, num_samples, num_labels)

        # stack labels on top of samples
        if labels:
            samples = torch.hstack([self.unsqueeze_labels(labels), samples])

        return samples
