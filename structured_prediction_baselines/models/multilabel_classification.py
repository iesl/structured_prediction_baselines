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
        }

        # metrics.update(self.sampler.get_metrics(reset))
        if reset:
            for key in self.eval_only_metrics:
                metrics[key] = float(np.mean(self.eval_only_metrics[key]))
            self.eval_only_metrics = {}
        return metrics

    @torch.no_grad()
    def on_epoch(self, x: Any, labels: torch.Tensor, y_pred: torch.Tensor, buffer: Dict, num_samples: int, **kwargs: Any):
        if not self.inference_module.is_normalized:
            y_pred = torch.sigmoid(y_pred)

        p = y_pred.squeeze(1)  # (batch, num_labels)
        distribution = torch.distributions.Bernoulli(probs=p)
        distribution_samples = torch.transpose(distribution.sample([num_samples]), 0, 1)
        random_samples = torch.transpose(
            torch.randint(low=0, high=2, size=(num_samples,) + p.shape, dtype=p.dtype, device=p.device), 0, 1)

        distribution_samples_score = float(torch.mean(self.score_nn(x, distribution_samples, buffer)))
        random_samples_score = float(torch.mean(self.score_nn(x, random_samples, buffer)))
        self.eval_only_metrics['distribution_samples_score'] = self.eval_only_metrics.get(
            'distribution_samples_score', []) + [distribution_samples_score]
        self.eval_only_metrics['random_samples_score'] = self.eval_only_metrics.get(
            'random_samples_score', []) + [random_samples_score]

        # call sampler on distribution samples
        self.eval_only_module(x, labels, buffer, distribution_samples)
        dist_sampler_loss = self.eval_only_module.get_metrics(reset=True).get(
            'total_' + self.eval_only_module.name + '_loss')
        self.eval_only_metrics['dist_sampler_loss'] = self.eval_only_metrics.get(
            'dist_sampler_loss', []) + [dist_sampler_loss]

        # call sampler on random samples
        self.eval_only_module(x, labels, buffer, random_samples)
        random_sampler_loss = self.eval_only_module.get_metrics(reset=True).get(
            'total_' + self.eval_only_module.name + '_loss')
        self.eval_only_metrics['random_sampler_loss'] = self.eval_only_metrics.get(
            'random_sampler_loss', []) + [random_sampler_loss]
