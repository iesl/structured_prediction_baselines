"""Implements F1 using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional

from sklearn.metrics import f1_score
from allennlp.training.metrics import Metric, Average
import torch


@Metric.register("multilabel-f1-score")
class MultilabelClassificationF1(Average):

    """Computes F1 score between true and predicted labels"""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor
    ) -> None:  # type: ignore

        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]
        scores[scores < self.threshold] = 0
        scores[scores >= self.threshold] = 1

        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            sample_f1 = f1_score(single_example_labels, single_example_scores)
            super().__call__(sample_f1)
