"""Implements F1 using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional

from sklearn.metrics import f1_score
from allennlp.training.metrics import Metric, Average
import numpy as np
import torch


@Metric.register("singlelabel-f1-score")
class SinglelabelClassificationF1(Average):

    """Computes F1 score between true and predicted labels"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor
    ) -> None:  # type: ignore

        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]
        pred_idxs = np.argmax(scores, axis=1)
        scores = np.zeros_like(scores)
        scores[np.arange(len(scores)), pred_idxs] = 1

        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            sample_f1 = f1_score(single_example_labels, single_example_scores)
            super().__call__(sample_f1)
