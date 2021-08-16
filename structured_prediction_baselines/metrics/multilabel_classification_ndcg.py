"""Implements Normalized Discounted Cumulative Gain using sklearn's ndcg_score"""

import torch
from allennlp.training.metrics import Metric, Average
from sklearn.metrics import ndcg_score


@Metric.register("multilabel-norm-discounted-cumulative-gain")
class MultilabelClassificationNormalizedDiscountedCumulativeGain(Average):

    """Computes normalized discounted cumulative gain"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, predicted_scores: torch.Tensor, true_scores: torch.Tensor
    ) -> None:  # type: ignore

        true_scores, predicted_scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(true_scores, predicted_scores)
        ]

        ndcg = ndcg_score(true_scores, predicted_scores)
        super().__call__(ndcg)