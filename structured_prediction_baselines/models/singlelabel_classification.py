import logging
from typing import List, Tuple, Union, Dict, Any, Optional
from overrides import overrides
from collections import defaultdict

import torch
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from structured_prediction_baselines.metrics import SinglelabelClassificationPairwiseDifference

from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register(
    "single-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@Model.register(
    "single-label-classification", constructor="from_partial_objects"
)
@ScoreBasedLearningModel.register(
    "single-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@ScoreBasedLearningModel.register(
    "single-label-classification", constructor="from_partial_objects"
)
class SinglelabelClassification(ScoreBasedLearningModel):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Metrics
        self.accuracy = CategoricalAccuracy()

        # If coarse labels are available, track label clusters to compute pairwise probability difference metrics
        coarse_clusters = None
        vocab = kwargs.pop("vocab")
        coarse_namespace = "coarse_labels"
        self.compute_pairwise = False
        if coarse_namespace in vocab._index_to_token.keys():
            coarse_idx2label = vocab.get_index_to_token_vocabulary(coarse_namespace)
            coarse_labels = coarse_idx2label.values()
            fine_idx2label = vocab.get_index_to_token_vocabulary("labels")
            coarse_clusters = defaultdict(list)
            sep_char = "_"  # Assuming fine labels are specified as "<coarse><sep_char><fine>"
            for idx in fine_idx2label:
                coarse_fine = fine_idx2label[idx].split(sep_char)
                if coarse_fine[0] in coarse_labels:
                    coarse_clusters[coarse_fine[0]].append(idx)
            coarse_clusters = list(map(lambda x: torch.tensor(x[1]), coarse_clusters.items()))
        if coarse_clusters is not None:
            self.compute_pairwise = True
            self.pairwise_diff_across = SinglelabelClassificationPairwiseDifference(coarse_clusters, mode="across")
            self.pairwise_diff_within = SinglelabelClassificationPairwiseDifference(coarse_clusters, mode="within")
            self.pairwise_diff_across_dropmax = SinglelabelClassificationPairwiseDifference(coarse_clusters,
                                                                                            mode="across",
                                                                                            drop_argmax=True)
            self.pairwise_diff_within_dropmax = SinglelabelClassificationPairwiseDifference(coarse_clusters,
                                                                                            mode="within",
                                                                                            drop_argmax=True)

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("x")
        _forward_args["labels"] = kwargs.pop("label")

        # Enable the use of meta dictionary:
        # metadata = kwargs.pop("meta")
        # if metadata is None:
        #     raise ValueError
        # _forward_args["buffer"]["meta"] = metadata

        # Enable the use of coarse label in the forward pass:
        # label_coarse = kwargs.pop("label_coarse")
        # if label_coarse is not None:
        #     _forward_args["buffer"]["label_coarse"] = label_coarse.cpu()

        return {**_forward_args, **kwargs}

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(1)

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(1)

    @torch.no_grad()
    @overrides
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:

        if not self.inference_module.is_normalized:
            y_hat_n = torch.softmax(y_hat, dim=-1)
        else:
            y_hat_n = y_hat

        self.accuracy(y_hat_n, labels)

        if self.compute_pairwise:
            self.pairwise_diff_across(y_hat_n)
            self.pairwise_diff_within(y_hat_n)
            self.pairwise_diff_across_dropmax(y_hat_n)
            self.pairwise_diff_within_dropmax(y_hat_n)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self.accuracy.get_metric(reset),
        }
        if self.compute_pairwise:
            metrics["pairwise_across"] = self.pairwise_diff_across.get_metric(reset)
            metrics["pairwise_within"] = self.pairwise_diff_within.get_metric(reset)
            metrics["pairwise_across_dropmax"] = self.pairwise_diff_across_dropmax.get_metric(reset)
            metrics["pairwise_within_dropmax"] = self.pairwise_diff_within_dropmax.get_metric(reset)
        return metrics
