"""Implements pairwise probability differences between classes"""
from typing import List, Tuple, Union, Dict, Any, Optional, Literal

from allennlp.training.metrics import Metric, Average
import torch


@Metric.register("singlelabel-pairwise-difference")
class SinglelabelClassificationPairwiseDifference(Average):
    """Computes the across- and within-class pairwise probability differences"""

    def __init__(self, clusters: list, mode: Literal["across", "within"] = "across", drop_argmax: bool = False) -> None:
        """
        Args:
            clusters: List of tensors, each containing fine-label indexes belonging to the same coarse-label
            mode: Specifies whether the metric is computed over fine-labels from different coarse-labels ("across")
                  or the same coarse-label ("within")
            drop_argmax: Specifies whether the highest probability label should be removed from the calculations
        """

        super().__init__()
        self.clusters = clusters
        self._mode = mode
        self._drop_argmax = drop_argmax

    def __call__(
            self, predictions: torch.Tensor
    ) -> None:
        scores = [t.cpu() for t in self.detach_tensors(predictions)][0]  # shape: (B x L)
        # Compute pairwise probability difference between each label class
        differences = torch.abs((scores.unsqueeze(-1) - scores.unsqueeze(1)))  # shape: (B x L x L)
        max_class = torch.argmax(scores, dim=1)  # shape: (B,)

        if self._mode == "within":
            for cluster in self.clusters:
                for _rc in torch.combinations(cluster, r=2):
                    for i, _differences in enumerate(differences):
                        if self._drop_argmax and max_class[i] in _rc:
                            # Skip if the highest probability class is either the row or column of the diff. cell
                            continue
                        diff = _differences[tuple(_rc)]
                        super().__call__(diff)
        elif self._mode == "across":
            for _cx in range(len(self.clusters) - 1):
                for _cy in range(_cx + 1, len(self.clusters)):
                    _x, _y = torch.meshgrid(self.clusters[_cx], self.clusters[_cy])
                    for tup in list(zip(_x.flatten(), _y.flatten())):
                        for i, _differences in enumerate(differences):
                            if self._drop_argmax and max_class[i] in tup:
                                # Skip if the highest probability class is either the row or column of the diff. cell
                                continue
                            diff = _differences[tup]
                            super().__call__(diff)
