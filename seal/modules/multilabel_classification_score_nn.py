from typing import List, Tuple, Union, Dict, Any, Optional
from .score_nn import ScoreNN
import torch


@ScoreNN.register("multi-label-classification")
class MultilabelClassificationScoreNN(ScoreNN):
    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        if self._cache and self._cached:  # (1,1): use cache
            assert (
                not self.training
            ), "caching is not supported in training. Do model.eval()"
            label_scores = self.feature_cache
        elif not self._cached:  # (any, 0) # compute and set cache if specified
            label_scores = self.task_nn(
                x, buffer
            )  # unormalized logit of shape (batch, num_labels)

            if self._cache:  # (1, 0) # set the cache
                self.feature_cache = label_scores
                self._cached = True
            else:  # (0, 0) # do nothing as caching is not enabled
                pass
        else:
            raise RuntimeError(
                "Inconsistent state: cache=False but cached=True. "
            )

        local_energy = torch.sum(
            label_scores.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy
