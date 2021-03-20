from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.loss import DVNLoss, Loss
import torch


@Loss.register("multi-label-dvn-bce")
class DVNCrossEntropy(DVNLoss):
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        # both oracle values and predicted scores are higher the better
        # The oracle value will be something like f1 or 1-hamming_loss
        # which will take values in [0,1] with best value being 1.
        # Predicted score are logits, hence bce with logit will
        # internally map them to [0,1]

        return torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_score,
            oracle_value,
            reduction=self.reduction,
        )
