from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.loss import DVNScoreLoss, Loss
import torch


@Loss.register("multi-label-dvn-score")
class MultilabelDVNScoreLoss(DVNScoreLoss):
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:

        return -torch.sum(torch.sigmoid(predicted_score))
