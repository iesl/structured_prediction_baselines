from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.loss import DVNScoreLoss, DVNScoreCostAugLoss, Loss
import torch


@Loss.register("multi-label-dvn-score")
class MultiLabelDVNScoreLoss(DVNScoreLoss):
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return -torch.sigmoid(predicted_score)


@Loss.register("multi-label-dvn-ca-score")
class MultiLabelDVNScoreCA(DVNScoreCostAugLoss):
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return -torch.sigmoid(predicted_score)
