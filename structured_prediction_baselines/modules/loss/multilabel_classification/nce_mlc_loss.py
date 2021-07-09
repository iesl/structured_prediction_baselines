from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.loss import Loss, NCERankingLoss
import torch

def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)


@Loss.register("multi-label-nce-ranking")
class MultiLabelNCERanking(NCERankingLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return predicted_score