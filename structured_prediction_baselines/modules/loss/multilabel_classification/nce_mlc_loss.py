from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.loss import Loss, NCERankingLoss, NCERevRankingLoss, NCERankingInterpolatedLoss, RankingWithoutNoise
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

@Loss.register("multi-label-ncerev-ranking")
class MultiLabelNCERanking(NCERevRankingLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return predicted_score


@Loss.register("multi-label-ranking")
class MultiLabelNCERanking(RankingWithoutNoise):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return predicted_score


@Loss.register("multi-label-nce-ranking-interp")
class MultiLabelNCERanking(NCERankingInterpolatedLoss):
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
        
    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return predicted_score

