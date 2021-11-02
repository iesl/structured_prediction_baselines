from typing import List, Tuple, Union, Dict, Any, Optional

from structured_prediction_baselines.modules.self_attention_encoder import SelfAttentionEncoder
from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
import torch


@StructuredScore.register("self-attention")
class SelfAttention(StructuredScore):
    def __init__(self, num_tags: int, reduction: str = "max", M: int = 0, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__()
        self.num_tags = num_tags
        self.reduction = reduction
        self.M = M
        # self.seq_length = seq_length
        self.attention_layer = SelfAttentionEncoder(1, num_tags, num_tags, num_tags)

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        mask = buffer["mask"]
        batch_size, n_samples, seq_length, _ = y.shape
        attention_mask = torch.BoolTensor(seq_length, seq_length).fill_(False)
        attention_mask.to(device=y.device)
        for i in range(seq_length):
            lower_idx, higher_idx = max(0, i - self.M), min(seq_length, i + self.M + 1)
            attention_mask[i][lower_idx:higher_idx] = True

        attention_output = self.attention_layer(
            y.view(batch_size * n_samples, seq_length, -1),
            attention_mask
        )  # (batch_size * n_samples, seq_length, num_tags)

        attention_output = attention_output.view(
            batch_size, n_samples, seq_length, -1
        ) * mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, n_samples, seq_length, num_tags)

        if self.reduction == "sum":
            return attention_output.sum((2, 3))

        # reduction = "max" (Default)
        return attention_output.amax(dim=3).sum(2)
