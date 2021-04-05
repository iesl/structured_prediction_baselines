from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score.structured_score import StructuredScore
import torch
from structured_prediction_baselines.modules.cnn_encoder import Cnn2dEncoder


@StructuredScore.register("cnn")
class CNN(StructuredScore):
    def __init__(self, num_tags: int, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__()
        self.num_tags = num_tags
        self.encoder = Cnn2dEncoder(num_tags, embedding_dim=1, num_filters=50, ngram_filter_sizes=(3,), dropout=0.1)

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        output = self.encoder(y, mask)
        output = output.sum(dim=2)
        output = output * mask.unsqueeze(1)
        return output.sum(dim=2)
