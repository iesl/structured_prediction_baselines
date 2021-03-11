from typing import List, Tuple, Union, Dict, Any, Optional
from .structured_energy import StructuredEnergy
import torch
from ..cnn_encoder import Cnn2dEncoder


@StructuredEnergy.register("cnn")
class CNN(StructuredEnergy):
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
        mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        output = self.encoder(y, mask)
        output = output.sum(dim=1)
        output = output * mask
        return output.sum(dim=1)
