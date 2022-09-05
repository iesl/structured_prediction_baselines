from typing import List, Tuple, Union, Dict, Any, Optional
from .task_nn import TaskNN, CostAugmentedLayer, TextEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.data import Vocabulary, TextFieldTensors

import torch.nn as nn
import torch
import numpy as np


@TaskNN.register("classification")
class ClassificationTaskNN(TaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: Union[FeedForward, TextEncoder],
        label_embeddings: Embedding,
    ):
        super().__init__()  # type:ignore
        self.feature_network: Union[FeedForward, TextEncoder] = feature_network
        self.label_embeddings = label_embeddings
        assert self.label_embeddings.weight.shape[1] == self.feature_network.get_output_dim(), (
            f"label_embeddings dim ({self.label_embeddings.weight.shape[1]}) "
            f"and hidden_dim of feature_network"
            f" ({self.feature_network.get_output_dim()}) do not match."
        )

    def forward(
        self,
        x: Union[torch.Tensor, TextFieldTensors],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        features = self.feature_network(x)  # (batch, hidden_dim)
        logits = torch.matmul(features, self.label_embeddings.weight.T)

        return logits  # unnormalized logits of shape (batch, num_labels)

