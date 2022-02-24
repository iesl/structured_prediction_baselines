from typing import List, Tuple, Union, Dict, Any, Optional
from .task_nn import TaskNN, CostAugmentedLayer, TextEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
import torch.nn as nn
import torch
import numpy as np


@TaskNN.register("single-label-classification")
class SinglelabelTaskNN(TaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: Union[FeedForward, TextEncoder],
        label_embeddings: Embedding,
    ):
        super().__init__()  # type:ignore
        self.feature_network: Union[FeedForward, TextEncoder] = feature_network
        self.label_embeddings = label_embeddings
        assert (
            self.label_embeddings.weight.shape[1]
            == self.feature_network.get_output_dim()  # type: ignore
        ), (
            f"label_embeddings dim ({self.label_embeddings.weight.shape[1]}) "
            f"and hidden_dim of feature_network"
            f" ({self.feature_network.get_output_dim()}) do not match."
        )

    def forward(
        self,
        x: torch.Tensor,
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        features = self.feature_network(x)  # (batch, hidden_dim)
        logits = torch.matmul(features, self.label_embeddings.weight.T)

        return logits  # unnormalized logits of shape (batch, num_labels)


@TaskNN.register("single-label-text-classification")
class SinglelabelTextTaskNN(SinglelabelTaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: TextEncoder,
        label_embeddings: Embedding,
    ):
        super().__init__(
            vocab=vocab,
            feature_network=feature_network,
            label_embeddings=label_embeddings,
        )
