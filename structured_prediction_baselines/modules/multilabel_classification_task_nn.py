from typing import List, Tuple, Union, Dict, Any, Optional
from .task_nn import TaskNN
from allennlp.modules.feedforward import FeedForward
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
import torch.nn as nn
import torch
import numpy as np


@TaskNN.register("multi-label-classification")
class MultilabelTaskNN(TaskNN):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: FeedForward,
        label_embeddings: Embedding,
    ):
        super().__init__()  # type:ignore
        self.feature_network = feature_network
        self.label_embeddings = label_embeddings
        assert (
            self.label_embeddings.weight.shape[1]
            == self.feature_network.get_output_dim()
        ), (
            f"label_embeddings dim ({self.label_embeddings.weight.shape[1]}) "
            f"and hidden_dim of feature_network"
            f" ({self.feature_network.get_output_dim()}) do not match."
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        features = self.feature_network(x)  # (batch, hidden_dim)
        logits = torch.matmul(features, self.label_embeddings.weight.T)

        return logits  # unormalized logit of shape (batch, num_labels)
