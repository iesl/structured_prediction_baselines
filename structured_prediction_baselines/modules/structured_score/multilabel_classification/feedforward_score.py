from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score import (
    StructuredScore,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.common.lazy import Lazy
from allennlp.nn.activations import Activation
from structured_prediction_baselines.modules.task_nn import TaskNN
import torch
import torch.nn.functional as F
import math

@StructuredScore.register("multi-label-feedforward")
class MultilabelClassificationFeedforwardStructuredScore(StructuredScore):
    def __init__(
        self,
        feedforward: FeedForward,
        task_nn: TaskNN
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward = feedforward
        hidden_dim = self.feedforward.get_output_dim()  # type:ignore
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.feedforward(y)  # (batch, num_samples, hidden_dim)
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)
        return score

@StructuredScore.register("multi-label-feedforward-global-v1")
class MultilabelClassificationFeedforwardStructuredScoreGlobalv1(StructuredScore):
    """
    EΘglobal (x, y) = v⊤ MLP([y ; hx] )
    """
    def __init__(
        self,
        feedforward1: FeedForward,
        feedforward2: FeedForward,
        task_nn: TaskNN
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward1 = feedforward1
        self.feedforward2 = feedforward2
        hidden_dim = self.feedforward2.get_output_dim()  # type:ignore
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        
        # x_representation.shape: (batch, D1)
        x_representation = self.task_nn(x, return_hidden_representation = True)        
        batch_size , x_hidden_dim = x_representation.shape

        # y.shape: (batch, num_samples, num_labels)
        batch_size, num_samples, num_labels = y.shape
        # y_representation.shape: (batch, N, D2)
        y_representation = self.feedforward1(y)

        # x_expanded_dim.shape: (batch, 1, D1)
        x_expanded_dim = x_representation.unsqueeze(1)
        # x_broadcasted.shape: (batch, num_samples, D1)
        x_broadcasted = x_expanded_dim.broadcast_to(batch_size, num_samples, x_hidden_dim)
        # x_concat_y.shape: (batch, num_samples, D1+D2)
        x_concat_y = torch.cat((x_broadcasted, y_representation),-1)

        hidden = self.feedforward2(x_concat_y)  # (batch, num_samples, hidden_dim)
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)

        return score

@StructuredScore.register("multi-label-feedforward-global-v2")
class MultilabelClassificationFeedforwardStructuredScoreGlobalv2(StructuredScore):
    """
    EΘglobal (x, y) = vT γ(Mx y)
    """
    def __init__(
        self,
        feedforward: FeedForward,
        task_nn: TaskNN,
        num_labels: int,
        activation: Activation
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward = feedforward
        self.num_labels = num_labels
        self.activation = activation
        self.hidden_dim = self.feedforward.get_output_dim() // num_labels
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / self.hidden_dim), (self.hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        # x_representation.shape: (batch, D)
        x_representation = self.task_nn(x, return_hidden_representation = True)
        # y.shape : (batch, num_samples, num_labels)
        batch_size, num_samples, num_labels = y.shape
        # y_transposed.shape : (batch, L, num_samples)
        y_transposed = torch.transpose(y, 1,2)
        # M_x.shape: (batch, H, num_labels)
        M_x = self.feedforward(x_representation).view(batch_size, self.hidden_dim, num_labels)
        # Mx_y.shape: (batch, num_samples, H)
        Mx_y = torch.bmm(M_x, y_transposed).transpose(1,2)
        Mx_y = self.activation(Mx_y)
        # score.shape: (batch, num_samples)
        score = torch.nn.functional.linear(
            Mx_y, self.projection_vector
        )  # unormalized (batch, num_samples)
        return score

@StructuredScore.register("multi-label-feedforward-global-v3")
class MultilabelClassificationFeedforwardStructuredScoreGlobalv3(StructuredScore):
    """
    EΘglobal (x, y) = v_x T γ(M y)
    """
    def __init__(
        self,
        feedforward1: FeedForward,
        feedforward2: FeedForward,
        task_nn: TaskNN,
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward1 = feedforward1
        self.feedforward2 = feedforward2
        assert self.feedforward1.get_output_dim() == self.feedforward2.get_output_dim()
        self.hidden_dim = self.feedforward1.get_output_dim()

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        x_representation = self.feedforward2(self.task_nn(x, return_hidden_representation = True))
        batch_size, num_samples, num_labels = y.shape
        hidden = self.feedforward1(y)  # (batch, num_samples, hidden_dim)
        score = torch.sum(x_representation.unsqueeze(1) * hidden,2)
        # unormalized (batch, num_samples)
        return score