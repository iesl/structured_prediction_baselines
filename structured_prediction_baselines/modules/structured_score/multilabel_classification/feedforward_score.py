from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score import (
    StructuredScore,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.common.lazy import Lazy
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
        
        # x_representation.shape: (B, L)
        x_representation = self.task_nn(x)
        batch_size , num_labels = x_representation.shape

        # y.shape: (B, S, L)
        batch_size, num_samples, num_labels = y.shape

        # x_expanded_dim.shape: (B, 1, L)
        x_expanded_dim = x_representation.unsqueeze(1)
        # x_broadcasted.shape: (B, S, L)
        x_broadcasted = x_expanded_dim.broadcast_to(batch_size, num_samples, num_labels)
        # x_concat_y.shape: (B, S, 2L)
        x_concat_y = torch.cat((x_broadcasted, y),-1)

        hidden = self.feedforward(x_concat_y)  # (batch, num_samples, hidden_dim)
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
        task_nn_hidden_dim: int
    ):
        super().__init__(task_nn=task_nn)  # type:ignore
        self.feedforward = feedforward
        self.task_nn_hidden_dim = task_nn_hidden_dim
        self.hidden_dim = self.feedforward.get_output_dim() // task_nn_hidden_dim
        
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
        x_representation = self.task_nn(x)
        batch_size, num_samples, num_labels = y.shape
        y_transposed = torch.transpose(y, 1,2)
        M_x = self.feedforward(x_representation).view(batch_size, self.hidden_dim, num_labels)
        Mx_y = torch.bmm(M_x, y_transposed).transpose(1,2)
        Mx_y = F.softplus(Mx_y)
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
        self.hidden_dim = self.feedforward1.get_output_dim()

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        x: Optional[torch.Tensor] = None, 
        **kwargs: Any,
    ) -> torch.Tensor:
        x_representation = self.feedforward2(self.task_nn(x))
        print(f'x_representation.shape: {x_representation.shape}')
        batch_size, num_samples, num_labels = y.shape
        hidden = self.feedforward1(y)  # (batch, num_samples, hidden_dim)
        score = torch.sum(x_representation.unsqueeze(1) * hidden,2)
        # unormalized (batch, num_samples)
        return score