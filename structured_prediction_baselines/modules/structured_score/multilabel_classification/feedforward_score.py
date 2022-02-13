from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.structured_score import (
    StructuredScore,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.common.lazy import Lazy
from structured_prediction_baselines.modules.task_nn import TaskNN
import torch
import math

@StructuredScore.register("multi-label-feedforward")
class MultilabelClassificationFeedforwardStructuredScore(StructuredScore):
    def __init__(
        self,
        feedforward: FeedForward,
        task_nn: TaskNN
    ):

        super().__init__(task_nn)  # type:ignore
        self.feedforward = feedforward
        hidden_dim = self.feedforward.get_output_dim()  # type:ignore
        self.projection_vector = torch.nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim,))
        )  # c2 -> shape (hidden_dim,)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.feedforward(y)  # (batch, num_samples, hidden_dim)
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)

        return score
    
    # @classmethod
    # def constructor(
    #     self,
    #     task_nn:TaskNN
    #     ):
    #     self.task_nn = task_nn

@StructuredScore.register("multi-label-feedforward-global-new")
class MultilabelClassificationFeedforwardStructuredScoreGlobalNew(StructuredScore):
    def __init__(
        self,
        feedforward: FeedForward,
    ):
        super().__init__()  # type:ignore
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
        
        # print("********************")
        # print("Running the forward of MultilabelClassificationFeedforwardStructuredScore")
        # print(f"x_representation.shape: {x_representation.shape}")
        # print(f'y.shape: {y.shape}')
        x_representation = self.task_nn(x)
        batch_size , num_labels = x_representation.shape
        batch_size, num_samples, num_labels = y.shape

        x_expanded_dim = x_representation.unsqueeze(1)
        x_broadcasted = x_expanded_dim.broadcast_to(batch_size, num_samples, num_labels)

        x_concat_y = torch.cat((x_broadcasted, y),-1)

        hidden = self.feedforward(x_concat_y)  # (batch, num_samples, hidden_dim)
        # print(f"hidden.shape: {hidden.shape}")
        score = torch.nn.functional.linear(
            hidden, self.projection_vector
        )  # unormalized (batch, num_samples)
        # print(f"score.shape: {score.shape}")

        # print("********************")

        return score
