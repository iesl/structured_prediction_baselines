from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.cost_function import CostFunction


class Loss(torch.nn.Module, Registrable):
    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        cost_function: Optional[CostFunction] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self.score_nn = score_nn
        self.cost_function = cost_function

    def forward(
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        y_hat_probabilities: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError
