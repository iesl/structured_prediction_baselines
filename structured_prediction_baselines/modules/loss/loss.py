from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from allennlp.common.lazy import Lazy


class Loss(torch.nn.Module, Registrable):
    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self.score_nn = score_nn
        self.oracle_value_function = oracle_value_function

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...)
        y_hat_probabilities: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
