from typing import List, Tuple, Union, Dict, Any, Optional
from structured_prediction_baselines.modules.sampler import Sampler
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.cost_function import CostFunction
from structured_prediction_baselines.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("multi-label-basic")
class MultilabelClassificationSampler(Sampler):
    def __init__(
        self,
        task_nn: MultilabelTaskNN,
        score_nn: Optional[ScoreNN] = None,
        cost_function: Optional[CostFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn,
            cost_function,
        )
        self.task_nn = task_nn

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.task_nn(x), None
