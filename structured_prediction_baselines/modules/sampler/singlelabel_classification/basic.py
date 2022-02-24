from typing import List, Tuple, Union, Dict, Any, Optional, overload
from structured_prediction_baselines.modules.sampler import (
    Sampler,
    BasicSampler,
)
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.loss import Loss

from structured_prediction_baselines.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("single-label-basic")
class SinglelabelClassificationBasicSampler(BasicSampler):
    @property
    def is_normalized(self) -> bool:
        return True

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is not None:
            return torch.softmax(y, dim=-1)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False
