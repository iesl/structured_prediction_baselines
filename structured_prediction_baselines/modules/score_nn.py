from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from .task_nn import TaskNN
from .structured_score.structured_score import StructuredScore


class ScoreNN(torch.nn.Module, Registrable):
    """Concrete base class for creating feature representation for any task."""

    def __init__(
        self,
        task_nn: TaskNN,  # (batch, num_samples, ...)
        global_score: Optional[StructuredScore] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type:ignore
        self.task_nn = task_nn
        self.global_score = global_score

    def compute_local_score(
        self, x: Any, y: Any, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        return None

    def compute_global_score(
        self, y: Any, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        if self.global_score is not None:
            return self.global_score(y, **kwargs)
        else:
            return None
