import logging
from typing import List, Tuple, Union, Dict, Any, Optional
from overrides import overrides

import torch
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register(
    "single-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@Model.register(
    "single-label-classification", constructor="from_partial_objects"
)
@ScoreBasedLearningModel.register(
    "single-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@ScoreBasedLearningModel.register(
    "single-label-classification", constructor="from_partial_objects"
)
class SinglelabelClassification(ScoreBasedLearningModel):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # metrics
        self.accuracy = CategoricalAccuracy()

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("x")
        _forward_args["labels"] = kwargs.pop("label")

        return {**_forward_args, **kwargs}

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(1)

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(1)

    @torch.no_grad()
    @overrides
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:

        if not self.inference_module.is_normalized:
            y_hat_n = torch.softmax(y_hat, dim=-1)
        else:
            y_hat_n = y_hat

        self.accuracy(y_hat_n, labels)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self.accuracy.get_metric(reset),
        }

        return metrics
