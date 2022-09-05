import logging
from typing import List, Tuple, Union, Dict, Any, Optional
from overrides import overrides
from collections import defaultdict

import torch
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register("classification", constructor="from_partial_objects")
@Model.register("classification-with-infnet", constructor="from_partial_objects_with_shared_tasknn")
@ScoreBasedLearningModel.register("classification", constructor="from_partial_objects")
@ScoreBasedLearningModel.register("classification-with-infnet", constructor="from_partial_objects_with_shared_tasknn")
class Classification(ScoreBasedLearningModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._accuracy = CategoricalAccuracy()

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(1) # (b, n=1, h) # TODO WZ

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze(1)

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("x")
        _forward_args["labels"] = kwargs.pop("label")
        return {**_forward_args, **kwargs}

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
        y_hat_n = y_hat if self.inference_module.is_normalized else torch.softmax(y_hat, dim=-1)
        self._accuracy(y_hat_n.detach(), labels)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset=reset)}
        return metrics