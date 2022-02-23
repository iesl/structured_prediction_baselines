import logging
from typing import Any, Optional, Dict, List, Tuple
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from structured_prediction_baselines.metrics import SegIoU
from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.sampler import Sampler
from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register("seal-weizmann-horse-seg", constructor="from_partial_objects_with_shared_tasknn")
class WeizmannHorseSegModel(ScoreBasedLearningModel):

    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        **kwargs: Any,
    ):
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        self.instantiate_metrics()

    def instantiate_metrics(self) -> None:
        self._seg_iou = SegIoU()

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze to get the samples dimension"""

        return labels
        # return labels.unsqueeze(-4) # last three dimensions are image dimensions

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        """Squeeze the samples dimension"""

        return y
        # return y.squeeze(-4) # last three dimensions are image dimensions

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("image")
        _forward_args["labels"] = kwargs.pop("mask")
        return {**_forward_args, **kwargs}

    @overrides
    def calculate_metrics(
        self,
        x: Any,
        labels: torch.Tensor,  # model ground truth, shape: (batch, ...)
        y_hat: torch.Tensor,  # model prediction, shape: (batch, ...)
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        self._seg_iou(y_hat.detach(), labels.long()) # TODO detach

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"seg_iou": self._seg_iou.get_metric(reset=reset)}

