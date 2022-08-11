from typing import List, Tuple, Union, Dict, Any, Optional, overload
from overrides import overrides
import torch
from allennlp.data import TextFieldTensors
from structured_prediction_baselines.modules.sampler import Sampler, BasicSampler


@Sampler.register("classification-basic")
class ClassificationBasicSampler(BasicSampler):
    @property
    def is_normalized(self) -> bool:
        return True

    @property
    def different_training_and_eval(self) -> bool:
        return False

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:  # (batch, 1, num_labels)
        if y is not None:
            return torch.softmax(y, dim=-1)
        return None

    @overrides
    def forward(
        self,
        x: Union[torch.Tensor, TextFieldTensors],
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        logits = self.inference_nn(x).unsqueeze(1)  # unnormalized logits (batch, 1, ...)
        loss = None
        if labels is not None:
            # compute loss for logging.
            loss = self.loss_fn(x,
                                labels.unsqueeze(1),  # (batch, num_samples or 1, ...)
                                logits,
                                logits,
                                buffer,
                                )
        return self.normalize(logits), self.normalize(logits), loss