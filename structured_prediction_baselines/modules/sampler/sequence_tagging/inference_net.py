from typing import overload, Optional

import torch

from structured_prediction_baselines.modules.sampler import InferenceNetSampler, Sampler


@Sampler.register("sequence-tagging-inference-net-normalized")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized",
)
class SequenceTaggingNormalized(InferenceNetSampler):
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
            return torch.sigmoid(y)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False