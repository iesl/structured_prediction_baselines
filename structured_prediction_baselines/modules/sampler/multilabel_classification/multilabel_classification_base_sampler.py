from typing import List, Tuple, Union, Dict, Any, Optional, overload
from structured_prediction_baselines.modules.sampler import (
    Sampler,
    SamplerModifier,
    InferenceNetSampler,
)
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("multi-label-basic")
class MultilabelClassificationSampler(Sampler):
    def __init__(
        self,
        inference_nn: MultilabelTaskNN,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.inference_nn = inference_nn

    @property
    def is_normalized(self) -> bool:
        return False

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return (
            self.inference_nn(x, buffer=buffer).unsqueeze(1),
            None,
            None,
        )  # unormalized logits (batch, 1, ...)


@Sampler.register("multi-label-inference-net-normalized")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized",
)
class MultiLabelNormalized(InferenceNetSampler):
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


@Sampler.register("multi-label-inference-net-normalized-or-sampled")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized-or-sampled"
)
class MultiLabelNormalizedOrSampled(InferenceNetSampler):
    """
    Samples during training and normalizes during evaluation.
    """

    def __init__(
        self, num_samples: int = 1, keep_probs: bool = True, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.keep_probs = keep_probs

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is not None:
            if self.training:
                return self.generate_samples(y)
            else:  # eval
                return y
        else:
            return None

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        assert (
            y.dim() == 3
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        p = torch.sigmoid(y).squeeze(1)  # (batch, num_labels)
        samples = torch.transpose(
            torch.distributions.Bernoulli(probs=p).sample(  # type: ignore
                [self.num_samples]  # (num_samples, batch, num_labels)
            ),
            0,
            1,
        )  # (batch, num_samples, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, p.unsqueeze(1)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True
