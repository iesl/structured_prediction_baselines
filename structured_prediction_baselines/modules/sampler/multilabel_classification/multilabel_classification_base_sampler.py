from typing import List, Tuple, Union, Dict, Any, Optional
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return (
            self.inference_nn(x, buffer=buffer).unsqueeze(1),
            None,
        )  # unormalized logits (batch, 1, ...)


@Sampler.register(
    "multi-label-inference-net-normalized", constructor="from_partial_objects"
)
class MultiLabelNormalized(InferenceNetSampler):
    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Note:
            This will only normalize the final output that goes out of the sampler.
            The internal learning loop of the tasknn will use unnormalized outputs only.
            Hence, the losses that train the tasknn should expect unnormalized outputs.
        """
        y_hat, y_hat_extra = super().forward(x, labels, buffer, **kwargs)

        return torch.sigmoid(y_hat), (
            torch.sigmoid(y_hat_extra) if y_hat_extra is not None else None
        )

    @property
    def is_normalized(self) -> bool:
        return True

    @property
    def different_training_and_eval(self) -> bool:
        return False


@Sampler.register(
    "multi-label-inference-net-normalized-or-sampled",
    constructor="from_partial_objects",
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

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        samples, samples_extra = super().forward(x, labels, buffer, **kwargs)
        assert samples.dim() == 3
        assert samples.shape[1] == 1
        y_hat_n = torch.sigmoid(samples)
        y_hat_extra_n = (
            torch.sigmoid(samples_extra) if samples_extra is not None else None
        )

        if self.training:  # sample during training
            p = y_hat_n.squeeze(1)  # (batch, num_labels)

            y_hat_n = torch.transpose(
                torch.distributions.Bernoulli(probs=p).sample(
                    [self.num_samples]  # (num_samples, batch, num_labels)
                ),
                0,
                1,
            )  # (batch, num_samples, num_labels)

            if self.keep_probs:
                y_hat_n = torch.cat(
                    (y_hat_n, y_hat_n), dim=1
                )  # (batch, num_samples+1, num_labels)

        return y_hat_n, y_hat_extra_n

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True


InferenceNetSampler.register(
    "multi-label-inference-net-normalized", constructor="from_partial_objects"
)(MultiLabelNormalized)
