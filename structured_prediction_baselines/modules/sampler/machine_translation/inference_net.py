from typing import overload, Optional, Any, Dict, Tuple

import torch
from allennlp.data import TextFieldTensors
from allennlp.nn import util

from structured_prediction_baselines.modules.sampler import Sampler, InferenceNetSampler


@Sampler.register("machine-translation-inference-net-normalized")
@InferenceNetSampler.register(
    "machine-translation-inference-net-normalized",
)
class MachineTranslationNormalized(InferenceNetSampler):
    @property
    def is_normalized(self) -> bool:
        return True

    @property
    def different_training_and_eval(self) -> bool:
        return False

    def forward(
        self,
        x: Any,
        labels: Optional[
            TextFieldTensors
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        # y_hat is logits here, while y_pred is the actual prediction from beam search
        y_hat, y_pred = self._get_values(
            x, labels, buffer
        )  # (batch_size, 1, ...) Unnormalized

        if labels is not None:
            # compute loss for logging.
            loss = self.loss_fn(
                x,
                labels,  # TextFieldTensors
                y_hat,
                None,
                buffer,
            )
        else:
            loss = None

        return self.normalize(y_pred), None, loss

    def _get_values(
        self,
        x: Any,
        labels: Optional[TextFieldTensors],  # (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        y_inf, y_pred = self.inference_nn(x, labels)
        y_inf = y_inf.unsqueeze(1)  # (batch_size, 1, ...) unormalized
        # inference_nn is TaskNN, so it will output tensor of shape (batch, ...)
        # hence the unsqueeze

        return y_inf, y_pred
