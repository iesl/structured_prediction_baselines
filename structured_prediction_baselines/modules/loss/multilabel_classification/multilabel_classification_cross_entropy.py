from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from structured_prediction_baselines.modules.loss import Loss


@Loss.register("multi-label-bce")
class MultilabelClassificationBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_probabilities: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None

        return self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype))
