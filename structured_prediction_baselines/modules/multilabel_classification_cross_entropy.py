from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from .loss import Loss


@Loss.register("multi-label-bce")
class MultilabelClassificationBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

    def forward(
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        y_hat_probabilities: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if y_hat.dim() == 3:  # (batch, 1, num_labels)
            # we assume that the extra dim is 1
            y_hat = y_hat.squeeze(1)

        return self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype))
