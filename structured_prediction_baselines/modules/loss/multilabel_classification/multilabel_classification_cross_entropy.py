from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from structured_prediction_baselines.modules.loss import Loss


@Loss.register("multi-label-bce")
class MultiLabelBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._total_loss = 0.0

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None
        loss = self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype)).sum(
            dim=-1
        )  # (batch, 1,)
        self._total_loss += loss
        return loss

    def get_metrics(self, reset: bool = False):
        metrics = {'cross_entropy_loss': self._total_loss}
        if reset:
            self._total_loss = 0.0
        return metrics
