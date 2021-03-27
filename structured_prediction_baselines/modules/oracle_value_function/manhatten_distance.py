from typing import Any

import torch

from structured_prediction_baselines.modules.oracle_value_function import OracleValueFunction


class ManhattanDistanceValueFunction(OracleValueFunction):
    """Return oracle value that is based on manhattan distance (L1 Norm)."""

    def compute(self,
                labels: torch.Tensor,
                y_hat: torch.Tensor,
                mask: torch.Tensor = None,
                **kwargs: Any
                ) -> torch.Tensor:
        distance = torch.sum(torch.abs(labels - y_hat), dim=-1)
        distance *= mask
        return torch.sum(distance, dim=-1)
