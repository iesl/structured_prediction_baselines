from typing import Any

import torch

from structured_prediction_baselines.modules.oracle_value_function import OracleValueFunction


@OracleValueFunction.register("manhattan")
class ManhattanDistanceValueFunction(OracleValueFunction):
    """Return oracle value that is based on manhattan distance (L1 Norm)."""

    def compute(self,
                labels: torch.Tensor,
                y_hat: torch.Tensor,
                **kwargs: Any
                ) -> torch.Tensor:
        distance = torch.sum(torch.abs(labels - y_hat), dim=-1)
        if "mask" not in kwargs:
            raise RuntimeWarning("Evaluating L1 Distance without mask. Make sure if mask is needed or not.")
        else:
            mask = kwargs["mask"]
            if mask.dim() == 3:
                # we might have added an extra dim to mask
                mask = mask.squeeze(1)
            distance *= mask
        return torch.sum(distance, dim=-1)
