from typing import List, Tuple, Union, Dict, Any, Optional
from .structured_energy import StructuredEnergy
import torch


@StructuredEnergy.register("linear-chain")
class LinearChain(StructuredEnergy):
    def __init__(self, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__()
        # TODO: initialize weights
        # self.W =

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # implement
        pass
