from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch


class CostFunction(torch.nn.Module, Registrable):
    """
    Either a differentiable (w.r.t y) or non-differentiable function that takes in true label
    and an set of arbitrary y's(either discrete in case of non-differentiable cost) or
    a continuous relaxations. The shape of input y will be (batch, num_samples or 1, ...).
    """

    def forward(
        self, labels: torch.Tensor, y_hat: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, float]:
        raise NotImplementedError
