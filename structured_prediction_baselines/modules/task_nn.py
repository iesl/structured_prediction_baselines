from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch


class TaskNN(torch.nn.Module, Registrable):
    """Base class for creating feature representation for any task.

    Inheriting classes should override the `foward` method.
    """

    def forward(
        self,
        x: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Returns:
            unnormalized_logits: tensor of shape (batch, ...)
        """
        raise NotImplementedError
