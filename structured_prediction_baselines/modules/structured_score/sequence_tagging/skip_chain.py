from typing import List, Tuple, Union, Dict, Any, Optional
from .linear_chain import LinearChain
from .structured_energy import StructuredEnergy
import torch
import torch.nn as nn
import numpy as np


@StructuredEnergy.register("skip-chain")
class SkipChain(LinearChain):
    def __init__(self, num_tags: int, M: int, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__(num_tags)
        self.M = M
        self.W = nn.Parameter(
            torch.FloatTensor(np.random.uniform(-0.02, 0.02, (self.M, num_tags + 1, num_tags + 1)).astype('float32')))
