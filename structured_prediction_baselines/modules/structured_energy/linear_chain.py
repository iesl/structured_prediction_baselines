from typing import List, Tuple, Union, Dict, Any, Optional
from .structured_energy import StructuredEnergy
import torch
import torch.nn as nn
import numpy as np


@StructuredEnergy.register("linear-chain")
class LinearChain(StructuredEnergy):
    def __init__(self, num_tags: int, **kwargs: Any):
        """
        TODO: Change kwargs to take hidden size and output size
        """
        super().__init__()
        self.M = 1
        self.num_tags = num_tags
        # TODO: initialize weights
        self.W = nn.Parameter(
                torch.FloatTensor(np.random.uniform(-0.02, 0.02, (self.M, num_tags + 1, num_tags + 1)).astype('float32')))

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # implement
        B, T, C = y.shape
        targets = y.transpose(0, 1)  # [T, B, C]
        length_index = mask.sum(1).long() - 1  # [B]

        trans_energy = torch.zeros(B, requires_grad=True)
        prev_labels = []
        for t in range(T):  # M=1 : Linear; M>1: Skip-Chain
            energy_t = 0
            target_t = targets[t]  # [B, C]
            if t < self.M:
                prev_labels.append(target_t)
                new_ta_energy = torch.mm(prev_labels[t], self.W[t, -1, :-1].unsqueeze(1)).squeeze(1)  # [B, C] x [C] -> [B]
                energy_t += new_ta_energy * mask[:, t]  # [B]
                for i in range(t):
                    new_ta_energy = torch.mm(prev_labels[t - 1 - i], self.W[i, :-1, :-1])  # [B, C]
                    energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
            else:
                for i in range(self.M):
                    new_ta_energy = torch.mm(prev_labels[self.M - 1 - i], self.W[i, :-1, :-1])  # [B, C]
                    energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [B]
                prev_labels.append(target_t)
                prev_labels.pop(0)
            trans_energy += energy_t

        for i in range(min(self.M, T)):
            pos_end_target = y[torch.arange(B), length_index - i, :]  # [B, C]
            trans_energy += torch.mm(pos_end_target, self.W[i, :-1, -1].unsqueeze(1)).squeeze(1)  # [B]
        return trans_energy
