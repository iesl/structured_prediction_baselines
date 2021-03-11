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
        self.W = nn.Parameter(
                torch.FloatTensor(
                    np.random.uniform(-0.02, 0.02, (self.M, num_tags + 1, num_tags + 1)).astype('float32')))

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:

        batch_size, seq_length, _ = y.shape
        targets = y.transpose(0, 1)  # [seq_length, batch_size, num_tags]
        length_index = mask.sum(1).long() - 1  # [batch_size]

        trans_energy = torch.zeros(batch_size, requires_grad=True)
        prev_labels = []
        for t in range(seq_length):
            energy_t = 0
            target_t = targets[t]  # [batch_size, num_tags]

            if t < self.M:
                prev_labels.append(target_t)
                new_ta_energy = torch.mm(prev_labels[t], self.W[t, -1, :-1].unsqueeze(1)).squeeze(1)
                # [batch_size, num_tags] x [num_tags] -> [batch_size]

                energy_t += new_ta_energy * mask[:, t]  # [batch_size]

                for i in range(t):
                    new_ta_energy = torch.mm(prev_labels[t - 1 - i], self.W[i, :-1, :-1])  # [batch_size, num_tags]
                    energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [batch_size]
            else:
                for i in range(self.M):
                    new_ta_energy = torch.mm(prev_labels[self.M - 1 - i], self.W[i, :-1, :-1])  # [batch_size, num_tags]
                    energy_t += ((new_ta_energy * target_t).sum(1)) * mask[:, t]  # [batch_size]
                prev_labels.append(target_t)
                prev_labels.pop(0)
            trans_energy += energy_t

        for i in range(min(self.M, seq_length)):
            pos_end_target = y[torch.arange(batch_size), length_index - i, :]  # [batch_size, num_tags]
            trans_energy += torch.mm(pos_end_target, self.W[i, :-1, -1].unsqueeze(1)).squeeze(1)  # [batch_size]

        return trans_energy
