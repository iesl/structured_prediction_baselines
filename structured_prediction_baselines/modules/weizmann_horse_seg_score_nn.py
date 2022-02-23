from typing import List, Tuple, Union, Dict, Any, Optional
from .task_nn import TaskNN
from .score_nn import ScoreNN
import torch
from torch import nn
from torch.nn import functional as F


@ScoreNN.register("weizmann-horse-seg")
class WeizmannHorseSegScoreNN(ScoreNN):
    def __init__(self, task_nn: TaskNN, dropout: float = 0.25, **kwargs: Any):
        super().__init__(task_nn, **kwargs)  # type:ignore

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(4, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 1)

        # apply dropout on the first FC layer as paper mentioned
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = torch.cat((image, mask), 1)

        z = self.non_linearity(self.conv1(z))
        z = self.non_linearity(self.conv2(z))
        z = self.non_linearity(self.conv3(z))

        # flatten before FC layers
        z = z.view(-1, 128 * 6 * 6)
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z