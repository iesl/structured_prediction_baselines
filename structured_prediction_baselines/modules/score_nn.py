from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from .task_nn import TaskNN
from .structured_energy.structured_energy import StructuredEnergy


class ScoreNN(torch.nn.Module, Registrable):
    """Concrete base class for creating feature representation for any task."""

    def __init__(self, task_nn: TaskNN, structured_energy: StructuredEnergy):
        self.task_nn = task_nn
        self.structured_energy = structured_energy
