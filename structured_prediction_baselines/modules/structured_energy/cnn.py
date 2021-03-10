from typing import List, Tuple, Union, Dict, Any, Optional
from .structured_energy import StructuredEnergy
import torch
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@StructuredEnergy.register("cnn")
class CNN(StructuredEnergy):
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
        mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # implement
        pass
