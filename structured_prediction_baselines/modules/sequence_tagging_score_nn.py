from typing import List, Tuple, Union, Dict, Any, Optional
from .score_nn import ScoreNN
import torch
from allennlp.data import TextFieldTensors, Vocabulary
import allennlp.nn.util as util


@ScoreNN.register("seq-tagging")
class SequenceTaggingScoreNN(ScoreNN):
    def forward(
        self,
        tokens: TextFieldTensors,
        y_input: torch.LongTensor,
        y_hat: torch.LongTensor,
        buffer: Dict = None,
    ) -> torch.Tensor:
        if buffer is None:
            buffer = {}

        y_local = buffer.get("y_local")
        if y_local is None:
            y_local = self.task_nn(tokens, buffer)
            buffer["y_local"] = y_local

        mask = buffer.get("mask")
        if mask is None:
            mask = util.get_text_field_mask(tokens)
            buffer["mask"] = mask

        local_energy = torch.sum(y_local * y_input, dim=-1)
        local_energy = torch.sum(local_energy * mask, dim=-1)

        global_energy = self.structured_energy(y_hat, mask)

        return local_energy + global_energy
