from typing import List, Tuple, Union, Dict, Any, Optional
from .score_nn import ScoreNN
import torch
from allennlp.data import TextFieldTensors, Vocabulary
import allennlp.nn.util as util


@ScoreNN.register("seq-tagging")
class SequenceTaggingScoreNN(ScoreNN):
    def compute_local_score(  # type:ignore
        self,
        x: TextFieldTensors,
        y: torch.Tensor,  #: shape (batch, num_samples or 1, seq_len, num_tags)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            y: tensor of labels of shape (batch, seq_len, tags)
        """
        y_hat = self.task_nn(x, buffer=buffer)  # (batch, seq_len, tags)

        if "mask" in buffer:
            mask = buffer["mask"]
        else:
            mask = util.get_text_field_mask(x)
            buffer["mask"] = mask

        assert False, "TODO: incorporate mask here"
        local_score = torch.sum(
            y_hat.unsqueeze(1) * y, dim=(-2, -1)
        )  # (batch, num_samples)

        return local_score

    def forward(
        self,
        tokens: TextFieldTensors,
        y_input: torch.LongTensor,
        y_hat: torch.LongTensor,
        buffer: Dict = None,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        score = None

        if buffer is None:
            buffer = {}
        local_score = self.compute_local_score(x, y, buffer=buffer)

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
