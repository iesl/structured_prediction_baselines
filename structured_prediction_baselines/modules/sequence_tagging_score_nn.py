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
        x: TextFieldTensors,
        y: torch.Tensor,  # (batch, num_samples or 1, seq_len, num_tags)
        mask: torch.BoolTensor,
        buffer: Dict = None,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        score = None

        if buffer is None:
            buffer = {}
        local_score = self.compute_local_score(x, y, buffer=buffer)

        if local_score is not None:
            score = local_score

        if "mask" in buffer:
            mask = buffer["mask"]
        else:
            mask = util.get_text_field_mask(x)
            buffer["mask"] = mask

        global_score = self.compute_global_score(y, mask)  # type: ignore

        if global_score is not None:
            if score is not None:
                score = score + global_score
            else:
                score = global_score

        return score  #: (batch, num_samples)
