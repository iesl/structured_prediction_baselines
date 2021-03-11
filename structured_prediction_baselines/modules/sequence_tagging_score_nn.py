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
        tags: torch.LongTensor,
        buffer: Dict = None,
    ) -> torch.Tensor:
        if buffer is None:
            buffer = {}
        y_hat = self.task_nn(tokens, buffer)

        buffer["y_hat"] = y_hat

        if "mask" in buffer:
            mask = buffer["mask"]
        else:
            mask = util.get_text_field_mask(tokens)
            buffer["mask"] = mask

        return self.structured_energy(tags, y_hat, mask)
