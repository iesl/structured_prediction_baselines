from typing import List, Tuple, Union, Dict, Any, Optional, overload
from structured_prediction_baselines.modules.sampler import (
    Sampler,
    SamplerModifier,
    InferenceNetSampler,
)
import torch
import torch.nn.functional as F
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sequence_tagging_task_nn import (
    SequenceTaggingTaskNN,
)


@Sampler.register("weizmann-horse-seg-inference-net")
@InferenceNetSampler.register("weizmann-horse-seg-inference-net")
class WeizmannHorseSegInferenceNetSampler(InferenceNetSampler):

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        output = self.inference_nn(x)

        n, c, h, w = output.size()
        log_p = F.log_softmax(output, dim=1)
        # prediction (pick higher probability after log softmax)
        pred = torch.argmax(log_p, dim=1) # (n, h, w)

        log_p = log_p.transpose(1, 2).transpose(2, 3)
        labels = labels.transpose(1, 2).transpose(2, 3)
        log_p = log_p[labels.repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c) # (n*h*w, c)
        labels = labels[labels >= 0] # (n*h*w,)
        loss = F.nll_loss(log_p, labels.long())

        return pred, None, loss

