from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)


class Sampler(torch.nn.Module, Registrable):
    """
    Given input x, returns samples of shape `(batch, num_samples or 1,...)`
    and optionally their corresponding probabilities of shape `(batch, num_samples)`.
    **The sampler can do and return different things during training and test.**
    We want the probabilities specifically in the [[Minimum Risk Training for Neural Machine Translation|MRT setting]].

    The cases that sampler will cover include:
        1. Inference network or `TaskNN`, where we just take the input x and produce either a
            relaxed output of shape `(batch, 1, ...)` or samples of shape `(batch, num_samples, ...)`.
            Note, when we include `TaskNN` here, we also need to update its parameters, right here.
            So when sampler uses `TaskNN`, we also need to give it an instance of `Optimizer` to update its parameters.
        2. Cost-augmented inference module that uses `ScoreNN` and `OracleValueFunction` to produce a single relaxed output or samples.
        3. Adversarial sampler which again uses `ScoreNN` and `OracleValueFunction` to produce adversarial samples.
            (I see no difference between this and the cost augmented inference)
        4. Random samples biased towards `labels`.
        5. In the case of MRT style training, it can be beam search.
        6. In the case of vanilla feedforward model, one can just return the logits with shape `(batch, 1, ... )`
    """

    def __init__(
        self,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self.score_nn = score_nn
        self.oracle_value_function = oracle_value_function
        self._different_training_and_eval = False

    def forward(
        self, x: Any, labels: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            samples: Tensor of shape (batch, num_samples, ...)
            probabilities: None or tensor of shape (batch, num_samples)
        """
        raise NotImplementedError

    @property
    def different_training_and_eval(self) -> bool:
        return self._different_training_and_eval
