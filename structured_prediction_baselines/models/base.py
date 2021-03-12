from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from allennlp.models import Model
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.cost_function import CostFunction
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.loss import Loss
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.lazy import Lazy
from allennlp.nn import InitializerApplicator, RegularizerApplicator


@Model.register("score-base-learning", constructor="from_partial_objects")
class ScoreBasedLearningModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        cost_function: Optional[CostFunction] = None,
        score_nn: Optional[ScoreNN] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab, regularizer=regularizer)  # type:ignore
        self.sampler = sampler
        self.loss_fn = loss_fn
        self.cost_function = cost_function
        self.score_nn = score_nn

        if initializer is not None:
            initializer(self)

    @classmethod
    def from_partial_objects(
        cls,
        vocab: Vocabulary,
        loss_fn: Lazy[Loss],
        sampler: Lazy[Sampler],
        score_nn: Optional[ScoreNN] = None,
        cost_function: Optional[CostFunction] = None,
    ) -> "ScoreBasedLearningModel":
        sampler_ = sampler.construct(
            score_nn=score_nn, cost_function=cost_function
        )
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, cost_function=cost_function
        )

        return cls(
            vocab,
            sampler_,
            loss_fn_,
            cost_function=cost_function,
            score_nn=score_nn,
        )

    def calculate_metrics(
        self, labels: torch.Tensor, y_hat: torch.Tensor, **kwargs: Any
    ) -> None:
        return None

    def forward(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        meta: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if meta is None:
            meta = {}
        results: Dict[str, Any] = {}
        y_hat, probabilities = self.sampler(x, labels)
        # (batch, num_samples or 1, ...), (batch, num_samples or 1)
        results["y_hat"] = y_hat
        results["sample_probabilities"] = probabilities

        if labels is not None:
            loss = self.loss_fn(x, labels, y_hat, probabilities)
            results["loss"] = loss

            # calculate metrics

            if self.sampler.is_seperate_eval:
                # the sampler behaves differently during training and eval.
                # so we need to run it again.
                self.sampler.eval()
                with torch.no_grad():  # type:ignore
                    y_pred, _ = self.sampler(x)

                self.sampler.train()
            else:
                y_pred = y_hat
            self.calculate_metrics(labels, y_pred)

        return results
