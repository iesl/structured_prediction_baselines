import logging
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.score_nn import ScoreNN
from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register("sequence-tagging", constructor="from_partial_objects")
class SequenceTagging(ScoreBasedLearningModel):
    def __init__(self, num_tags: int, vocab: Vocabulary, sampler: Sampler, loss_fn: Loss,
                 label_namespace: str = "labels", label_encoding: Optional[str] = None,
                 **kwargs: Any):
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        self.num_tags = num_tags

        if not label_encoding:
            raise ConfigurationError(
                "no label_encoding was specified."
            )
        self._f1_metric = SpanBasedF1Measure(
            vocab, tag_namespace=label_namespace, label_encoding=label_encoding
        )

    @classmethod
    def from_partial_objects(
        cls,
        vocab: Vocabulary,
        loss_fn: Lazy[Loss],
        sampler: Lazy[Sampler],
        num_tags: int = 0,
        label_namespace: str = "labels",
        label_encoding: Optional[str] = None,
        inference_module: Optional[Lazy[Sampler]] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> "ScoreBasedLearningModel":

        if num_tags == 0:
            raise ConfigurationError("num_tags can't be zero")

        cost_augmented_layer = TimeDistributed(  # type: ignore
            nn.Sequential(nn.Linear(2 * num_tags, num_tags), nn.Softmax(dim=-1))
        )
        sampler_ = sampler.construct(
            cost_augmented_layer=cost_augmented_layer, score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        # if no seperate inference module is given,
        # we will be using the same sampler

        if inference_module is None:
            inference_module_ = sampler_
        else:
            inference_module_ = inference_module.construct(
                cost_augmented_layer=cost_augmented_layer, score_nn=score_nn,
                oracle_value_function=oracle_value_function
            )

        return cls(
            num_tags=num_tags,
            vocab=vocab,
            label_namespace=label_namespace,
            label_encoding=label_encoding,
            sampler=sampler_,
            loss_fn=loss_fn_,
            oracle_value_function=oracle_value_function,
            score_nn=score_nn,
            inference_module=inference_module_,
            regularizer=regularizer,
            initializer=initializer,
            **kwargs,
        )

    def convert_to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Converts the labels to one-hot if not already"""
        labels = F.one_hot(labels, num_classes=self.num_tags)
        return labels

    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze and turn the labels into one-hot if required"""
        # for mlc the labels already are in shape (batch, num_labels)
        # we just need to unsqueeze

        return labels.unsqueeze(1)

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
        buffer = self.get_extra_args_for_loss(x, labels, **kwargs)

        if labels is not None:
            labels = self.convert_to_one_hot(labels)  # sampler needs one-hot labels of shape (batch, ...)
            y_hat, y_probabilities = self.sampler(x, labels, buffer)  # here y_hat = y_cost_aug
            # (batch, num_samples or 1, ...), (batch, num_samples or 1)
            results["y_hat"] = y_hat
            results["y_probabilities"] = y_probabilities

            # prepare for calculating metrics
            # y_pred is predictions for metric calculations
            # y_hat are for loss computation
            # For some models this two can be different

            if (self.sampler != self.inference_module) or (
                self.inference_module.different_training_and_eval
            ):
                # we have different sampler for training and inference
                # or the sampler behaves differently
                # so we need to run it again.
                # Note: It is vital to set the module in the eval mode
                # ie with .training = False because the implementation
                # checks this
                model_state = self.training
                self.inference_module.eval()
                y_probabilities, _ = self.inference_module(x)
                _, y_pred = torch.max(y_probabilities, -1)
                self.inference_module.train(model_state)
            else:
                # y_pred = buffer["y_inf"]
                _, y_pred = torch.max(y_probabilities, -1)
            # Loss needs one-hot labels of shape (batch, 1, ...)
            labels = self.unsqueeze_labels(labels)
            loss = self.loss_fn(
                x,
                labels,
                y_probabilities,  # y_inf
                y_hat,   # y_cost_aug
                buffer,  # used to compute mask if needed.
            )
            results["loss"] = loss
            self.calculate_metrics(labels, y_probabilities, mask=buffer["mask"])
        else:
            # labels not present. Just predict.
            model_state = self.training
            self.inference_module.eval()
            y_pred, _ = self.inference_module(x)
            self.inference_module.train(model_state)

        results["y_pred"] = y_pred

        return results

    def calculate_metrics(  # type: ignore
        self, labels: torch.Tensor, y_hat: torch.Tensor, mask: torch.tensor = None, **kwargs: Any
    ) -> None:

        if y_hat.dim() == 3:  # (batch, num_samples or 1, num_labels)
            # While calculating metrics, we assume that
            # there aren't multiple samples.
            assert (
                y_hat.shape[1] == 1
            ), f"Incorrect size ({y_hat.shape[1]}) of samples dimension. Expected (1)"
            y_hat = y_hat.squeeze(1)
        # At this point we assume that y_hat is of shape (batch, num_labels)
        # where each entry in [0,1]

        if labels.dim() == 3:
            # we might have added an extra dim to labels
            labels = labels.squeeze(1)

        if mask.dim() == 3:
            # we might have added an extra dim to mask
            mask = mask.squeeze(1)

        self._f1_metric(y_hat, labels, mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        f1_dict = self._f1_metric.get_metric(reset=reset)
        metrics.update({x: y for x, y in f1_dict.items() if "overall" in x})

        return metrics
