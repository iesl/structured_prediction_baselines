from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Iterator,
    cast,
    Generator,
)
import contextlib
import torch
from allennlp.models import Model
from structured_prediction_baselines.modules.sampler import (
    Sampler,
    SamplerContainer,
    AppendingSamplerContainer,
)
from structured_prediction_baselines.common import ModelMode
from structured_prediction_baselines.modules.sampler.inference_net import (
    InferenceNetSampler,
)
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.loss import Loss
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.lazy import Lazy
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from structured_prediction_baselines.modules.logging import (
    LoggingMixin,
    LoggedValue,
    LoggedScalarScalar,
    LoggedScalarScalarSample,
    LoggedNPArrayNPArraySample,
)
from structured_prediction_baselines.modules.task_nn import TaskNN
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@Model.register("score-based-learning", constructor="from_partial_objects") # energy prediction networks
@Model.register(
    "score-based-learning-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn"
) # task feedforward networks and seal
class ScoreBasedLearningModel(LoggingMixin, Model):
    """
    The model supports three types of forward
        1. By calling `forward_on_tasknn(x, labels)`,
            the inference_module is run in "inference" mode to produce predictions,
            and if labels present, loss; then if labels present, metrics are calculated.
            This will be used to update the parameters of the inference_module and to produce test-time predictions.
        2. By calling `forward_on_scorenn()`,
            the sampler is run in "sample" mode with no_grad and logging turned off to produce sampled y for
            training ScoreNN; then ScoreNN loss is computed.
            This will be used to update the parameters of the ScoreNN.
        3. By calling `compute_score(x,y)`,
            the score for (x,y) will be computed. This is useful for doing custom evaluations of ScoreNN.
            In order for such evaluations to not interfere with the training of ScoreNN,
            we need to do these after the optimizer step for ScoreNN.
            Hence, we will use on_batch or on_epoch callback for this.
            All such evaluations should log values in their own attributes,
            and it is their responsibility to add these values to `metrics` so that
            they can be logged to wandb and console.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,  # loss to optimize ScoreNN
        oracle_value_function: Optional[OracleValueFunction] = None,
        score_nn: Optional[ScoreNN] = None,
        inference_module: Optional[Sampler] = None,
        evaluation_module: Optional[Sampler] = None,
        num_eval_samples: int = 10,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer, **kwargs)
        self.oracle_value_function = oracle_value_function
        self.score_nn = score_nn
        self.sampler = sampler
        self.loss_fn = loss_fn

        self.inference_module = inference_module if inference_module is not None else sampler
        self.evaluation_module = evaluation_module
        self.num_eval_samples = num_eval_samples
        # self.eval_only_metrics = {}

        if initializer is not None:
            initializer(self)
        self.logging_children.append(self.loss_fn)
        self.logging_children.append(self.sampler)
        self.logging_children.append(self.inference_module)
        if evaluation_module is not None:
            self.logging_children.append(self.evaluation_module)

        mode = ModelMode.UPDATE_SCORE_NN
        if self.score_nn is not None:
            for param in self.score_nn.parameters():
                mode.mark_parameter_with_model_mode(param)

        mode = ModelMode.UPDATE_TASK_NN
        if inference_module is not None:
            for param in self.inference_module.parameters_with_model_mode(mode):
                mode.mark_parameter_with_model_mode(param)

        for n, p in self.named_parameters():
            if not ModelMode.hasattr_model_mode(p):
                logger.warning(f"{n} does not have ModelMode set.")


    @classmethod
    def from_partial_objects(
        cls,
        vocab: Vocabulary,
        sampler: Lazy[Sampler],
        loss_fn: Lazy[Loss],
        inference_module: Optional[Lazy[Sampler]] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        evaluation_module: Optional[Lazy[Sampler]] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> "ScoreBasedLearningModel":

        if oracle_value_function is not None:
            sampler_ = sampler.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            sampler_ = sampler.construct(
                score_nn=score_nn,
            )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn,
            )

        if inference_module is None:
            # if no seperate inference module is given, use the sampler for test-time inference.
            inference_module_ = sampler_
        else:
            inference_module_ = inference_module.construct(
                score_nn=score_nn,
                oracle_value_function=oracle_value_function,
                main_sampler=sampler_,
            )

        if evaluation_module is not None:
            evaluation_module_ = evaluation_module.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            evaluation_module_ = None

        return cls(
            vocab=vocab,
            sampler=sampler_,
            loss_fn=loss_fn_,
            oracle_value_function=oracle_value_function,
            score_nn=score_nn,
            inference_module=inference_module_,
            evaluation_module=evaluation_module_,
            regularizer=regularizer,
            initializer=initializer,
            **kwargs,
        )


    @classmethod
    def from_partial_objects_with_shared_tasknn(
        cls,
        vocab: Vocabulary,
        loss_fn: Lazy[Loss],
        inference_module: Lazy[Sampler],
        task_nn: TaskNN,
        sampler: Optional[Lazy[SamplerContainer]] = None,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        evaluation_module: Optional[Lazy[Sampler]] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> "ScoreBasedLearningModel":
        """
        To use this constructor, `sampler` must be an instance of `SamplerContainer`.
        We share the `task_nn` between a sampler in the `SamplerContainer` and the `inference_module`.
        """

        if oracle_value_function is not None:
            if sampler is None:
                sampler_ = AppendingSamplerContainer(
                    score_nn=score_nn,
                    oracle_value_function=oracle_value_function,
                    constituent_samplers=[],
                    log_key="sampler",
                )
            else:
                sampler_ = sampler.construct(
                    score_nn=score_nn,
                    oracle_value_function=oracle_value_function,
                )
            loss_fn_ = loss_fn.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            if sampler is None:
                sampler_ = AppendingSamplerContainer(
                    score_nn=score_nn,
                    constituent_samplers=[],
                    log_key="sampler",
                )
            else:
                sampler_ = sampler.construct(score_nn=score_nn)
            loss_fn_ = loss_fn.construct(score_nn=score_nn)

        # add an infnet sampler
        sampler_.append_sampler(inference_module.construct(
            inference_nn=task_nn,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        ))

        # test-time inference_module which shares task_nn weights with the infnet sampler
        inference_module_ = inference_module.construct(
            inference_nn=task_nn,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
        )
        inference_module_.log_key = inference_module_.log_key + "_inf"

        if evaluation_module is not None:
            evaluation_module_ = evaluation_module.construct(
                score_nn=score_nn, oracle_value_function=oracle_value_function
            )
        else:
            evaluation_module_ = None

        return cls(
            vocab=vocab,
            sampler=sampler_,
            loss_fn=loss_fn_,
            oracle_value_function=oracle_value_function,
            score_nn=score_nn,
            inference_module=inference_module_,
            evaluation_module=evaluation_module_,
            regularizer=regularizer,
            initializer=initializer,
            **kwargs,
        )


    def parameters_for_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        for param in self.parameters():
            if mode.is_parameter_model_mode(param):
                yield param


    def named_parameters_for_model_mode(
        self, mode: ModelMode
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        for name, param in self.named_parameters():
            param = cast(torch.nn.Parameter, param)
            if mode.is_parameter_model_mode(param):
                yield (name, param)


    def convert_to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Converts labels to one-hot before passing them to sampler and inference_module if needed.
        Required for more complex tasks like sequence tagging.
        """
        return labels


    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unsqueeze to add a sample dimension before inputting into score_nn's loss"""
        return labels


    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        """Squeeze to remove the sample dimension in inference_module outputs"""
        raise NotImplementedError


    def forward_on_tasknn(  # type: ignore
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # print("\n==================FORWARD TASKNN======================")

        results: Dict[str, Any] = {}
        if labels is not None:
            labels = self.convert_to_one_hot(labels)

        with self.inference_module.mode("inference"):
            y_pred, _, loss = self.inference_module(x, labels=labels, buffer=buffer)

        results["loss"] = loss
        results["y_pred"] = self.squeeze_y(y_pred)
        if labels is not None:
            self.calculate_metrics(x, labels, results["y_pred"], buffer, results)

        return results


    def forward_on_scorenn(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # print("\n==================FORWARD SCORENN======================")

        results: Dict[str, Any] = {}
        assert labels is not None
        labels = self.convert_to_one_hot(labels)

        with torch.no_grad():
            with self.sampler.mode("sample"):
                y_hat, y_hat_extra, sampler_loss = self.sampler(x, labels=labels, buffer=buffer) # generate samples
        loss = self.loss_fn(x, self.unsqueeze_labels(labels), y_hat, y_hat_extra, buffer)

        results["y_hat"] = y_hat
        results["y_hat_extra"] = y_hat_extra
        results["loss"] = loss

        return results


    def compute_score(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples or 1, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert self.score_nn is not None
        return self.score_nn(x, y, buffer, **kwargs)  # (batch, num_samples or 1)


    def _forward(
        self,
        x: Any,
        labels: torch.Tensor,
        mode: Optional[ModelMode] = ModelMode.UPDATE_TASK_NN,
        **kwargs: Any,
    ) -> Dict:
        if mode == ModelMode.UPDATE_TASK_NN or mode is None:
            results = self.forward_on_tasknn(x, labels, **kwargs)
        elif mode == ModelMode.UPDATE_SCORE_NN:
            results = self.forward_on_scorenn(x, labels, **kwargs)
        elif mode == ModelMode.COMPUTE_SCORE:
            score = self.compute_score(x, labels, **kwargs)
            results = {"score": score}
        else:
            raise ValueError
        return results


    def initialize_buffer(self, **kwargs: Any) -> Dict:
        return {}


    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        kwargs["buffer"] = self.initialize_buffer(**kwargs)
        return kwargs


    def forward(self, **kwargs: Any) -> Dict:
        return self._forward(**self.construct_args_for_forward(**kwargs))


    def calculate_metrics(
        self,
        x: Any,
        labels: torch.Tensor,  # shape: (batch, ...)
        y_hat: torch.Tensor,  # shape: (batch, ...)
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        return None


    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        non_metrics: Dict[str, Union[float, int]] = self.get_all(
            reset=reset, type_=(LoggedScalarScalar,)
        )
        metrics = self.get_true_metrics(reset=reset)
        return {**metrics, **non_metrics}

