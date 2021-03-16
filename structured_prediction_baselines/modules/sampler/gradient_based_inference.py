from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    Generator,
)
from types import ModuleType
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.task_nn import TaskNN
import torch
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.output_space import OutputSpace
from allennlp.common.registrable import Registrable
from allennlp.training.optimizers import Optimizer
from allennlp.training import optimizers
from allennlp.common.params import Params
from allennlp.common import params
from allennlp.common.lazy import Lazy
import contextlib
import warnings
import numpy as np
import logging


# TODO: Add a general stopping criterion instead of number of gradient steps
# in GradientDescentLoop
# TODO: Return loss values along with trajectory


class StoppingCriteria(Registrable):
    default_implementation = "number-of-steps"

    def __call__(self, step_number: int, loss_value: float) -> bool:
        raise NotImplementedError


@StoppingCriteria.register("number-of-steps")
class StopAfterNumberOfSteps(StoppingCriteria):
    def __init__(self, number_of_steps: int = 10):
        super().__init__()
        self.number_of_steps = number_of_steps

    def __call__(self, step_number: int, loss_value: float) -> bool:
        return step_number >= self.number_of_steps


@contextlib.contextmanager
def disable_log(
    python_modules: List[ModuleType],
) -> Generator[None, None, None]:
    levels = {}
    try:
        for module in python_modules:
            module_logger = logging.getLogger(module.__name__)
            levels[module.__name__] = module_logger.level
            module_logger.setLevel(logging.WARNING)
        yield
    finally:
        # reset back

        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


class GradientDescentLoop(Registrable):
    """
    Performs gradient descent w.r.t input tensor
    """

    default_implementation = "basic"

    def __init__(self, optimizer: Lazy[Optimizer]):
        self.lazy_optimizer = optimizer
        self.active_optimizer: Optional[Optimizer] = None

    def init_optimizer(self, inp: torch.Tensor) -> Optimizer:
        # disable INFO log because we will repeatedly create
        # optimizer and we don't want the creation to flood
        # our logs
        with disable_log([params, optimizers]):
            self.active_optimizer = self.lazy_optimizer.construct(
                model_parameters=[("y", inp)]
            )

        return self.active_optimizer

    def reset_optimizer(self) -> None:
        self.active_optimizer = None

    @contextlib.contextmanager
    def input(
        self, initial_input: torch.Tensor
    ) -> Generator[Optimizer, None, None]:
        """Initialize a new instance of optimzer with wrt input"""
        try:
            yield self.init_optimizer(initial_input)
        finally:
            self.reset_optimizer()

    def update(
        self,
        inp: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            inp: next point after the gradient update
            loss: loss value at the previous point
        """
        # make sure the caller class has turned off requires_grad to everything except
        assert (
            inp.requires_grad
        ), "Input to step should have requires_grad=True"
        inp.grad = None  # zero grad
        loss = loss_fn(inp)
        loss.backward()  # type:ignore
        assert self.active_optimizer is not None
        self.active_optimizer.step()  # this will update `inp`

        return inp, loss

    def __call__(
        self,
        initial_input: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        stop: Union[
            int, Callable[[int, float], bool]
        ],  #: (current_step, current_loss)
        projection_function_: Callable[[torch.Tensor], None],
    ) -> Tuple[List[torch.Tensor], List[float]]:
        initial_input.requires_grad = True
        inp = initial_input
        trajectory: List[torch.Tensor] = [inp.detach().clone()]
        loss_values: List[float] = []
        step_number = 0
        loss_value: Union[torch.Tensor, float] = float("inf")

        if isinstance(stop, int):
            stop = StopAfterNumberOfSteps(stop)
        # we need to enable grad because if the top-level model
        # was being called in a validation loop, the training
        # flag will be False for all modules. This will not allow
        # gradient based inference to progress.
        with torch.enable_grad():
            with self.input(inp):
                while not stop(step_number, float(loss_value)):
                    inp, loss_value = self.update(inp, loss_fn)
                    projection_function_(inp)
                    trajectory.append(inp.detach().clone())
                    loss_values.append(float(loss_value))
                    step_number += 1
            inp.requires_grad = False
        with torch.no_grad():  # type: ignore
            loss_values.append(float(loss_fn(inp)))

        return trajectory, loss_values


GradientDescentLoop.register("basic")(GradientDescentLoop)


class SamplePicker(Registrable):
    default_implementation = "lastn"

    def __call__(
        self, trajectory: List[torch.Tensor], loss_values: List[float]
    ) -> Tuple[List[torch.Tensor], List[float]]:
        raise NotImplementedError


@SamplePicker.register("lastn")
class LastNSamplePicker(SamplePicker):
    def __init__(self, fraction_of_samples_to_keep: float = 1.0):
        self.fraction_of_samples_to_keep = fraction_of_samples_to_keep

    def __call__(
        self, trajectory: List[torch.Tensor], loss_values: List[float]
    ) -> Tuple[List[torch.Tensor], List[float]]:
        cutoff_index = -(
            int(len(trajectory) * self.fraction_of_samples_to_keep)
        )

        return trajectory[cutoff_index:], loss_values[cutoff_index:]


@SamplePicker.register("best")
class BestSamplePicker(SamplePicker):
    def __call__(
        self, trajectory: List[torch.Tensor], loss_values: List[float]
    ) -> Tuple[List[torch.Tensor], List[float]]:
        best_index = int(np.argmin(loss_values))

        return [trajectory[best_index]], [loss_values[best_index]]


@Sampler.register(
    "gradient-based-inference", constructor="from_partial_objects"
)
class GradientBasedInferenceSampler(Sampler):
    def __init__(
        self,
        gradient_descent_loop: GradientDescentLoop,
        loss_fn: Loss,  #: This loss can be different from the main loss
        output_space: OutputSpace,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        sample_picker: SamplePicker = None,
        number_init_samples: int = 1,
        random_mixing_in_init: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.loss_fn = loss_fn
        self.gradient_descent_loop = gradient_descent_loop
        self.stopping_criteria = stopping_criteria
        self.sample_picker = sample_picker or BestSamplePicker()
        self.output_space = output_space
        self.number_init_samples = number_init_samples
        self.random_mixing_in_init = random_mixing_in_init
        self._different_training_and_eval = True

    @classmethod
    def from_partial_objects(
        cls,
        gradient_descent_loop: GradientDescentLoop,
        loss_fn: Lazy[Loss],  #: This loss can be different from the main loss
        output_space: OutputSpace,
        score_nn: Optional[ScoreNN] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        sample_picker: SamplePicker = None,
        number_init_samples: int = 1,
        random_mixing_in_init: float = 0.5,
    ) -> "GradientBasedInferenceSampler":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )

        return cls(
            gradient_descent_loop,
            loss_fn_,
            output_space,
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
            sample_picker=sample_picker,
            number_init_samples=number_init_samples,
            random_mixing_in_init=random_mixing_in_init,
        )

    def get_loss_fn(
        self, x: Any, labels: Optional[torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        # Sampler gets labels of shape (batch, ...), hence this
        # function will get labels of shape (batch*num_init_samples, ...)
        # but Loss expect y or shape (batch, num_samples or 1, ...)
        # Also during eval the loss is different. We inform the loss function
        # using None

        if self.training and (labels is None):
            warnings.warn("Labels should not be None in training mode!")

        def loss_fn(inp: torch.Tensor) -> torch.Tensor:
            return self.loss_fn(
                x,
                (
                    labels.unsqueeze(1)
                    if (self.training and labels is not None)
                    else None
                ),
                inp.unsqueeze(1),
                None,
            )

        return loss_fn

    def get_dtype_device(self) -> Tuple[torch.dtype, torch.device]:
        for param in self.loss_fn.parameters():
            dtype = param.dtype
            device = param.device

            break

        return dtype, device

    def get_batch_size(self, x: Any) -> int:
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        else:
            raise NotImplementedError

    def get_initial_output(
        self, x: Any, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype, device = self.get_dtype_device()

        if labels is None:
            samples = self.output_space.get_random_samples(
                (self.get_batch_size(x), self.number_init_samples),
                device=device,
                dtype=dtype,
            )  # (batch, num_init_samples, ...)
        else:
            samples = self.output_space.get_mixed_samples(
                self.number_init_samples,
                dtype=dtype,
                reference=labels,
                device=device,
            )  # (batch, num_init_samples,...)

        return samples.flatten(0, 1)  # (batch*num_init_samples, ...)

    @contextlib.contextmanager
    def no_param_grad(self) -> Generator[None, None, None]:

        if self.loss_fn.score_nn is not None:
            # cache the requires_grad of all params before setting them off
            requires_grad_map = {
                name: param.requires_grad
                for name, param in self.loss_fn.named_parameters()
            }
            try:
                for param in self.loss_fn.parameters():
                    param.requires_grad = False
                yield
            finally:
                # set the requires_grad of all params to original

                for n, p in self.loss_fn.named_parameters():
                    p.requires_grad = requires_grad_map[n]
        else:  # if there is no loss_fn, we have nothing to do.
            warnings.warn(
                (
                    "There is no score_nn on loss_fn in gradient based inference sampler."
                    " Are you using the right sampler?"
                )
            )
            try:
                yield
            finally:
                pass

    def get_samples_from_trajectory(
        self, trajectory: List[torch.Tensor], loss_values: List[float]
    ) -> torch.Tensor:
        samples_to_keep, loss_values_to_keep = self.sample_picker(
            trajectory, loss_values
        )
        num_samples = len(samples_to_keep)
        temp = torch.stack(
            samples_to_keep, dim=1
        )  # (batch*num_init_samples, num_samples, ...)
        shape = temp.shape

        return temp.reshape(
            shape[0] // self.number_init_samples,
            self.number_init_samples * num_samples,
            *shape[2:],
        )  # (batch, num_init_samples*num_samples, ...)

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ] = None,  #: If given will have shape (batch, ...)
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        init = self.get_initial_output(
            x, labels
        )  # (batch*num_init_samples, ...)
        # we have to reshape labels from (batch, ...) to (batch*num_init_samples, ...)

        if labels is not None:
            labels = torch.repeat_interleave(
                labels, self.number_init_samples, dim=0
            )
        # switch of gradients on parameters using context manager
        with self.no_param_grad():

            loss_fn = self.get_loss_fn(
                x, labels
            )  #: Loss function will expect labels in form (batch, num_samples or 1, ...)
            trajectory, loss_values = self.gradient_descent_loop(
                init,
                loss_fn,
                self.stopping_criteria,
                self.output_space.projection_function_,
            )

            if labels is not None:  # add groud truth to samples
                trajectory.append(labels.to(dtype=init.dtype))
                loss_values.append(float(loss_fn(labels.to(dtype=init.dtype))))
                assert len(trajectory) == len(loss_values)

        # print(f"\nloss_values:\n{loss_values}")

        return self.get_samples_from_trajectory(trajectory, loss_values), None
