import contextlib
import warnings
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

import torch
import numpy as np
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler
from structured_prediction_baselines.modules.stopping_criteria import (
    StoppingCriteria,
    StopAfterNumberOfSteps,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)



@Sampler.register("infnet-nce", constructor="from_partial_objects")
# --> change it to "multi-sample-logprob"
class InfnetRankingNCE(Sampler):
    """
    Implements the infnet that trains Score-NN with multiple samples via Ranking Loss.
    Draw discrete samples (s_i) from Task-NN output (y_hat) and then 
    make a ranking loss between score(y_i)-P_N(y_i) where P_N is probability of the noise model.
    We assume that noise model is provided by the Task-NN. (whereas most NCE method regards fixed Task-NN)
    """
    def __init__(
        self,
        optimizer: Optimizer, 
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Loss,
        num_samples: int = 1,
        keep_labels: bool = True, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn,
            oracle_value_function,
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if isinstance(stopping_criteria, int):
            self.stopping_criteria: StoppingCriteria = StopAfterNumberOfSteps(
                stopping_criteria
            )
        else:
            self.stopping_criteria = stopping_criteria

        # Till here: the same as inference_net.py  
        # After this point: new variables for InfnetMultiSampleLearner
        self.keep_labels = keep_labels
        self.num_samples = num_samples
        # self.logging_children.append(self.loss_fn)
        
    ## copied from "InferenceNetSampler"
    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return False

    @classmethod
    def from_partial_objects(
        cls,
        optimizer: Lazy[Optimizer],
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Lazy[Loss], # first loss is always multi-sample loss.
        num_samples: int = 1,
        keep_labels: bool = True, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ) -> "InfnetMultiSampleLearner":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )        
        trainable_parameters: Dict[str, torch.Tensor] = {}

        trainable_parameters.update(inference_nn.named_parameters())

        if cost_augmented_layer is not None:
            trainable_parameters.update(
                cost_augmented_layer.named_parameters()
            )
        optimizer_ = None
        if optimizer is not None:
            optimizer_ = optimizer.construct(
                model_parameters=list(trainable_parameters.items())
            )

        return cls(
            optimizer=optimizer_,
            inference_nn=inference_nn,
            score_nn=score_nn,
            loss_fn=loss_fn_,
            num_samples=num_samples,
            keep_labels=keep_labels,
            cost_augmented_layer=cost_augmented_layer,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
        )
    
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

    @contextlib.contextmanager
    def only_inference_nn_grad_on(self) -> Generator[None, None, None]:

        # switch off the gradients for score_nn but first cache their requires grad
        assert self.score_nn is not None
        score_nn_requires_grad_map = {
            name: param.requires_grad
            for name, param in self.score_nn.named_parameters()
        }
        try:
            # first switch off all.

            for p in self.score_nn.parameters():
                p.requires_grad_(False)
            # then switch on inf net params

            if self.optimizer:
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        p.requires_grad_(True)
            yield

        finally:
            # set the requires_grad back to false for inf net

            for n, p in self.score_nn.named_parameters():  # type: ignore
                p.requires_grad_(score_nn_requires_grad_map[n])

            if self.optimizer:
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        p.requires_grad_(False)

    def draw_samples(self, y_inf):
        assert y_inf.dim() == 3
        assert y_inf.shape[1] == 1
        y_hat_n = torch.sigmoid(y_inf)

        if self.num_samples>0:
            p = y_hat_n.squeeze(1)  # (batch, num_labels)

            discrete_samples = torch.transpose(
                torch.distributions.Bernoulli(probs=p).sample(
                    [self.num_samples]  # (num_samples, batch, num_labels)
                ),
                0,
                1,
            )  # (batch, num_samples, num_labels)
        else: # if self.num_sample <= 0 then return None.
            discrete_samples = None 

        return discrete_samples
        
    def _get_values(
        self,
        x: Any,
        labels: torch.Tensor,
        buffer: Dict,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        y_inf: torch.Tensor = self.inference_nn(
            x, buffer
        ).unsqueeze(
            1
        )  # (batch_size, 1, ...) unormalized
        # inference_nn is TaskNN so it will output tensor of shape (batch, ...)
        # hence the unsqueeze

        if self.cost_augmented_layer is not None:
            y_cost_aug = self.cost_augmented_layer(
                torch.cat(
                    (
                        y_inf.squeeze(1),
                        labels.to(dtype=y_inf.dtype).squeeze(1),
                    ),
                    dim=-1,
                ),
                buffer,
            ).unsqueeze(
                1
            )  # (batch_size,1, ...)
        else:
            y_cost_aug = None

        return y_inf, y_cost_aug

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if labels is None or (not self.training):
            y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
                1
            )  # (batch_size, 1, ...)
            samples = self.draw_samples(y_inf)
            return y_inf, samples
        else:
            # switch on gradients on the parameters of inference network using context manager
            with self.only_inference_nn_grad_on():
                labels = labels.unsqueeze(1)
                # loss_fn = self.get_loss_fn(
                #     x, labels
                # )  #: Loss function will expect labels in form (batch, num_samples or 1, labels)
                ## This part is removed. 
                loss_values: List[float] = []
                step_number = 0
                loss_value: Union[torch.Tensor, float] = float("inf")

                while not self.stopping_criteria(
                    step_number, float(loss_value)
                ):
                    if self.optimizer:
                        self.optimizer.zero_grad(set_to_none=True)

                    y_inf, y_cost_aug = self._get_values(x, labels, buffer) # 
                    samples = self.draw_samples(y_inf)
                    loss_value = self.update( # made self.update to be the same as Loss class forward()
                        x, labels, samples, y_inf, y_cost_aug, buffer 
                    )
                    loss_values.append(float(loss_value))
                    step_number += 1
                
                # self._metrics[self.name + '_loss'] = np.mean(loss_values)
                # self._total_loss += np.mean(loss_values)
                # self._num_batches += 1

            # once out of Sampler, y_inf and y_cost_aug should not get gradients
            return (
                y_inf.detach().clone(),
                y_cost_aug.detach().clone()
                if y_cost_aug is not None
                else None,
            )    
    
    def update(
        self, 
        x: Any, 
        labels: torch.Tensor, 
        samples: torch.Tensor,
        y_hat: torch.Tensor, 
        y_cost_aug: torch.Tensor, 
        buffer: Dict,
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """

        buffer['samples'] = samples
        total_loss = self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
            ) # (batch, 1, num_labels)
        if self.optimizer:
            total_loss.backward()  # type:ignore
            self.optimizer.step()

        return total_loss


@Sampler.register("infnet-nce-interpolate", constructor="from_partial_objects")
# --> change it to "multi-sample-logprob"
class InfnetRankingNCEInterpolate(InfnetRankingNCE):
    """
    The same class as the NCE ranking loss but provides samples that are interpolated 
    between samples & label. i.e. sampels = (a* samples + b* label)/(a+b) 
    """
    def __init__(
        self,
        optimizer: Optimizer, #loss_fn: Loss,  
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Loss,
        num_samples: int = 1,
        keep_labels: bool = True, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn,
            oracle_value_function,
        )

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if labels is None or (not self.training):
            y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
                1
            )  # (batch_size, 1, ...)
            samples = self.draw_samples(y_inf)
            # Newly added lines on top of InfnetRankingNCE
            samples = (samples + labels)/2
            return y_inf, samples
        else:
            # switch on gradients on the parameters of inference network using context manager
            with self.only_inference_nn_grad_on():
                labels = labels.unsqueeze(1)
                # loss_fn = self.get_loss_fn(
                #     x, labels
                # )  #: Loss function will expect labels in form (batch, num_samples or 1, labels)
                ## This part is removed. 
                loss_values: List[float] = []
                step_number = 0
                loss_value: Union[torch.Tensor, float] = float("inf")

                while not self.stopping_criteria(
                    step_number, float(loss_value)
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    y_inf, y_cost_aug = self._get_values(x, labels, buffer) # 
                    samples = self.draw_samples(y_inf) # (batch, num_samples, num_labels)
                    

                    loss_value = self.update( # made self.update to be the same as Loss class forward()
                        x, labels, samples, y_inf, y_cost_aug, buffer 
                    )
                    loss_values.append(float(loss_value))
                    step_number += 1
                
                self._metrics[self.name + '_loss'] = np.mean(loss_values)
                self._total_loss += np.mean(loss_values)
                self._num_batches += 1

            # once out of Sampler, y_inf and y_cost_aug should not get gradients
            return (
                y_inf.detach().clone(),
                y_cost_aug.detach().clone()
                if y_cost_aug is not None
                else None,
            )    