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
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer

from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
from structured_prediction_baselines.modules.sampler import Sampler, InfnetMultiSampleLearner
from structured_prediction_baselines.modules.stopping_criteria import (
    StoppingCriteria,
    StopAfterNumberOfSteps,
)
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)

@Sampler.register("infnet-multi-sample-reducefix", constructor="from_partial_objects")
class InfnetMultiSampleProbReduceFix(InfnetMultiSampleLearner):
    """
    Makes the InfnetMultiSampleLearner to work probability loss rather than logP. 
    i.e. loss_function: g * binary_cross_entropy(labels=s_i,y_hat) 
                    --> g * torch.exp(binary_cross_entropy(labels=s_i,y_hat))
    
    Also made reducing sequence change. 

    In InfnetMultiSampleLearner:
        We were taking "mean" reducing in the get_sample_grads(), 
        and add "sum" of sample_loss on top of "mean" of total loss.
    Now:
        We are not taking any "mean" reducing before the end.
        When adding the sample_loss, we only take torch.mean() across samples,
        but the rest of reduce happens after all the addition of the losses.
    """
    def __init__(
        self,
        optimizer: Optimizer, #loss_fn: Loss,  
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Loss,
        loss_fn_sample: Optional[Loss] = None, 
        loss_fn_for_grad: Optional[Loss] = None,
        sample_loss_weight: Optional[float] = 1.0,
        num_samples: int = 1,
        keep_probs: bool = False, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            optimizer,
            inference_nn,
            score_nn,
            loss_fn,
            loss_fn_sample,
            loss_fn_for_grad,
            sample_loss_weight,
            num_samples,
            keep_probs,
            cost_augmented_layer,
            oracle_value_function,
            stopping_criteria
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
        total_loss = self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
            ) # (batch, 1, num_labels)
        if self.num_samples>0:
            loss_samples = self.loss_fn_sample(
                    x,
                    samples,
                    y_hat.expand_as(samples),
                    None, # y_cost_aug shouldn't be calculated.
                    buffer,
                ) # (batch, num_samples, num_labels)
            
            # only added line in InfnetMultiSampleProb.
            sample_prob = torch.exp(loss_samples)
            
            grad_samples = self.get_sample_grads(
                    x,
                    samples,
                    buffer,
            ) # (batch, num_samples, num_labels) grab gradients w.r.t.samples from score loss.

            loss_samples = grad_samples*sample_prob
            self._metrics["sampling_loss"] = float(torch.mean(loss_samples))
            total_loss = torch.mean(
                    total_loss + torch.mean(self.sample_loss_weight * loss_samples, dim=1, keepdim=True)
            )
        total_loss.backward()  # type:ignore
        # y_inf.expand_as(pseudo_labels)
        self.optimizer.step()

        return total_loss

    def get_sample_grads(
        self, 
        x: Any, 
        samples: torch.Tensor,
        buffer: Dict,
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """
        
        samples.requires_grad = True
        assert (
            samples.requires_grad
        ), "Samples to grab gradient should have requires_grad=True"
        # if I get error here, I should fix it later.

        with torch.enable_grad():
            samples.grad = None # --> maybe not needed
            loss_for_grad = self.loss_fn_for_grad(
                x,
                None, # labels not required to get score-NN score.
                samples,
                None, # y_cost_aug shouldn't be used for gradient grabbing.
                buffer,
            )
            loss = torch.sum(loss_for_grad) 
            grad_samples = torch.autograd.grad(outputs=loss, inputs=samples, only_inputs=True   )

        return grad_samples[0].clone().detach() # grad returns tuple. list of length 1.



@Sampler.register("infnet-multi-sample-real-reducefix", constructor="from_partial_objects")
class InfnetMultiSampleReduceFix(InfnetMultiSampleProbReduceFix):
    """
    Made reducing sequence change from InfnetMultiSampleLearner. 
    (But the same as InfnetMultiSampleProbReduceFix )

    In InfnetMultiSampleLearner:
        We were taking "mean" reducing in the get_sample_grads(), 
        and add "sum" of sample_loss on top of "mean" of total loss.
    Now (InfnetMultiSampleProbReduceFix and InfnetMultiSampleReduceFix)
        We are not taking any "mean" reducing before the end.
        When adding the sample_loss, we only take torch.mean() across samples,
        but the rest of reduce happens after all the addition of the losses.
    """
    def __init__(
        self,
        optimizer: Optimizer, #loss_fn: Loss,  
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Loss,
        loss_fn_sample: Optional[Loss] = None, 
        loss_fn_for_grad: Optional[Loss] = None,
        sample_loss_weight: Optional[float] = 1.0,
        num_samples: int = 1,
        keep_probs: bool = False, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            optimizer,
            inference_nn,
            score_nn,
            loss_fn,
            loss_fn_sample,
            loss_fn_for_grad,
            sample_loss_weight,
            num_samples,
            keep_probs,
            cost_augmented_layer,
            oracle_value_function,
            stopping_criteria
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
        total_loss = self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
            ) # (batch, 1, num_labels)
        if self.num_samples>0:
            loss_samples = self.loss_fn_sample(
                    x,
                    samples,
                    y_hat.expand_as(samples),
                    None, # y_cost_aug shouldn't be calculated.
                    buffer,
                ) # (batch, num_samples, num_labels)
            
            # only added line in InfnetMultiSampleProb.
            sample_prob = torch.exp(loss_samples)
            
            grad_samples = self.get_sample_grads(
                    x,
                    samples,
                    buffer,
            ) # (batch, num_samples, num_labels) grab gradients w.r.t.samples from score loss.
            loss_samples = grad_samples*sample_prob
            self._metrics["sampling_loss"] = float(torch.mean(loss_samples))
            total_loss = torch.mean(
                    total_loss + torch.mean(self.sample_loss_weight * loss_samples, dim=1, keepdim=True)
            )
        total_loss.backward()  # type:ignore
        self.optimizer.step()

        return total_loss

@Sampler.register("infnet-multi-sample-debug", constructor="from_partial_objects")
class InfnetMultiSampleDebug(InfnetMultiSampleLearner):
    def __init__(
        self,
        optimizer: Optimizer, #loss_fn: Loss,  
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Loss,
        loss_fn_sample: Optional[Loss] = None, 
        loss_fn_for_grad: Optional[Loss] = None,
        sample_loss_weight: Optional[float] = 1.0,
        num_samples: int = 1,
        keep_probs: bool = False, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            optimizer,
            inference_nn,
            score_nn,
            loss_fn,
            loss_fn_sample,
            loss_fn_for_grad,
            sample_loss_weight,
            num_samples,
            keep_probs,
            cost_augmented_layer,
            oracle_value_function,
            stopping_criteria,
        )

    @classmethod
    def from_partial_objects(
        cls,
        optimizer: Lazy[Optimizer],
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        loss_fn: Lazy[Loss], # first loss is always multi-sample loss.
        loss_fn_sample: Lazy[Loss], 
        loss_fn_for_grad: Lazy[Loss],
        sample_loss_weight: Optional[float] = 1.0,
        num_samples: int = 1,
        keep_probs: bool = False, # newly added
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        stopping_criteria: Union[int, StoppingCriteria] = 1,
        **kwargs: Any,
    ) -> "InfnetMultiSampleDebug":
        loss_fn_ = loss_fn.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        loss_fn_sample_ = loss_fn_sample.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        loss_fn_for_grad_ = loss_fn_for_grad.construct(
            score_nn=score_nn, oracle_value_function=oracle_value_function
        )
        
        trainable_parameters: Dict[str, torch.Tensor] = {}

        trainable_parameters.update(inference_nn.named_parameters())

        if cost_augmented_layer is not None:
            trainable_parameters.update(
                cost_augmented_layer.named_parameters()
            )
        optimizer_ = optimizer.construct(
            model_parameters=list(trainable_parameters.items())
        )

        return cls(
            optimizer=optimizer_,
            inference_nn=inference_nn,
            score_nn=score_nn,
            loss_fn=loss_fn_,
            loss_fn_sample=loss_fn_sample_,
            loss_fn_for_grad=loss_fn_for_grad_,
            sample_loss_weight=sample_loss_weight,
            num_samples=num_samples,
            keep_probs=keep_probs,
            cost_augmented_layer=cost_augmented_layer,
            oracle_value_function=oracle_value_function,
            stopping_criteria=stopping_criteria,
        )
    # the only override done.
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
        total_loss = self.loss_fn(
                x,
                labels,
                y_hat,
                y_cost_aug,
                buffer,
            ) # (batch, 1, num_labels)

        # assert self.num_samples==1  --> removed statement related to num_samples.
        # change the sample --> y_hat
        loss_samples = self.loss_fn_sample(
                x,
                y_hat,
                y_hat,
                None, # y_cost_aug shouldn't be calculated.
                buffer,
            ) # (batch, num_samples, num_labels)
        
        
        grad_samples = self.get_sample_grads(
                x,
                torch.sigmoid(y_hat),
                buffer,
        ) # (batch, num_samples, num_labels) grab gradients w.r.t.samples from score loss.
        # print(loss_samples.size())
        # print(grad_samples.size())
        loss_samples = grad_samples*loss_samples
        self._metrics["sampling_loss"] = float(torch.mean(loss_samples))
        total_loss = torch.mean( total_loss + self.sample_loss_weight * loss_samples )
        total_loss.backward()  # type:ignore
        # y_inf.expand_as(pseudo_labels)
        self.optimizer.step()

        return total_loss

    def get_sample_grads(
        self, 
        x: Any, 
        samples: torch.Tensor,
        buffer: Dict,
    ) -> torch.Tensor:
        """
        Returns:
            loss: loss value at the previous point (unreduced)
        """
        
        # samples.requires_grad = True
        assert (
            samples.requires_grad
        ), "Samples to grab gradient should have requires_grad=True"
        # if I get error here, I should fix it later.

        with torch.enable_grad():
            samples.grad = None # --> maybe not needed
            loss_for_grad = self.loss_fn_for_grad(
                x,
                None, # labels not required to get score-NN score.
                samples,
                None, # y_cost_aug shouldn't be used for gradient grabbing.
                buffer,
            )
            
            loss = torch.sum(loss_for_grad) 
            grad_samples = torch.autograd.grad(outputs=loss, inputs=samples, only_inputs=True)

        return grad_samples[0].clone().detach() # grad returns tuple. list of length 1.