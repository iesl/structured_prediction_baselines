from typing import List, Tuple, Union, Dict, Any, Optional, cast
from structured_prediction_baselines.modules.loss import Loss
from structured_prediction_baselines.modules.loss.inference_net_loss import MarginBasedLoss
from allennlp.common.checks import ConfigurationError
from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.oracle_value_function import (
    OracleValueFunction,
)
import torch

# Losses to train score-NN with noise contrastive estimation (NCE) techniques

class NCERankingLoss(Loss):
    """
    Loss function to train NCE ranking energy model, 
    NCE ranking loss: 0 index has label & rest has samples from Task-NN.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bce_wlogit_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for NCERankingLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # I want to get individual Energy scores & probability scores.
        ranking_loss = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        return self.compute_loss(ranking_loss)
    
    def _compute_ranking_loss(
        self,
        predicted_score: torch.Tensor,
        neg_p_n: torch.Tensor,
    ) -> torch.Tensor:
        neg_p_n = torch.sum(neg_p_n, dim=2) # (batch, num_samples)
        new_score = predicted_score + neg_p_n
        ranking_loss = self.ce_loss(new_score, torch.zeros(new_score.shape[0],dtype=torch.long).cuda())

        # for debugging 
        # y_hat.expand_as(samples): (batch, 1, labels) --> (batch, num_samples, labels)
        # print("==============================================predicted_score.shape:{}".format(predicted_score.shape))
        # print("==============================================p_n.shape:{}".format(p_n.shape))

        return ranking_loss # (batch, num_samples)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, 1, ...)
        # samples shape (batch, num_samples, num_labels)
        if 'samples' in buffer.keys():
            samples = buffer['samples']    
        else:
            samples = y_hat_extra # hack for validation loop. (buffer not explicitly passed)

        if labels is not None:
            samples = torch.cat(
                        (labels, samples), dim=1
                    ) # concatenate true label on the front (0 index).
        num_samples = samples.shape[1] 

        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized.
        # Thus peform the normalization based on the task
        
        # self.normalize_y = False --> as it's already normalzied.
        if self.normalize_y: 
            samples = self.normalize(samples)

        predicted_score = self.score_nn(
            x, samples, buffer, **kwargs
        )  # (batch, num_samples)
        neg_p_n = self.bce_wlogit_loss(
                y_hat.expand_as(samples), samples.to(dtype=y_hat.dtype)
        ) 
        # y_hat should not be normalized for BCEWithLogitLoss(),
        # samples should be between [0,1] for each entry.
        ranking_loss = self._compute_ranking_loss(predicted_score,neg_p_n)
        return ranking_loss

    def get_metrics(self, reset: bool = False):
        metrics = {}
        return metrics

# organize this so that _get_values has two parts.

class NCERevRankingLoss(Loss):
    """
    Loss function to train DVN, 
    NCE ranking loss: 0 index has label & rest has samples from Task-NN.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bce_wlogit_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for NCERankingLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # I want to get individual Energy scores & probability scores.
        ranking_loss = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        return self.compute_loss(ranking_loss)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, 1, ...)
        # samples shape (batch, num_samples, num_labels)
        if 'samples' in buffer.keys():
            samples = buffer['samples']    
        else:
            samples = y_hat_extra # hack for validation loop. (buffer not explicitly passed)

        if labels is not None:
            samples = torch.cat(
                        (labels, samples), dim=1
                    ) # concatenate true label on the front (0 index).
        num_samples = samples.shape[1] 

        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task
        
        # self.normalize_y = False --> as it's already normalzied.
        if self.normalize_y: 
            samples = self.normalize(samples)

        predicted_score = self.score_nn(
            x, samples, buffer, **kwargs
        )  # (batch, num_samples)
        neg_p_n = self.bce_wlogit_loss(
                y_hat.expand_as(samples), samples.to(dtype=y_hat.dtype)
        ) 
        # y_hat should not be normalized for BCEWithLogitLoss(),
        # samples should be between [0,1] for each entry.
        
        neg_p_n = torch.sum(neg_p_n, dim=2)
        # (batch, num_samples)
        # y_hat.expand_as(samples): (batch, 1, labels) --> (batch, num_samples, labels)
        # print("==============================================predicted_score.shape:{}".format(predicted_score.shape))
        # print("==============================================p_n.shape:{}".format(p_n.shape))
        new_score = predicted_score - neg_p_n
        ranking_loss = self.ce_loss(new_score, torch.zeros(new_score.shape[0],dtype=torch.long).cuda())
        
        return ranking_loss
    
    def get_metrics(self, reset: bool = False):
        metrics = {}
        return metrics



class RankingWithoutNoise(Loss):
    """
    Loss function to train DVN, 
    NCE ranking loss: 0 index has label & rest has samples from Task-NN.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.bce_wlogit_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for NCERankingLoss")

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # (batch, num_samples)
        oracle_value: Optional[torch.Tensor],  # (batch, num_samples)
    ) -> torch.Tensor:
        raise NotImplementedError

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # I want to get individual Energy scores & probability scores.
        ranking_loss = self._get_values(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        return self.compute_loss(ranking_loss)

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, 1, ...)
        # samples shape (batch, num_samples, num_labels)
        if 'samples' in buffer.keys():
            samples = buffer['samples']    
        else:
            samples = y_hat_extra # hack for validation loop. (buffer not explicitly passed)

        if labels is not None:
            samples = torch.cat(
                        (labels, samples), dim=1
                    ) # concatenate true label on the front (0 index).
        num_samples = samples.shape[1] 

        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task
        
        # self.normalize_y = False --> as it's already normalzied.
        if self.normalize_y: 
            samples = self.normalize(samples)

        predicted_score = self.score_nn(
            x, samples, buffer, **kwargs
        )  # (batch, num_samples)
        ranking_loss = self.ce_loss(predicted_score, torch.zeros(predicted_score.shape[0],dtype=torch.long).cuda())
        
        return ranking_loss

    def get_metrics(self, reset: bool = False):
        metrics = {}
        return metrics

class NCERankingInterpolatedLoss(NCERankingLoss):
    """
    On top of the hard samples, 
    interpolating the discrete samples so that continuous points are included.
    """
    def __init__(
        self, 
        sample_weight=0.5, 
        **kwargs: Any
    ):
        """
        Args: sample_weight --> to control interpolation weights on sample.
        """
        super().__init__(**kwargs)
        self.sample_weight = sample_weight

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, 1, ...)
        # samples shape (batch, num_samples, num_labels)
        if 'samples' in buffer.keys():
            samples = buffer['samples']    
        else:
            samples = y_hat_extra # hack for validation loop. (buffer not explicitly passed)
        
        k = self.sample_weight # sample weight between 0,1
        samples_interpolated =  k*samples + (1-k)*labels
        if labels is not None:
            samples = torch.cat(
                        (labels, samples, samples_interpolated), dim=1
                    ) # concatenate true label on the front (0 index).
        num_samples = samples.shape[1] 

        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect
        # score_nn always expects y to be normalized
        # do the normalization based on the task
        
        # self.normalize_y = False --> as it's already normalzied.
        if self.normalize_y: 
            samples = self.normalize(samples)

        predicted_score = self.score_nn(
            x, samples, buffer, **kwargs
        )  # (batch, num_samples)
        neg_p_n = self.bce_wlogit_loss(
                y_hat.expand_as(samples), samples.to(dtype=y_hat.dtype)
        ) 
        # y_hat should not be normalized for BCEWithLogitLoss(),
        # samples should be between [0,1] for each entry.
        
        neg_p_n = torch.sum(neg_p_n, dim=2)
        # (batch, num_samples)
        # y_hat.expand_as(samples): (batch, 1, labels) --> (batch, num_samples, labels)
        # print("==============================================predicted_score.shape:{}".format(predicted_score.shape))
        # print("==============================================p_n.shape:{}".format(p_n.shape))
        new_score = predicted_score + neg_p_n
        ranking_loss = self.ce_loss(new_score, torch.zeros(new_score.shape[0],dtype=torch.long).cuda())
        
        return ranking_loss