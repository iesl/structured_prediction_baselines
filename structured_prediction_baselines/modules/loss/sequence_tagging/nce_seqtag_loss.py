from typing import List, Tuple, Union, Dict, Any, Optional, Literal
from structured_prediction_baselines.modules.loss import (
    Loss,
    NCELoss,
    NCERankingLoss,
)
import torch


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y,dim=-1)


class SeqTagNCERankingLoss(NCERankingLoss):
    def __init__(self, 
                sign: Literal["-", "+"] = "-", 
                use_scorenn: bool = True,
                use_distance: bool = True,
                **kwargs: Any):
        super().__init__(use_scorenn, **kwargs)
        self.sign = sign
        self.mul = -1 if sign == "-" else 1
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none") # works on logits, not prob.
        self.use_distance = use_distance
        # when self.use_scorenn=False, the sign should always be +,
        # as we want to have P_0/\sum(P_i) rather than (1/P_0) /\sum(1/P_i)
        assert (sign == "+" if not self.use_scorenn else True)  

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def distance(
        self,
        samples: torch.Tensor,  # (batch, num_samples, num_seq, num_labels)
        probs: torch.Tensor,  # (batch, num_samples, num_seq, num_labels) # expanded
    ) -> torch.Tensor:  # (batch, num_samples)
        """
        mul*CE(inp=probs, target=samples). Here mul is 1 or -1. If mul = 1 the ranking loss will
        use adjusted_score of score - CE. (mul=-1 corresponds to standard NCE)

        Note:
            Remember that CE = -y ln(x). Hence if samples are discrete, then CE = -ln Pn.
            So in that case sign of + in this class will result in adjusted_score = score - (- ln Pn) = score + ln Pn.
        """
        if not self.use_distance: # if not using distance then skip the CE computation.
            return torch.zeros([samples.shape[0], samples.shape[1]], dtype=torch.long, device=probs.device) # (batch,sample)
        
        def softCE(prediction, target):
            return -(target * torch.log(prediction)).sum(dim=-1).sum(dim=-1)
            
        # return self.mul * torch.sum( torch.sum(
        #     self.cross_entropy(torch.log(probs), samples), #torch.log() to make the prob value to be logit.
        # dim=-1), dim=-1)  # (batch, num_samples) 

        return self.mul * softCE(probs, samples) 


@Loss.register("multi-label-nce-ranking-with-discrete-sampling")
class SeqTagNCERankingLossWithDiscreteSamples(SeqTagNCERankingLoss):
    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples, num_labels)
        """
        Discrete sampling from the Bernoulli distribution.
        """
        assert probs.dim() == 3
        p = probs.squeeze(1)  # (batch, num_labels)
        samples = torch.transpose(
            torch.distributions.Bernoulli(probs=p).sample(  # type: ignore
                [self.num_samples]  # (num_samples, batch, num_labels)
            ),
            0,
            1,
        )  # (batch, num_samples, num_labels)

        return samples


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    x should be in [0, 1]
    """

    return -torch.log((1.0 / (x + 1e-13)) - 1.0 + 1e-35)


@Loss.register("multi-label-nce-ranking-with-cont-sampling")
class SeqTagNCERankingLossWithContSamples(SeqTagNCERankingLoss):
    def __init__(self, std: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.std = std

    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples, num_labels)
        """
        Cont sampling from by adding gaussian noise to logits
        """
        assert probs.dim() == 3
        logits = inverse_sigmoid(probs)
        samples = torch.sigmoid(
            torch.normal(
                logits.expand(
                    -1, self.num_samples, -1
                ),  # (batch, num_samples, num_labels)
                std=self.std,
            )
        )  # (batch, num_samples, num_labels)

        return samples
