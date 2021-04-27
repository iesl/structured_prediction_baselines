from typing import Optional, Union, Dict

import torch
from allennlp.models import Model
from allennlp.training import util as training_util


def get_metrics(
    model: Model,
    total_loss: float,
    total_reg_loss: Optional[float],
    batch_loss: Optional[float],
    batch_reg_loss: Optional[float],
    num_batches: int,
    reset: bool = False,
    world_size: int = 1,
    cuda_device: Union[int, torch.device] = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Works same as get_metrics function from allennlp.training.utils but for the sampler_loss
    """
    metrics = training_util.get_metrics(
        model, total_loss, total_reg_loss, batch_loss, batch_reg_loss, num_batches, reset, world_size, cuda_device)
    sampler_loss = metrics.get('sampler_loss', None)
    if sampler_loss is not None:
        metrics["batch_sampler_loss"] = sampler_loss
        # metrics["sampler_loss"] = float(metrics['total_sampler_loss'] / num_batches) if num_batches > 0 else 0.0
        metrics.pop('sampler_loss')
    return metrics
