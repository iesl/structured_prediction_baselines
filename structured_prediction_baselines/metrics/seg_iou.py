from typing import Optional
import torch
from allennlp.nn.util import dist_reduce_sum
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


def get_iou(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.flatten(y_pred).reshape(1, -1)
    y_true = torch.flatten(y_true).reshape(1, -1)
    y_concat = torch.cat([y_pred, y_true], 0)
    intersect = torch.sum(torch.min(y_concat, 0)[0]).float()
    union = torch.sum(torch.max(y_concat, 0)[0]).float()
    return intersect / max(10 ** -8, union)


@Metric.register("seg-iou")
class SegIoU(Metric):
    """
    Segmentation IoU.
    """

    supports_distributed = True

    def __init__(self) -> None:
        self.sum_seg_iou = 0.0
        self.total_count = 0.0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        batch_size = y_pred.shape[0]
        for i in range(batch_size):
            self.sum_seg_iou += get_iou(y_pred[i], y_true[i])
        self.total_count += batch_size

    def get_metric(self, reset: bool = False) -> float:
        """
        # Returns
        The accumulated segmentation iou.
        """
        if self.total_count > 1e-12:
            seg_iou = float(self.sum_seg_iou) / float(self.total_count)
        else:
            seg_iou = 0.0

        if reset:
            self.reset()

        return seg_iou

    def reset(self):
        self.sum_seg_iou = 0.0
        self.total_count = 0.0