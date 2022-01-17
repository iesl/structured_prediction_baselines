from typing import Optional
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum


@Metric.register("sequence_exact_match")
class SequenceExactMatch(Metric):
    """
    Sequence exact match accuracy for sequence tagging tasks.
    Correctness require all tags to be correct in a sequence.
    """

    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.dim() - 1 but "
                "found tensor of shape: {}".format(gold_labels.size())
            )
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )
        if mask is not None and mask.size() != gold_labels.size():
            raise ConfigurationError(
                "mask must have the same size as predictions but "
                "found tensor of shape: {}".format(mask.size())
            )

        predictions = predictions.max(-1)[1]

        if mask is not None:
            predictions *= mask
            gold_labels *= mask
        correct = predictions.eq(gold_labels).float() # the masked positions are True
        correct = correct.min(dim=-1)[0].sum().item() # min is 1 iff exact match, else 0
                    # Sanity check: if changed to max, most seq has 1 and the metric should be very high

        # Sanity check: To verify that the metric is properly implemented,
        # comment the above block and uncomment this block.
        # Here for each sequence, we calculate the percentage of correctly tagged tokens
        # instead of an overall 1 or 0 for whether the sequence is entirely tagged correctly.
        # correct = predictions.eq(gold_labels).float()
        # correct *= mask
        # correct = correct.sum(dim=-1) / mask.float().sum(dim=-1)
        # correct = correct.sum().item()

        _total_count = predictions.size()[0]
        _correct_count = correct

        self.correct_count += dist_reduce_sum(_correct_count)
        self.total_count += dist_reduce_sum(_total_count)

    def get_metric(self, reset: bool = False) -> float:
        """
        # Returns
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0

        if reset:
            self.reset()

        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0