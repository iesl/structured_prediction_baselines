import torch
import torch.nn as nn
from torch.nn.functional import relu

from allennlp.models import Model
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TimeDistributed
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure
import allennlp.nn.util as util

from typing import Dict, Optional, Any, List
from overrides import overrides
from torch.nn.functional import one_hot

from structured_prediction_baselines.modules.score_nn import ScoreNN
from structured_prediction_baselines.modules.task_nn import TaskNN


# @Model.register('sequence-tagging-bert')
class SequenceTaggingBert(Model):
    eps = 1e-9

    def __init__(self, vocab: Vocabulary, num_tags: int, inf_nn: TaskNN, score_nn: ScoreNN, margin_type: int,
                 regularization: bool = True, _lambda: float = 1.0, ce_weight: float = 1.0):
        super().__init__(vocab)
        self.inf_nn = inf_nn
        self.score_nn = score_nn
        self.num_tags = num_tags
        # self.score_nn.structured_energy = structured_energy
        # self.structured_energy = structured_energy
        output_dim = 2*num_tags
        self.tag_projection_layer = TimeDistributed(  # type: ignore
            nn.Sequential(nn.Linear(output_dim, num_tags), nn.Softmax(dim=-1))
        )
        self.margin_type = margin_type
        self.regularization = regularization
        self._lambda = _lambda
        self.ce_weight = ce_weight

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        buffer = {}

        x_train, label_tags = inputs
        y_train = one_hot(label_tags, num_classes=self.num_tags).float()
        batch_size, seq_len, _ = y_train.shape

        y_inf = self.inf_nn(*inputs, buffer)
        mask = buffer.get("mask")
        buffer['y_inf'] = y_inf
        inf_cost = self.score_nn(x_train, y_inf, buffer)

        y_cost_aug = self.tag_projection_layer(torch.cat((y_inf, y_train), dim=2))
        cost_aug_cost = self.score_nn(x_train, y_cost_aug, buffer)
        ground_truth_cost = self.score_nn(x_train, y_train, buffer)

        hinge_cost_inf = inf_cost - ground_truth_cost
        delta0 = (torch.abs(y_train - y_cost_aug).sum(2) * mask).sum(1)

        hinge_cost = cost_aug_cost - ground_truth_cost

        if self.margin_type == 0:
            hinge_cost = delta0 + hinge_cost
        elif self.margin_type == 1:
            hinge_cost = 1.0 + hinge_cost
        elif self.margin_type == 2:
            hinge_cost = hinge_cost
        elif self.margin_type == 3:
            hinge_cost = delta0 * (1.0 + hinge_cost)

        nll_loss = nn.NLLLoss(reduction='none')
        log_y_hat = torch.log(self.eps + y_inf.view(-1, self.num_tags))  # [B*T, C]
        ce_hinge_inf = nll_loss(log_y_hat, label_tags.view(-1))
        ce_hinge_inf = (ce_hinge_inf.view(batch_size, seq_len) * mask).sum(1)

        g_cost = torch.mean(-hinge_cost) + self._lambda * torch.mean(-hinge_cost_inf)
        d_cost = torch.mean(relu(hinge_cost)) + self._lambda * torch.mean(relu(hinge_cost_inf))

        if self.regularization:
            g_cost += self.ce_weight * torch.mean(ce_hinge_inf)

        return g_cost, d_cost
