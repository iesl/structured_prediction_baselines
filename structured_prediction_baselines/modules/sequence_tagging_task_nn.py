from .task_nn import TaskNN
from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import (
    Seq2SeqEncoder,
    TimeDistributed,
    TextFieldEmbedder,
    FeedForward,
)
from torch.nn.modules.linear import Linear
import allennlp.nn.util as util


@TaskNN.register("seq-tagging")
class SequenceTaggingTaskNN(TaskNN):
    def __init__(
        self,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        num_tags: int,
        feedforward: Optional[FeedForward] = None,
        dropout: float = 0,
    ):
        """

        Args:
            text_field_embedder : `TextFieldEmbedder`, required
                Used to embed the tokens `TextField` we get as input to the model.
            encoder : `Seq2SeqEncoder`
                The encoder that we will use in between embedding tokens and predicting output tags.
            feedforward : `FeedForward`, optional, (default = `None`).
                An optional feedforward layer to apply after the encoder.

        """
        self.num_tags
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()  # type: ignore
        else:
            output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(  # type: ignore
            torch.nn.Sequential(
                Linear(output_dim, num_tags, bias=False),
                torch.nn.Softmax(dim=-1),
            )
        )

        if dropout:
            self.dropout: Optional[torch.nn.Module] = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        buffer: Dict = None,
    ) -> torch.Tensor:
        if not buffer:
            buffer = {}
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        buffer["mask"] = mask

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self.feedforward is not None:
            encoded_text = self.feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)

        return logits  # shape (batch, 1, sequence, num_tags)
