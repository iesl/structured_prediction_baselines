from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
from allennlp.modules import (
    FeedForward,
    Seq2SeqEncoder,
    Seq2VecEncoder,
    TextFieldEmbedder,
)
from allennlp.nn.util import get_text_field_mask
from allennlp.data import Vocabulary, TextFieldTensors
import torch


class TaskNN(torch.nn.Module, Registrable):
    """Base class for creating feature representation for any task.

    Inheriting classes should override the `foward` method.
    """

    pass


class CostAugmentedLayer(torch.nn.Module, Registrable):
    def forward(self, inp: torch.Tensor, buffer: Dict) -> torch.Tensor:
        raise NotImplementedError


class TextEncoder(torch.nn.Module, Registrable):
    """Base class for creating feature representation for tasks with textual input.

    See `BasicClassifier <https://github.com/allenai/allennlp/blob/v2.5.0/allennlp/models/basic_classifier.py>`_ for reference.
    """

    default_implementation = "text-encoder"

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        final_dropout: Optional[float] = None,
        **kwargs: Any,
    ):
        """

        Args:
            vocab : `Vocabulary`
            text_field_embedder : `TextFieldEmbedder`
                Used to embed the input text into a `TextField`
            seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
                Optional Seq2Seq encoder layer for the input text.
            seq2vec_encoder : `Seq2VecEncoder`
                Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
                will pool its output. Otherwise, this encoder will operate directly on the output
                of the `text_field_embedder`.
            feedforward : `FeedForward`, optional, (default = `None`)
                An optional feedforward layer to apply after the seq2vec_encoder.
        """
        super().__init__()  # type: ignore
        self.vocab = vocab
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.feedforward = feedforward

        if final_dropout:
            self.final_dropout: Optional[torch.nn.Module] = torch.nn.Dropout(
                final_dropout
            )
        else:
            self.final_dropout = None

        self._output_dim = seq2vec_encoder.get_output_dim()

        if self.feedforward:
            self._output_dim = self.feedforward.get_output_dim()  # type:ignore

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: TextFieldTensors) -> torch.Tensor:
        """
        Encodes the text input into a feature vector.
        """
        embedded_text = self.text_field_embedder(x)
        mask = get_text_field_mask(x)

        if self.seq2seq_encoder:
            embedded_text = self.seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self.seq2vec_encoder(embedded_text, mask=mask)

        if self.final_dropout:
            embedded_text = self.final_dropout(embedded_text)

        if self.feedforward is not None:
            embedded_text = self.feedforward(embedded_text)

        return embedded_text


# register itself as a concrete class
TextEncoder.register("text-encoder")(TextEncoder)
