import logging
from typing import Dict, Optional

import allennlp.nn.util as util
import torch
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import (
  Seq2SeqEncoder,
  TextFieldEmbedder,
)
from allennlp_models.generation import SeqDecoder

from .task_nn import TaskNN

logger = logging.getLogger(__name__)


@TaskNN.register("machine-translation-shared-embedder", constructor="from_partial_objects")
@TaskNN.register("machine-translation")
class MachineTranslationTaskNN(TaskNN):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        decoder: SeqDecoder,
        encoder: Optional[Seq2SeqEncoder] = None,
    ):
        """

        Args:
            vocab : `Vocabulary`, required
                Vocabulary containing source and target vocabularies. They may be under the same namespace
                (`tokens`) or the target tokens can have a different namespace, in which case it needs to
                be specified as `target_namespace`.
            source_embedder : `TextFieldEmbedder`, required
                Used to embed the source tokens `TextField` we get as input to the model.
            encoder : `Seq2SeqEncoder`
                The encoder, optional, that we will use in between embedding source tokens and passing them to the decoder
            decoder: `SeqDecoder`
                The decoder of the model, currently we use the default implementation `auto_regressive_seq_decoder` with
                `stacked_self_attention` as the decoder_net


        """
        super().__init__()  # type:ignore
        self.source_embedder = source_embedder
        self.encoder = encoder
        self.decoder = decoder

        encoder_output_dim = self.encoder.get_output_dim() if self.encoder else self.source_embedder.get_output_dim()
        if encoder_output_dim != self._decoder.get_output_dim():
            raise ConfigurationError(
                f"Encoder output dimension {self._encoder.get_output_dim()} should be"
                f" equal to decoder dimension {self._decoder.get_output_dim()}."
            )

    @classmethod
    def from_partial_objects(
        cls,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        decoder: Lazy[SeqDecoder],
        encoder: Optional[Seq2SeqEncoder] = None,
    ) -> "MachineTranslationTaskNN":
        decoder_ = decoder.construct(target_embedder=source_embedder)
        return cls(
            vocab=vocab,
            source_embedder=source_embedder,
            decoder=decoder_,
            encoder=encoder
        )

    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """
        Make forward pass on the encoder and decoder for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        # Returns

        `Dict[str, torch.Tensor]`
            The output tensors from the decoder.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self.source_embedder(source_tokens)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self.encoder(embedded_input, source_mask) if self.encoder else embedded_input
        encoded_state = {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

        decoder_output = self.decoder(encoded_state, target_tokens)
        return decoder_output['logits'], decoder_output.get('predictions')

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        return self._decoder.post_process(output_dict)

    def mark_parameters_with_optimizer_mode(self):
        pass
