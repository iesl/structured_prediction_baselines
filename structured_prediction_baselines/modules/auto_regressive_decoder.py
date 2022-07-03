from typing import Dict, List, Tuple, Optional

import numpy
import torch
import torch.nn.functional as F
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp_models.generation import SeqDecoder, StackedSelfAttentionDecoderNet
from allennlp_models.lm import TransformerBeamSearchGenerator
from overrides import overrides
from torch.nn import Linear

START_SYMBOL = '[CLS]'
END_SYMBOL = '[SEP]'


@SeqDecoder.register("auto_regressive_decoder")
class AutoRegressiveDecoder(SeqDecoder):
    """
    An autoregressive decoder that can be used for most seq2seq tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    decoder_net : `DecoderNet`, required
        Module that contains implementation of neural network for decoding output elements
    target_embedder : `Embedding`
        Embedder for target tokens.
    target_namespace : `str`, optional (default = `'tokens'`)
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    beam_search : `BeamSearch`, optional (default = `Lazy(BeamSearch)`)
        This is used to during inference to select the tokens of the decoded output sequence.
    tensor_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = `None`)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    scheduled_sampling_ratio : `float` optional (default = `0.0`)
        Defines ratio between teacher forced training and real output usage. If its zero
        (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
        predictions in a single forward pass of the `decoder_net`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        decoder_net: StackedSelfAttentionDecoderNet,
        target_embedder: TextFieldEmbedder,
        target_namespace: str = "labels",
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        tie_output_embedding: bool = False,
        label_smoothing_ratio: Optional[float] = None,
        **kwargs
    ) -> None:
        super().__init__(target_embedder)
        self._vocab = vocab

        # Decodes the sequence of encoded hidden states into e new sequence of hidden states.
        self._decoder_net = decoder_net
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)

        _beam_search = beam_search.construct(end_index=self._end_index)
        self._beam_search_generator = TransformerBeamSearchGenerator(beam_search=_beam_search)

        # Ensure beam_search_generator is compatable with text_field_embedder.
        self._beam_search_generator.validate_text_field_embedder(self.target_embedder)

        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(
            self._decoder_net.get_output_dim(), target_vocab_size
        )

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError(
                    "Can't tie embeddings with output linear layer, due to shape mismatch"
                )
            self._output_projection_layer.weight = self.target_embedder.weight

        # self._scheduled_sampling_ratio = scheduled_sampling_ratio

    def _forward_loss(
        self, state: Dict[str, torch.Tensor], target_tokens: TextFieldTensors
    ) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self.target_embedder(target_tokens)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        _, decoder_output = self._decoder_net(
            previous_state=state,
            previous_steps_predictions=target_embedding[:, :-1, :],
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_mask=target_mask[:, :-1],
        )

        # shape: (group_size, max_target_sequence_length, num_classes)
        logits = self._output_projection_layer(decoder_output)

        output_dict = {'logits': logits}
        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size, 1), fill_value=self._start_index, dtype=torch.long
        )

        decoder_init_state = {
            'token_ids': start_predictions,
            'mask': start_predictions.new_full(start_predictions.size(), True, dtype=bool),
            'type_ids': start_predictions.new_full(start_predictions.size(), 0, dtype=int),
        }

        state.update(decoder_init_state)

        # state = self._beam_search_generator.get_step_state(tokens)

        # Shape (top_indices): (batch_size, beam_size, num_predicted_tokens)
        # Shape (top_log_probs): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search_generator.search(
            start_predictions, state, self._beam_search_step
        )

        # Shape: (batch_size, beam_size)
        # top_probs = top_log_probs.exp()

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        # all_top_k_predictions, log_probabilities = self._beam_search.search(
        #     start_predictions, state, self.take_step
        # )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _beam_search_step(
        self, predicted_tokens: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Step function to use with `BeamSearch`.

        `predicted_tokens` is a tensor of shape `(group_size,)` and
        `state` is a dictionary of tensors with the following fields:
        - "token_ids": shape `(group_size, num_tokens)`
        - "mask": shape `(group_size, num_tokens)`
        - "type_ids": shape `(group_size, num_tokens)`
        """
        assert self._beam_search_generator is not None

        decoder_state = self._get_decoder_state(state)
        if step == 0:
            inputs = {self._beam_search_generator._namespace: decoder_state}
        else:
            inputs = self._beam_search_generator.prepare_step_input(predicted_tokens, decoder_state)
            decoder_state = self._beam_search_generator.get_step_state(inputs)
            state.update(decoder_state)

        # Shape: (group_size, vocab_size)
        next_token_scores = self._next_token_scores(inputs, state)

        # Shape: (group_size, vocab_size)
        log_probs = torch.nn.functional.log_softmax(next_token_scores, dim=-1)

        return log_probs, state

    def _get_decoder_state(self, state: Dict[str, torch.Tensor]):
        return {
            'token_ids': state["token_ids"],
            'mask': state["mask"],
            'type_ids': state["type_ids"]
        }

    def _next_token_scores(self, tokens: TextFieldTensors, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the unnormalized log probabilities of the potential next token.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # Shape: (group_size, num_tokens, embedding_dim)
        embeddings = self.target_embedder(tokens)

        # Shape: (group_size, num_tokens)
        mask = util.get_text_field_mask(tokens)

        _, decoder_output = self._decoder_net(
            previous_state=None,
            previous_steps_predictions=embeddings,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
        )

        decoder_output = decoder_output[:, -1, :]

        # shape: (group_size, num_classes)
        final_embeddings = self._output_projection_layer(decoder_output)

        return final_embeddings

    def _get_loss(
        self, logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of `targets` is expected to be greater than that of `logits` because the
        decoder does not need to compute the output corresponding to the last timestep of
        `targets`. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, label_smoothing=self._label_smoothing_ratio
        )

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(
                    self._tensor_based_metric.get_metric(reset=reset)  # type: ignore
                )
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    @overrides
    def forward(
        self,
        encoder_out: Dict[str, torch.LongTensor],
        target_tokens: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:
        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)

        if target_tokens:
            state_forward_loss = (
                state if self.training else {k: v.clone() for k, v in state.items()}
            )
            output_dict = self._forward_loss(state_forward_loss, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called `predicted_tokens` to the `output_dict`.
        """
        predicted_indices = output_dict["predictions"]
        all_predicted_tokens = self.indices_to_tokens(predicted_indices)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def indices_to_tokens(self, batch_indeces: numpy.ndarray) -> List[List[str]]:

        if not isinstance(batch_indeces, numpy.ndarray):
            batch_indeces = batch_indeces.detach().cpu().numpy()

        all_tokens = []
        for indices in batch_indeces:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            tokens = [
                self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_tokens.append(tokens)

        return all_tokens
