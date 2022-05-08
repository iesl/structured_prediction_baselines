import logging
from typing import Dict, List, Iterable, Tuple, Any

from allennlp.data import Tokenizer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence
from allennlp_models.structured_prediction.dataset_readers.srl import _convert_verb_indices_to_wordpiece_indices, _convert_tags_to_wordpiece_tags

logger = logging.getLogger(__name__)


@DatasetReader.register("srl-custom")
class SrlReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : `TextField`
        The tokens in the sentence.
    verb_indicator : `SequenceLabelField`
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : `SequenceLabelField`
        A sequence of Propbank tags for the given verb in a BIO format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    model_name : `Optional[str]`, (default = `None`)
        The transformer model name to be used, we will load this PretrainedTransformerTokenizer. If not,
        the tokens will be indexed as normal with the token_indexers.
    domain_identifier : `str`, (default = `None`)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    # Returns

    A `Dataset` of `Instances` for Semantic Role Labelling.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        model_name: str = None,
        domain_identifier: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if token_indexers is not None:
            self._token_indexers = token_indexers
        elif model_name is not None:
            self._token_indexers = {"tokens": PretrainedTransformerIndexer(model_name)}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

        if model_name is not None:
            self._tokenizer = PretrainedTransformerTokenizer(model_name=model_name, add_special_tokens=False)
        else:
            self._tokenizer = None

    def _tokenize_input(
        self, tokens: List[str]
    ) -> Tuple[List[Token], List[int], List[int]]:
        """
        Convert a list of tokens to tokens and offsets, as well as adding
        separator tokens to the beginning and end of the sentence.
        """
        word_piece_tokens = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            word_pieces = self._tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = self._tokenizer.single_sequence_start_tokens + word_piece_tokens + self._tokenizer.single_sequence_end_tokens

        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info(
                "Filtering to only include file paths containing the %s domain",
                self._domain_identifier,
            )

        for sentence in self._ontonotes_subset(
            ontonotes_reader, file_path, self._domain_identifier
        ):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    yield self.text_to_instance(tokens, verb_indicator, tags)

    @staticmethod
    def _ontonotes_subset(
        ontonotes_reader: Ontonotes, file_path: str, domain_identifier: str
    ) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(  # type: ignore
        self, tokens: List[Token], verb_label: List[int], tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """

        metadata_dict: Dict[str, Any] = {}
        if self._tokenizer is not None:
            wordpieces, offsets, start_offsets = self._tokenize_input(
                [t.text for t in tokens]
            )
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField(
                wordpieces,
                token_indexers=self._token_indexers,
            )
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {
            "tokens": text_field,
            "verb_indicator": verb_indicator
        }

        if all(x == 0 for x in verb_label):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            if self._tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields["tags"] = SequenceLabelField(new_tags, text_field)
            else:
                fields["tags"] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
