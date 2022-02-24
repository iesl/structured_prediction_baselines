from typing import (
    Dict,
    List,
    Union,
    Any,
    Iterator,
    Tuple,
    cast,
    Optional,
    Iterable,
)
import sys
import itertools
from wcmatch import glob

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    ArrayField,
    ListField,
    MetadataField,
    MultiLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token

import allennlp

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("trec")
class TrecReader(DatasetReader):
    """
    Single-label classification `dataset <https://cogcomp.seas.upenn.edu/Data/QA/QC/>`.

    The output of this DatasetReader follows :class:`MultiInstanceEntityTyping`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        # use_transitive_closure: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            tokenizer: The tokenizer to be used.
            token_indexers: The token_indexers to be used--one per embedder. Will usually be only one.
            # use_transitive_closure: use types_extra
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_

        """
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        # self._use_transitive_closure = use_transitive_closure

    def example_to_fields(
        self,
        text: str,
        label_coarse: str,
        label_fine: str,
        idx: str,
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can be used
        to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
        the meta dict.

        Args:
            text: question
            label_coarse: coarse-grained label
            label_fine: fine-grained label
            idx: unique id
            meta: None
            **kwargs: TODO

        Returns:
            Dictionary of fields with the following entries:
                x: contains the question text
                labels: contains the coarse-grained label

        """

        if meta is None:
            meta = {}

        meta["text"] = text
        meta["label_coarse"] = label_coarse
        meta["label_fine"] = label_fine
        meta["idx"] = idx

        x = TextField(
            self._tokenizer.tokenize(text),
        )
        labels = MultiLabelField([label_coarse])

        return {
            "x": x,
            "labels": labels,
        }

    def text_to_instance(  # type:ignore
        self,
        text: str,
        label_coarse: str,
        label_fine: str,
        idx: str,
        **kwargs: Any,
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            text: Question text
            label_coarse: Coarse-grained label
            label_fine: Fine-grained label
            idx: Identification number
            **kwargs: unused

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            text, label_coarse, label_fine, idx, meta=meta_dict
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: Path to the data file

        Yields:
            data instances

        """

        for file_ in glob.glob(file_path, flags=glob.EXTGLOB):
            with open(file_) as f:
                for line in self.shard_iterable(f):
                    example = json.loads(line)
                    instance = self.text_to_instance(**example)
                    yield instance

    def apply_token_indexers(self, instance: Instance) -> None:
        text_field = cast(TextField, instance.fields["x"])  # no runtime effect
        text_field._token_indexers = self._token_indexers
