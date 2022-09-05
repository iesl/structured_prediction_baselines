import sys, itertools, logging, json
from typing import Dict, List, Union, Any, Iterator, Tuple, cast, Optional, Iterable, Literal
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict
from wcmatch import glob

import allennlp
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token

logger = logging.getLogger(__name__)


@DatasetReader.register("trec")
class TRECReader(DatasetReader):
    """
    Question classification `dataset <https://cogcomp.seas.upenn.edu/Data/QA/QC/>`.
    """

    def __init__(
        self,
        granularity: Literal["coarse", "fine"],
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            granularity: The level of label granularity.
            tokenizer: The tokenizer to be used.
            token_indexers: The token_indexers to be used --- one per embedder. Will usually be only one.
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_
        """
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._granularity = granularity
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

    def text_to_instance(self, text: str, label_coarse: str, label_fine: str, idx: int) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.
        Args:
            text: Question text
            label_coarse: Coarse-grained label
            label_fine: Fine-grained label
            idx: Identification number
        Returns:
             :class:`Instance` of data
        """
        x = TextField(self._tokenizer.tokenize(text))
        label_text = f"{label_coarse}_{label_fine}" if self._granularity == "fine" else label_coarse
                        # distinguish multiple fine "other" labels
        label = LabelField(label_text) # default label_namespace is "labels"

        return Instance({"x": x, "label": label})

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
        text_field = cast(TextField, instance.fields["x"]) # no runtime effect
        text_field._token_indexers = self._token_indexers