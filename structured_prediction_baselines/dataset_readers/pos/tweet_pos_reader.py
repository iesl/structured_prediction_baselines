from typing import Dict, Iterable, List
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.fields import LabelField, TextField, SequenceLabelField
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
import os, pickle

@DatasetReader.register("tweet_pos")
class TweetPosReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        model_name: str = "bert-base-uncased",
        max_length: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers or {
            "transformer_indexer": PretrainedTransformerMismatchedIndexer(model_name=model_name, max_length=max_length)
        }
        self.model_name = model_name
        self.max_length = max_length

    def text_to_instance(self, tokens: str, tags: str = None) -> Instance:
        tokens = [Token(token) for token in tokens.split()]
        tags = tags.split()
        if len(tokens) != len(tags):
            print(tokens)
            print(tags)
            return None
        if self.max_length:
            tokens = tokens[:self.max_length]
            tags = tags[:self.max_length]
        text_field = TextField(tokens, token_indexers=self.token_indexers)
        sequence_field = SequenceLabelField(tags, sequence_field=text_field, label_namespace='labels')

        # sanity check
        # print("tokens:", tokens) # original
        # print("tags:", tags) # original
        # vocab = Vocabulary()
        # text_field.index(vocab)
        # token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
        # print("token tensor:", token_tensor)
        #     # token_ids: [id for CLS], ids for SUBtokens, [id for SEP]
        #     # mask: [True] * n_ORIGINAL_tokens
        #     # type_ids: [0] * (n_SUBtokens+2)
        #     # wordpiece_mask: [True] * (n_SUBtokens+2)
        #     # segment_concat_mask: [True] * (n_SUBtokens+2)
        #     # offsets: [subtoken_start, subtoken_end] for SUBtokens of original tokens, not [CLS] or [SEP]
        # print()
        # embedding = PretrainedTransformerMismatchedEmbedder(model_name=self.model_name)
        # embedder = BasicTextFieldEmbedder(token_embedders={"transformer_indexer": embedding})
        # tensor_dict = text_field.batch_tensors([token_tensor])
        # embedded_tokens = embedder(tensor_dict)
        # print("Embedded tokens size:", embedded_tokens.size()) # [1, n_original_tokens, hidden_size]
        # print("Embedded tokens:", embedded_tokens)
        # exit()

        return Instance({'tokens': text_field, 'tags': sequence_field})

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        :param file_path: data directory, eg. data/Tweet_POS/daily547.proc.cnn
        """

        if os.path.isdir(file_path):
            cached_file = os.path.join(file_path, f'cached_{self.model_name}')
        else:
            dir, file = os.path.split(file_path)
            cached_file = os.path.join(dir, f'cached_{self.model_name}_{file}')
        if os.path.exists(cached_file):
            print(f"Loading instances from cached file {cached_file}")
            with open(cached_file, 'rb') as f:
                instances = pickle.load(f)
        else:
            print(f"Creating instances and saving at {cached_file}")

            def _read_file(input_file):
                with open(input_file, "r") as f:
                    return [line.strip().split(" ||| ") for line in f]
            texts_and_tags = _read_file(os.path.join(file_path))

            instances = []
            n_error_instance = 0
            for text, tags in texts_and_tags:
                instance = self.text_to_instance(text, tags)
                if instance:
                    instances.append(instance)
                else:
                    n_error_instance += 1
            print(f"file: {file_path}")
            print(f"{len(instances)} instances, {n_error_instance} error instances.")
            with open(cached_file, 'wb') as f:
                pickle.dump(instances, f)

        for instance in instances:
            yield instance


if __name__ == "__main__":
    reader = TweetPosReader()
    dataset = list(reader.read("/mnt/nfs/scratch1/wenlongzhao/SEAL/structured_prediction_baselines/data/Tweet_POS/oct27.traindev.proc.cnn"))
        # 1324 instances, 3 error instances.
    dataset = list(reader.read("/mnt/nfs/scratch1/wenlongzhao/SEAL/structured_prediction_baselines/data/Tweet_POS/oct27.test.proc.cnn"))
        # 500 instances, 0 error instances.
    dataset = list(reader.read("/mnt/nfs/scratch1/wenlongzhao/SEAL/structured_prediction_baselines/data/Tweet_POS/daily547.proc.cnn"))
        # 547 instances, 0 error instances.
    print("type of its first element: ", type(dataset[0]))
    print("size of dataset: ", len(dataset))