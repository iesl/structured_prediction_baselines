import argparse
import csv
import os
import sys
import time
import numpy as np

path = os.path.dirname(os.path.abspath('.'))
if path not in sys.path:
    sys.path.append(path)

import torch
from allennlp.models import Model
from allennlp.common import Params
from allennlp.common import util as common_util, Tqdm
from allennlp.nn import util

common_util.import_module_and_submodules('structured_prediction_baselines')

from allennlp.data import (
    DataLoader,
    DatasetReader,
    Vocabulary,
)

from allennlp_models.structured_prediction.models.srl import convert_bio_tags_to_conll_format

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str,
                    help='path to the directory containing model wts, config and vocab', required=True)
parser.add_argument('--data_dir', type=str,
                    help='path to the directory containing the data directory which holds all the datasets', required=True)
parser.add_argument('--output_file', type=str,
                    help='file name will be appended with val and test and saved as csv files', required=True)
parser.add_argument('--output_dir', type=str,
                    help='path to the directory to save output files', required=True)

args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
val_output_file = output_dir + '/' + args.output_file + "_val.csv"
test_output_file = output_dir + '/' + args.output_file + "_test.csv"
model_dir = args.model_dir  # path to the directory containing config, model weights and vocab
config_file = os.path.join(model_dir, "config.json")
vocabulary_dir = os.path.join(model_dir, "vocabulary")
weights_file = os.path.join(model_dir, "best.th")
device = 0 if torch.cuda.is_available() else -1

loaded_params = Params.from_file(config_file)
loaded_model = Model.load(loaded_params, model_dir, weights_file, cuda_device=device)
loaded_model.eval()

dataset_reader = DatasetReader.from_params(loaded_params.get("dataset_reader"))

data_loader_params = loaded_params.get("data_loader")
# train_data_loader = DataLoader.from_params(
#     params=data_loader_params.duplicate(),
#     reader=dataset_reader,
#     data_path=data_dir + loaded_params.get("train_data_path")
# )

dev_data_loader = DataLoader.from_params(
    reader=dataset_reader,
    data_path=data_dir + loaded_params.get("validation_data_path"),
    params=data_loader_params.duplicate(),
)

test_data_loader = DataLoader.from_params(
    reader=dataset_reader,
    data_path=data_dir + loaded_params.get("test_data_path"),
    params=data_loader_params.duplicate(),
)

# vocab = Vocabulary.from_instances(train_data_loader.iter_instances())
# vocab = Vocabulary.from_files_and_instances(train_data_loader.iter_instances(), vocabulary_dir)
vocab = loaded_model.vocab
dev_data_loader.index_with(vocab)
test_data_loader.index_with(vocab)
val_generator_tqdm = Tqdm.tqdm(dev_data_loader)
val_pred = []
for batch in val_generator_tqdm:
    with torch.no_grad():
        batch = util.move_to_device(batch, device=device)
        batch_mask = util.get_text_field_mask(batch['tokens'])
        batch_offsets = [
                    x["offsets"] for x in batch['metadata']
                ]
        batch['labels'] = None
        batch_pred = loaded_model(**batch, mode=None)['y_pred']
        batch_word_tags, _, _, _ = loaded_model.constrained_decode(batch_pred, batch_mask, batch_offsets)
        batch_pred_tags = [
                convert_bio_tags_to_conll_format(tags) for tags in batch_word_tags
            ]
        val_pred.append(batch_pred_tags)

with open(val_output_file, 'w') as f:
    write = csv.writer(f)
    write.writerows(val_pred)

test_generator_tqdm = Tqdm.tqdm(test_data_loader)
test_pred = []
for batch in test_generator_tqdm:
    with torch.no_grad():
        batch = util.move_to_device(batch, device=device)
        batch_mask = batch['mask']
        batch_offsets = [
            x["offsets"] for x in batch['metadata']
        ]
        batch['labels'] = None
        batch_pred = loaded_model(**batch, mode=None)['y_pred']
        batch_word_tags, _, _, _ = loaded_model.constrained_decode(batch_pred, batch_mask, batch_offsets)
        batch_pred_tags = [
            convert_bio_tags_to_conll_format(tags) for tags in batch_word_tags
        ]
        test_pred.append(batch_pred_tags)

with open(test_output_file, 'w') as f:
    write = csv.writer(f)
    write.writerows(test_pred)

