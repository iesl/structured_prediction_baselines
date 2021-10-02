from structured_prediction_baselines.dataset_readers.multilabel_classification.arff_reader import ARFFReader                                                                                            
from allennlp.training.util import data_loaders_from_params
from allennlp.common.params import Params
import argparse
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="count training, validation and test instance from allennlp json config."
    )
    parser.add_argument("input_file", type=Path, help="Path to json config")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    params  = Params.from_file(args.input_file)
    data_loaders = data_loaders_from_params(params, train=True, validation=True, test=True)                                                                                                                 
    for type_, loader in data_loaders.items():
        print(f"Number of {type_} instances: {sum(1 for _ in loader.iter_instances())}")

