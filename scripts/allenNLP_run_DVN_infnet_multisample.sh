#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=logs/infnet_wDVN-%j.out
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=1
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables
export ff_dropout=0.2
export ff_linear_layers=2
export ff_weight_decay=0.1
export score_loss_weight=1


#wandb_allennlp --subcommand=train \
# --config_file=model_configs/<path_to_config_file> \
# --include-package=structured_prediction_baselines \
# --wandb_run_name=<some_informative_name_for_run>  \
# --wandb_project structure_prediction_baselines \
# --wandb_entity score-based-learning \
# --wandb_tags=baselines,as_reported

## running with jsonnet
#   --config_file=./model_configs/multilabel_classification/bibtex_inference_net_wDVN_samples.jsonnet \
#   --config_file=model_configs/multilabel_classification/infnet_wDVN_samples.config \

## running with wandb
# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/infnet_wDVN_samples.config \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_inference_net_wDVN \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wDVN,without_sampling"

wandb_allennlp --subcommand=train \
  --config_file=model_configs/multilabel_classification/bibtex_inference_net_wDVN_samples_reducefix.jsonnet \
  --include-package=structured_prediction_baselines \
  --wandb_run_name=bibtex_infnet_wDVN_samples_reducefix \
  --wandb_project structured_prediction_baselines \
  --wandb_entity score-based-learning \
  --wandb_tags="bibtex,infnet_wDVN,with_sampling,testing"


## running with allennlp
# allennlp train <path_to_config> -s <path to serialization dir> --include_package structure_prediction_baselines

# with samples
# allennlp train \
#   model_configs/multilabel_classification/Infnet_wDVN_samples.config \
#   -f -s test_infnetDVN_samples \
#   --include-package=structured_prediction_baselines

# with samples + prob
# allennlp train \
#   model_configs/multilabel_classification/Infnet_wDVN_samples_prob.config \
#   -f -s test_infnetDVN_samples_prob \
#   --include-package=structured_prediction_baselines

# with smaples + prob + debug original dvn + infnet.
# allennlp train \
#   model_configs/multilabel_classification/Infnet_wDVN_samples_debug.config \
#   -f -s test_infnetDVN_samples_debug \
#   --include-package=structured_prediction_baselines



# allennlp train \
#   model_configs/multilabel_classification/Infnet_wDVN.config \
#   -f -s test_infnetDVN_samples \
#   --include-package=structured_prediction_baselines
