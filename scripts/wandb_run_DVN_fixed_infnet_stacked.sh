#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=logs/infnet_wDVN_stacked-%j.out
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

source .venv_allennlp-models-structured-prediction/bin/activate 

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables
export ff_dropout=0.5
export ff_linear_layers=3
export ff_weight_decay=0.001
export cross_entorpy_loss_weight=1.0
export inference_score_weight=0.8
export margin_based_loss_weight=0.5
# export score_loss_weight=0.8

## Template ####################################################
#wandb_allennlp --subcommand=train \
# --config_file=model_configs/<path_to_config_file> \
# --include-package=structured_prediction_baselines \
# --wandb_run_name=<some_informative_name_for_run>  \
# --wandb_project structure_prediction_baselines \
# --wandb_entity score-based-learning \
# --wandb_tags=baselines,as_reported
################################################################

#################################################################
## running with jsonnet
#   --config_file=./model_configs/multilabel_classification/bibtex_inference_net_wDVN.jsonnet \
## running wiht config file
#   --config_file=./model_configs/multilabel_classification/Infnet_wDVN_stacked.config \
#################################################################

## main run.
wandb_allennlp --subcommand=train \
  --config_file=./model_configs/multilabel_classification/bibtex_inference_net_wDVN_fixed.jsonnet \
  --include-package=structured_prediction_baselines \
  --wandb_run_name=bibtex_inference_net_wDVN_stacked_diffmargin \
  --wandb_project structured_prediction_baselines \
  --wandb_entity score-based-learning \
  --wandb_tags="bibtex,infnet_wDVN_stacked,without_sampling"