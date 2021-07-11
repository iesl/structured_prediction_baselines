#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=logs/infnet-wNCE-s50-%j.out
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=1
export CUDA_DEVICE=0
export DATA_DIR=./data/

## running with jsonnet or config.
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

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce10_negpn_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs10_negpn \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s10"


# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce20_negpn_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs20_negpn \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s20"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce30_negpn_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs30_negpn \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s30"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce40_negpn_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs40_negpn \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s40"

wandb_allennlp --subcommand=train \
  --config_file=model_configs/multilabel_classification/NCE_comparison/nce50_negpn_config.json \
  --include-package=structured_prediction_baselines \
  --wandb_run_name=bibtex_infnet_wNCEs50_negpn \
  --wandb_project structured_prediction_baselines \
  --wandb_entity score-based-learning \
  --wandb_tags="bibtex,infnet_wNCE,with_sampling,s50"