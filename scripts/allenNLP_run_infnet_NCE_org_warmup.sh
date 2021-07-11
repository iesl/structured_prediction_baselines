#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=logs/infnet-wNCE-org-warmup-sctest-%j.out
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=1
export CUDA_DEVICE=0
export DATA_DIR=./data/


# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce10_org_warmup.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs10_org_warmup_sc3 \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s10"


# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce20_org_warmup.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs20_org_warmup_sc3 \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s20"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce30_org_warmup.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs30_org_warmup_sc3 \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s30"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_comparison/nce40_org_warmup.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs40_org_warmup_sc3 \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s40"

wandb_allennlp --subcommand=train \
  --config_file=model_configs/multilabel_classification/NCE_comparison/nce50_org_warmup.json \
  --include-package=structured_prediction_baselines \
  --wandb_run_name=bibtex_infnet_wNCEs50_org_warmup_sc3 \
  --wandb_project structured_prediction_baselines \
  --wandb_entity score-based-learning \
  --wandb_tags="bibtex,infnet_wNCE,with_sampling,s50"