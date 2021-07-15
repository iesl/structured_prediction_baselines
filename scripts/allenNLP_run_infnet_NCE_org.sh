#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=logs/infnet-wNCE-orgrun-%j.out
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/


# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce10_org_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs10_org \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s10"


# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce20_org_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs20_org \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s20"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce30_org_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs30_org \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s30"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce40_org_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs40_org \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s40"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce50_org_config.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs50_org \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s50"

##################################################################


# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce10_negpn_discrete.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs10_negpn_discrete \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s10"


allennlp train_with_wandb \
  model_configs/multilabel_classification/NCE_old_comparison/nce20_negpn_discrete.json \
  --include-package=structured_prediction_baselines \
  --wandb_run_name=bibtex_infnet_wNCEs20_negpn_discrete \
  --wandb_project structured_prediction_baselines \
  --wandb_entity score-based-learning \
  --wandb_tags="bibtex,infnet_wNCE,with_sampling,s20"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce30_negpn_discrete.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs30_negpn_discrete \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s30"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce40_negpn_discrete.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs40_negpn_discrete \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s40"

# wandb_allennlp --subcommand=train \
#   --config_file=model_configs/multilabel_classification/NCE_old_comparison/nce50_negpn_discrete.json \
#   --include-package=structured_prediction_baselines \
#   --wandb_run_name=bibtex_infnet_wNCEs50_negpn_discrete \q
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wNCE,with_sampling,s50"

