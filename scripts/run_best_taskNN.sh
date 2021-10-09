#!/bin/sh
#SBATCH --job-name=best_taskNN
#SBATCH --output=../logs/best_taskNN-%j.out
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node030,node051,node095

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables

# export ALLENNLP_DEBUG=1
allennlp train_with_wandb \
  model_configs/multilabel_classification/v2.5/best_xtropy_config/spo_go_tasknn.json \
  --include-package=structured_prediction_baselines \
  --wandb_name="expr_go_best_tasknn_run" \
  --wandb_project="mlc" \
  --wandb_entity="score-based-learning" \
  --wandb_tags="expr_go, best_tasknn_run"
# --wandb_name=revNCE_test_with_pretrainedTaskNN_lossAsMetric \

# /mnt/nfs/scratch1/jaylee/repository/structured_prediction/model_configs/multilabel_classification/bibtex_revNCE_zeroScoreLoss.jsonnet \
