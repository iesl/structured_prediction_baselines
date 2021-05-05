#!/bin/sh
#SBATCH --job-name=tn_high_reg
#SBATCH -o logs/%j.log
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=0
export CUDA_DEVICE=0
export WANDB_IGNORE_GLOBS=**/*.th
allennlp train model_configs/multilabel_classification/bibtex_dvn_basic_original_data.jsonnet -s out_jy2  --include-package structured_prediction_baselines
