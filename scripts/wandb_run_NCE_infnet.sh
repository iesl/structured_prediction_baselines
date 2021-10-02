#!/bin/sh
#SBATCH --job-name=structured_prediction
#SBATCH --output=../logs/infnet_wNCE-%j.out
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables
# export dataset_name=genbase
export ff_hidden=400
export ff_dropout=0.5
export ff_dropout_10x=5
export ff_linear_layers=2
export ff_weight_decay=0.0001
export global_score_hidden_dim=200
export cross_entropy_loss_weight=1.0
export inference_score_weight=0
export dvn_score_loss_weight=0
export num_samples=10
export stopping_criteria=0
export task_nn_steps=1
export score_nn_steps=2

#wandb_allennlp --subcommand=train \
# --config_file=model_configs/<path_to_config_file> \
# --include-package=structured_prediction_baselines \
# --wandb_run_name=<some_informative_name_for_run>  \
# --wandb_project structure_prediction_baselines \
# --wandb_entity score-based-learning \
# --wandb_tags=baselines,as_reported

## running with jsonnet
#   --config_file=./model_configs/multilabel_classification/bibtex_inference_net_wDVN.jsonnet \
#   --config_file=model_configs/multilabel_classification/Infnet_wDVN.config \
# --config_file=./model_configs/multilabel_classification/bibtex_inference_net_wDVN_fscratch.jsonnet \
## running with config file.

# infnet + NCE 
# allennlp train_with_wandb \
#   model_configs/multilabel_classification/bibtex_infnet_wNCE_infscore.jsonnet \
#   --include-package=structured_prediction_baselines \
#   --wandb_name=bibtex_inference_net_wNCE_test \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,infnet_wDVN,without_sampling"

# # revNCE with task-NN zero loss + pre-trained task-NN
# allennlp train_with_wandb \
#   model_configs/multilabel_classification/bibtex_revNCE_zerotasknn.jsonnet \
#   --include-package=structured_prediction_baselines \
#   --wandb_name="optimizer=adam (lr!=0, wd!=0) & typical loss  & main_wd!=0" \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,revNCE testing"
# # --wandb_name=revNCE_test_with_pretrainedTaskNN_lossAsMetric \

# NCE with task-NN zero loss + pre-trained task-NN
# allennlp train_with_wandb \
#   model_configs/multilabel_classification/bibtex_NCE_zerotasknn.jsonnet \
#   --include-package=structured_prediction_baselines \
#   --wandb_name=NCE_test_with_pretrainedTaskNN_lossAsMetric \
#   --wandb_project structured_prediction_baselines \
#   --wandb_entity score-based-learning \
#   --wandb_tags="bibtex,NCE testing"

# revNCE: pre-trained score-NN + no update on score-NN (zero loss)
# export ALLENNLP_DEBUG=1
allennlp train_with_wandb \
  model_configs/multilabel_classification/v2.5/gendata_nce_discrete_on_tasknn_reverse.jsonnet \
  --include-package=structured_prediction_baselines \
  --wandb_name="NCE_on_taskNN_test" \
  --wandb_project="mlc" \
  --wandb_entity="score-based-learning" \
  --wandb_tags="bibtex,revNCE testing"
# --wandb_name=revNCE_test_with_pretrainedTaskNN_lossAsMetric \

# /mnt/nfs/scratch1/jaylee/repository/structured_prediction/model_configs/multilabel_classification/bibtex_revNCE_zeroScoreLoss.jsonnet \