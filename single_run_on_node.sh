#!/bin/bash

export TEST=1
export CUDA_DEVICE=0
export DATA_DIR=./data/
export WANDB_IGNORE_GLOBS=*\*\*\*.th,*\*\*\*.tar.gz,*\*\*.th,*\*\*.tar.gz,*\*.th,*\*.tar.gz,*.tar.gz,*.th

export n_epochs=40
export batch_size=32
export ff_hidden_dim=768
export n_ff_linear_layers=1
export task_nn_dropout=0.1
export task_nn_lr=0.00005
export task_nn_weight_decay=0.001

#export global_score_hidden_dim=200
#export cross_entropy_loss_weight=1.0
#export inference_score_weight=0
#export dvn_score_loss_weight=0
#export num_samples=10

export EXP_NAME=trec_fine_bert
export MODEL_CONFIG=model_configs/classification/${EXP_NAME}.jsonnet
export EXP_DIR=/work/wenlongzhao_umass_edu/energy/structured_prediction_baselines_distillation/logs/${EXP_NAME}
mkdir -p ${EXP_DIR}
export SERIALIZATION_DIR=${EXP_DIR}/on_node

# without sending to wandb
allennlp train ${MODEL_CONFIG} -s ${SERIALIZATION_DIR} --include-package structured_prediction_baselines > ${SERIALIZATION_DIR}.log
