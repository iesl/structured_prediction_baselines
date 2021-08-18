#!/:bin/sh

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables
export ff_dropout=0.2
export ff_linear_layers=2
export ff_weight_decay=0.1
export score_loss_weight=1

source /mnt/nfs/scratch1/jaylee/repository/structured_prediction/.venv_allennlp-models-structured-prediction/bin/activate
