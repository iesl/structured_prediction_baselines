#!/bin/sh

export TEST=0
export CUDA_DEVICE=0
export DATA_DIR=./data/

# Model Variables
export ff_dropout=0.2
export ff_linear_layers=2
export ff_weight_decay=0.1
export score_loss_weight=1


#wandb_allennlp --subcommand=train --config_file=model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb_run_name=<some_informative_name_for_run>  --wandb_project structure_prediction_baselines --wandb_entity score-based-learning --wandb_tags=baselines,as_reported

wandb_allennlp --subcommand=train \
  --config_file=./model_configs/multilabel_classification/bibtex_inference_net_wDVN.jsonnet \
  --include-package=structured_prediction_baselines \
  --wandb_run_name=bibtex-infent-dvn \
  --wandb_project structured_prediction_baselines \
  --wandb_entity score-based-learning \
  --wandb_tags=bibtex,infnet_wDVN