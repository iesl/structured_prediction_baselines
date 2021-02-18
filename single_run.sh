#!/bin/sh
#SBATCH --job-name=tn_high_reg
#SBATCH -o logs/%j.log
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --exclude=node072,node035,node030

wandb_allennlp --subcommand=train --config_file=model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb_run_name=<some_informative_name_for_run>  --wandb_project structure_prediction_baselines --wandb_entity score-based-learning --wandb_tags=baselines,as_reported

