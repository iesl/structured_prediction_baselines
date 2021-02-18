# Structure Prediction Baselines Using AllenNLP

Implements baselines for tasks like POS tagging, NER and SRL.

# Setup for Development

1. Clone the repo 

2. Run setup enviroment bash script

```
/bin/bash setup_env.sh
```

# Running the models

1. Download datasets

    ```
    /bin/bash download_datasets.sh
    ```

2. Export enviroment variables

    ```
    export CUDA_DEVICE=0 # 0 for GPU, -1 for CPU
    export TEST=1 # for a dryrun and without uploading results to wandb
    export WANDB_IGNORE_GLOBS=**/*.th
    ```

3. Training single models

    1. Using slurm (on gypsum)
  
      Open `single_run.sh`, make modifications as needed, close and submit job using `sbatch single_run.sh`. Do not push local updates to this file to the repo.


    2. On you local machine

        1. Without sending output to wandb

        ```
        export TEST=1
        export CUDA_DEVICE=-1
        allennlp train <path_to_config> -s <path to serialization dir> --include_package structure_prediction_baselines
        ```

        2. With output to wandb

        ```
        export TEST=0
        export CUDA_DEVICE=-1
        wandb_allennlp --subcommand=train --config_file=model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb_run_name=<some_informative_name_for_run>  --wandb_project structure_prediction_baselines --wandb_entity score-based-learning --wandb_tags=baselines,as_reported
        ```

# Directory Structure



# Contributors



# Cite

