#  SEAL 🦭

This is the official implementation for the paper [Structured Energy Network As a Loss](https://openreview.net/pdf?id=F0DowhX7_x).


# Setup

1. Clone the repo

2. Run setup enviroment bash script

```
$ chmod +x setup_env.sh
$ /bin/bash setup_env.sh
```


3. Download datasets

```
chmod +x download_datasets.sh
/bin/bash download_datasets.sh
```

4. Export environment variables

```
export CUDA_DEVICE=0 # 0 for GPU, -1 for CPU
export DATA_DIR="./data/"
export TEST=1 # for a dryrun and without uploading results to wandb
export WANDB_IGNORE_GLOBS=*\*\*\*.th,*\*\*\*.tar.gz,*\*\*.th,*\*\*.tar.gz,*\*.th,*\*.tar.gz,*.tar.gz,*.th
```

5. Training single models

The model configs are stored in `best_models_configs`. To run, for example, the cross-entropy model on bibtex dataset you would use `best_models_configs/bibtex_strat_cross-entropy_ezllp30k/config.json`. 


        From the config file, remove the "callback" that logs to wandb by removing the following lines

        ```
        "trainer": {
        ...
        "callbacks": [
            ...
            <REMOVE STARTING HERE> <REMOVE TILL HERE>
        ],
        ```

        ```
        allennlp train best_models_configs/bibtex_strat_cross-entropy_ezllp30k/config.json -s run_bibtex_cross-entropy --include-package seal --overrides {\"trainer.cuda_device\":-1}
        ```

    2. With output to wandb (see [creating account and login into wandb](https://docs.wandb.ai/quickstart#2-create-account) for details on getting started with wandb.)

        ```
        export TEST=0
        export CUDA_DEVICE=-1
        allennlp train-with-wandb model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb-run-name=<some_informative_name_for_run>  --wandb-project <project name> --wandb-entity <your wandb account name or team name> 
        ```



# Cite

```
@inproceedings{
lee2022structured,
title={Structured Energy Network As a Loss},
author={Jay-Yoon Lee and Dhruvesh Patel and Purujit Goyal and Wenlong Zhao and Zhiyang Xu and Andrew McCallum},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=F0DowhX7_x}
}
```