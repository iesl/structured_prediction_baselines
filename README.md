# SEAL framework

This code repository contains the implementation of SEAL framework. A particular instantiation of the framework for multilabel classification is presented.


# Setup

1. Clone the repo

2. Run setup environment bash script

```
/bin/bash setup_env.sh
```

# Training the models

1. Download datasets

    ```
    /bin/bash download_datasets.sh
    ```

2. The configs for all the models used in the paper are stored in `model_configs/multilabel_classification/v2.5/best_runs/`. To train any model, use the native `allennlp train` command. For instance:


```
allennlp train <path_to_config> -s <path to serialization dir> --include_package structure_prediction_baselines
```


3. Running hyperparameter sweeps


The complete setup used to search for best hyper-parameters uses Bayesian search through  [Weights and Biases](https://docs.wandb.ai/sweeps) library. This, along with the wandb project with the runs will be opened once the paper is accepted.


# Cite

Under submission.
