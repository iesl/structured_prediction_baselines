# Structure Prediction Baselines Using AllenNLP

Implements baselines for tasks like POS tagging, NER and SRL.

# Setup for Development

1. Clone the repo

2. Run setup enviroment bash script

```
/bin/bash setup_env.sh
```
From allennlp v2.5, additional steps are required: 
```
pip install --upgrade allennlp==2.5.0
```
Install wandb allennlp to experiment with  wandb (if not already installed):
```
pip install wandb_allennlp
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
    export WANDB_IGNORE_GLOBS=*\*\*\*.th,*\*\*\*.tar.gz,*\*\*.th,*\*\*.tar.gz,*\*.th,*\*.tar.gz,*.tar.gz,*.th
    export DATA_DIR="./data/"
    ```

3. Training single models

    1. Using slurm (on gypsum)

      Open `single_run.sh`, make modifications as needed, close and submit job using `sbatch single_run.sh`. Do not push local updates to this file to the repo.


    2. On you local machine

        1. Without sending output to wandb

        ```
        export TEST=1
        export CUDA_DEVICE=-1
        allennlp train <path_to_config> -s <path to serialization dir> --include-package structured_prediction_baselines

        ```

        2. With output to wandb (see [creating account and login into wandb](https://docs.wandb.ai/quickstart#2-create-account) for details on getting started with wandb.)

        ```
        export TEST=0
        export CUDA_DEVICE=-1
        allennlp train-with-wandb --config_file=model_configs/<path_to_config_file> --include-package=structured_prediction_baselines --wandb_run_name=<some_informative_name_for_run>  --wandb_project structured_prediction_baselines --wandb_entity <your wandb account name or team name> -- some hyperparameters to add (please refer to 5)
        ```

4. Running hyperparameter sweeps

    1. Create a sweep using a sweep config file. See `sweep_configs` directory for examples. Refer sweeps documentation [here](https://docs.wandb.ai/sweeps).

    ```
    wandb sweep -e <your wandb account name or team name> -p baselines sweep_configs/<path/to/config.yaml>

    < you will see an alpha numeric sweep_id as output here. Copy it.>
    ```

    2. Start search agent on slurm using the following (This script will internally submit to sbatch. So you can run this command on the head node eventhough it is a python script because it exit withing seconds.)

    ```
    export TEST=0
    python slurm_wandb_agent.py <sweep_id> -p baselines -e <your wandb account name or team name> --num-jobs 5 -f --edit-sbatch --edit-srun
    ```

    You can use `squeue` to see the running agents on nodes. You can rerun this command to start more agents.

5. Running Example

# Directory Structure

## Modules

The following diagram give input/output and compositional structure of the base model.

```
Model(x, labels=None)
|-Sampler(x, labels, ScoreNN, CostFunction) -> (y_hat, probability)
|-ScoreNN(x,y) -> score
|-OracleValueFunction(y,labels) -> oracle cost
|-Loss(oracle costs, scores of labels, scores of y_hat, sample probabilities)
```

The model owns `score_nn` and `oracle_value_function` but during model construction, the references to these objects are passed to `sampler` and `loss` as shown below:

```
                              Model Construction



┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
│                                     │ │                                     │ │                                     │
│                                     │ │                                     │ │                                     │
│ inference_module:Sampler            │ │     sampler: Sampler                │ │        loss: Loss                   │
├─────────────┬───────────────────────┤ ├─────────────┬───────────────────────┤ ├─────────────┬───────────────────────┤
│Ref: score_nn│Ref:oracle_value_func. │ │Ref: score_nn│Ref:oracle_value_func. │ │Ref: score_nn│Ref:oracle_value_func. │
└──────▲──────┴──────────▲────────────┘ └──────▲──────┴──────────▲────────────┘ └────▲────────┴────────▲──────────────┘
       ▲                 ▲                     │                 │                   │                 │
       │                 │                     │                 └───────────────────┼──────┐          │
       │                 │                     │                                     │      │          │
       └─────────────────┼────────────────┐    │                                     │      │        ┌─┘
                         │                │    └────────────────────┐    ┌───────────┘      │        │
                         │                │                         │    │                  │        │
                         └────────────────┼─────────────────────────┼────┼───────────┐      │        │
                                          │                         │    │           │      │        │
                                        ┌─┴─────────────────────────┴────┴────┐  ┌───┴──────┴────────┴────────────────┐
                                        │                                     │  │                                    │
                                        │    score_nn : ScoreNN               │  │ oracle_value_function :            │
                                        │                                     │  │          OracelValueFunction       │
                                        │                                     │  │                                    │
                                        └─────────────────────────────────────┘  └────────────────────────────────────┘
```


Flow of data and computations happening in the model are as follows:

```
┌─────────────────────────────────────────────────────────┐  ┌───────────────────────────────────────────────────────────────────────────────┐
│         Training Flow                                   │  │    Validation/evaluation flow                                                 │
│                                                         │  │                                                                               │
│                                                         │  │                                                                               │
│                                                         │  │                                                                               │
│                   scalar Tensor                         │  │                scalar Tensor                           metrics                │
│                      ▲                                  │  │                   ▲                                       ▲                   │
│                      │                                  │  │                   │                                       │                   │
│     ┌────────────────┴────────────────────┐             │  │  ┌────────────────┴────────────────────┐    ┌─────────────┴────────────────┐  │
│     │                                     │             │  │  │                                     │    │                              │  │
│     │                                     │             │  │  │                                     │    │   metrics: List[Metric]      │  │
│     │        loss: Loss                   │             │  │  │        loss: Loss                   │    │                              │  │
│     ├─────────────┬───────────────────────┤             │  │  ├─────────────┬───────────────────────┤    └─────────────▲────────────────┘  │
│     │Ref: score_nn│Ref:oracle_value_func. │             │  │  │Ref: score_nn│Ref:oracle_value_func. │                  │                   │
│     └─────────────┴───────────────────────┘             │  │  └─────────────┴───────────────────────┘                  │                   │
│                      ▲                                  │  │                   ▲                                       │                   │
│                      │                                  │  │                   │                                       │                   │
│                      │                                  │  │                   └───────────────────────────────────────┘                   │
│         (y_hat: Tensor(batch,num_samples,...),          │  │   (y_pred:   Tensor(batch,num_samples,...),                                   │
│           y_probs: Optional[Tensor(batch,num_samples)]) │  │        y_probs: Optional[Tensor(batch,num_samples)])                          │
│                                                         │  │                                                                               │
│                      ▲                                  │  │                   ▲                                                           │
│                      │                                  │  │                   │                                                           │
│                      │                                  │  │                   │                                                           │
│      ┌───────────────┴─────────────────────┐            │  │                                                                               │
│      │                                     │            │  │   ┌─────────────────────────────────────┐                                     │
│      │                                     │            │  │   │                                     │                                     │
│      │     sampler: Sampler                │            │  │   │                                     │                                     │
│      ├─────────────┬───────────────────────┤            │  │   │ infelence_module:Sampler            │                                     │
│      │Ref: score_nn│Ref:oracle_value_func. │            │  │   ├─────────────┬───────────────────────┤                                     │
│      └─────────────┴───────────────────────┘            │  │   │Ref: score_nn│Ref:oracle_value_func. │                                     │
│                                                         │  │   └─────────────┴───────────────────────┘                                     │
│                     ▲                                   │  │                  ▲                                                            │
│                     │                                   │  │                  │                                                            │
│                     │                                   │  │                  │                                                            │
│                     │                                   │  │                  │                                                            │
│                     │                                   │  │                  │                                                            │
│              (x: Any, y: Tensor(batch, ...) )           │  │           (x: Any, y: None)                                                   │
│                                                         │  │                                                                               │
└─────────────────────────────────────────────────────────┘  └───────────────────────────────────────────────────────────────────────────────┘
```


1. **Sampler**:

Given input x, returns samples of shape `(batch, num_samples or 1,...)`  and optionally their corresponding probabilities of shape `(batch, num_samples)`. **The sampler can do and return different things during training and test.** We want the probabilities specifically in the [[Minimum Risk Training for Neural Machine Translation|MRT setting]]. The cases that sampler will cover include:

	1. Inference network or `TaskNN`, where we just take the input x and produce either a relaxed output of shape `(batch, 1, ...)` or samples of shape `(batch, num_samples, ...)`. Note, when we include `inference_net: TaskNN` here, we also need to update its parameters, right here. So when sampler uses `inference_net: TaskNN`, we also need to give it an instance of `Optimizer` to update its parameters.

	2. Cost-augmented inference module that uses `ScoreNN` and `OracleValueFunction` to produce a single relaxed output or samples.

	3. Adversarial sampler which again uses `ScoreNN` and `OracleValueFunction` to produce adversarial samples. (I see no difference between this and the cost augmented inference)

	4. Random samples biased towards `labels`.

	5. In the case of MRT style training, it can be beam search.

	6. In the case of vanilla feedforward model, one can just return the logits with shape `(batch, 1, ... )`

2. **ScoreNN**:

This is the parameterized value network or  negative energy function that takes in `(x,y)` and produces a value or score or negative-energy (higher the better). The shape of `y` will be `(batch, num_samples or 1, ...)` and the shape of output score will be `(batch, num_samples or 1)`.

3. **OracelValueFunction**:

Either a differentiable (w.r.t `y`) or non-differentiable function that takes in true label and an set of arbitrary `y`'s(either discrete in case of non-differentiable cost) or a continuous relaxations. The shape of input `y` will be `(batch, num_samples or 1, ...)`.

4. **Loss**:

Take in x, the output of the sampler, true labels, and references to ScoreNN and OracelValueFunction to produce a loss to back prop on.







# Cite
