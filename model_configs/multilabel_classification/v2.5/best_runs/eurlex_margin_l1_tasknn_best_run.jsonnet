// Run ID: d2iw92pp

{
    "dataset_reader": {
        "type": "arff",
        "num_labels": 3993
    },
    "model": {
        "type": "multi-label-classification-with-infnet-and-scorenn-evaluation",
        "evaluation_module": {
            "type": "indexed-container",
            "constituent_samplers": [
                {
                    "gradient_descent_loop": {
                        "optimizer": {
                            "lr": 0.1,
                            "type": "sgd",
                            "weight_decay": 0
                        }
                    },
                    "log_key": "tasknn_gbi",
                    "loss_fn": {
                        "log_key": "neg.dvn_score",
                        "reduction": "none",
                        "type": "multi-label-dvn-score"
                    },
                    "output_space": {
                        "default_value": 0,
                        "num_labels": 3993,
                        "type": "multi-label-relaxed"
                    },
                    "sample_picker": {
                        "type": "best"
                    },
                    "stopping_criteria": 20,
                    "type": "gradient-based-inference"
                },
                {
                    "gradient_descent_loop": {
                        "optimizer": {
                            "lr": 0.1,
                            "type": "sgd",
                            "weight_decay": 0
                        }
                    },
                    "log_key": "random_gbi",
                    "loss_fn": {
                        "log_key": "neg.dvn_score",
                        "reduction": "none",
                        "type": "multi-label-dvn-score"
                    },
                    "output_space": {
                        "default_value": 0,
                        "num_labels": 3993,
                        "type": "multi-label-relaxed"
                    },
                    "sample_picker": {
                        "type": "best"
                    },
                    "stopping_criteria": 20,
                    "type": "gradient-based-inference"
                }
            ],
            "log_key": "evaluation"
        },
        "inference_module": {
            "type": "multi-label-inference-net-normalized",
            "cost_augmented_layer": {
                "type": "multi-label-stacked",
                "feedforward": {
                    "activations": [
                        "softplus",
                        "linear"
                    ],
                    "hidden_dims": 3993,
                    "input_dim": 7986,
                    "num_layers": 2
                },
                "normalize_y": true
            },
            "log_key": "inference_module",
            "loss_fn": {
                "type": "combination-loss",
                "constituent_losses": [
                    {
                        "inference_score_weight": 0.0016410581812096757,
                        "log_key": "neg_inference",
                        "normalize_y": true,
                        "reduction": "none",
                        "type": "multi-label-inference"
                    },
                    {
                        "log_key": "bce",
                        "reduction": "none",
                        "type": "multi-label-bce"
                    }
                ],
                "log_key": "loss",
                "loss_weights": [
                    1,
                    0.8189037148504734
                ],
                "reduction": "mean"
            }
        },
        "initializer": {
            "regexes": [
                [
                    ".*_linear_layers.*weight",
                    {
                        "nonlinearity": "relu",
                        "type": "kaiming_uniform"
                    }
                ],
                [
                    ".*linear_layers.*bias",
                    {
                        "type": "zero"
                    }
                ]
            ]
        },
        "loss_fn": {
            "type": "multi-label-margin-based",
            "log_key": "margin_loss",
            "oracle_cost_weight": 1,
            "perceptron_loss_weight": 0.0016410581812096757,
            "reduction": "mean"
        },
        "oracle_value_function": {
            "type": "manhattan",
            "differentiable": true
        },
        "sampler": {
            "type": "appending-container",
            "constituent_samplers": [],
            "log_key": "sampler"
        },
        "score_nn": {
            "type": "multi-label-classification",
            "global_score": {
                "type": "multi-label-feedforward",
                "feedforward": {
                    "activations": "softplus",
                    "hidden_dims": 300,
                    "input_dim": 3993,
                    "num_layers": 1
                }
            },
            "task_nn": {
                "type": "multi-label-classification",
                "feature_network": {
                    "activations": [
                        "softplus",
                        "softplus"
                    ],
                    "dropout": [
                        0.2,
                        0
                    ],
                    "hidden_dims": 500,
                    "input_dim": 5000,
                    "num_layers": 2
                },
                "label_embeddings": {
                    "embedding_dim": 500,
                    "vocab_namespace": "labels"
                }
            }
        },
        "task_nn": {
            "type": "multi-label-classification",
            "feature_network": {
                "activations": [
                    "softplus",
                    "softplus"
                ],
                "dropout": [
                    0.2,
                    0
                ],
                "hidden_dims": 500,
                "input_dim": 5000,
                "num_layers": 2
            },
            "label_embeddings": {
                "embedding_dim": 500,
                "vocab_namespace": "labels"
            }
        }
    },
    "train_data_path": "./data//eurlex-ev-stratified10folds-meka/Eurlex-ev-fold@(1|2|3|4|5|6)-normalised.arff",
    "validation_data_path": "./data//eurlex-ev-stratified10folds-meka/Eurlex-ev-fold@(7|8)-normalised.arff",
    "test_data_path": "./data//eurlex-ev-stratified10folds-meka/Eurlex-ev-fold@(9|10)-normalised.arff",
    "trainer": {
        "type": "gradient_descent_minimax",
        "callbacks": [
            "track_epoch_callback",
            "slurm",
            {
                "sub_callbacks": [
                    {
                        "priority": 100,
                        "type": "log_best_validation_metrics"
                    }
                ],
                "type": "wandb_allennlp"
            }
        ],
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "cuda_device": 0,
        "grad_norm": {
            "task_nn": 10
        },
        "inner_mode": "task_nn",
        "learning_rate_schedulers": {
            "task_nn": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "mode": "max",
                "patience": 5,
                "verbose": true
            }
        },
        "num_epochs": 300,
        "num_steps": {
            "score_nn": 1,
            "task_nn": 5
        },
        "optimizer": {
            "optimizers": {
                "score_nn": {
                    "type": "adamw",
                    "lr": 0.00034011646151576013,
                    "weight_decay": 1e-05
                },
                "task_nn": {
                    "type": "adamw",
                    "lr": 0.0020883077789526716,
                    "weight_decay": 1e-05
                }
            }
        },
        "patience": 20,
        "validation_metric": "+fixed_f1"
    },
    "vocabulary": {
        "type": "from_files",
        "directory": "./data//eurlex-ev-stratified10folds-meka/eurlex-ev-vocab"
    },
    "type": "train_test_log_to_wandb",
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "evaluate_on_test": true,
    "validation_dataset_reader": {
        "type": "arff",
        "num_labels": 3993
    }
}