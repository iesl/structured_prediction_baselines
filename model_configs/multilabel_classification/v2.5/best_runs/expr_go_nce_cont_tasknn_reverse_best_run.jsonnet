// Run ID: jumk0nz2

{
    "dataset_reader": {
        "type": "arff",
        "num_labels": 4132
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
                        "num_labels": 4132,
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
                        "num_labels": 4132,
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
            "log_key": "inference_module",
            "loss_fn": {
                "type": "combination-loss",
                "constituent_losses": [
                    {
                        "log_key": "neg.nce_score",
                        "normalize_y": true,
                        "reduction": "none",
                        "type": "multi-label-score-loss"
                    },
                    {
                        "log_key": "bce",
                        "reduction": "none",
                        "type": "multi-label-bce"
                    }
                ],
                "log_key": "loss",
                "loss_weights": [
                    0.1972193528832343,
                    1
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
            "type": "multi-label-nce-ranking-with-cont-sampling",
            "log_key": "nce",
            "num_samples": 60,
            "sign": [
                null
            ],
            "std": 0.06364383288391087
        },
        "oracle_value_function": {
            "type": "per-instance-f1",
            "differentiable": false
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
                    "hidden_dims": 400,
                    "input_dim": 4132,
                    "num_layers": 1
                }
            },
            "task_nn": {
                "type": "multi-label-classification",
                "feature_network": {
                    "activations": [
                        "softplus",
                        "softplus",
                        "softplus",
                        "softplus"
                    ],
                    "dropout": [
                        0.1,
                        0.1,
                        0.1,
                        0
                    ],
                    "hidden_dims": 500,
                    "input_dim": 561,
                    "num_layers": 4
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
                    "softplus",
                    "softplus",
                    "softplus"
                ],
                "dropout": [
                    0.1,
                    0.1,
                    0.1,
                    0
                ],
                "hidden_dims": 500,
                "input_dim": 561,
                "num_layers": 4
            },
            "label_embeddings": {
                "embedding_dim": 500,
                "vocab_namespace": "labels"
            }
        }
    },
    "train_data_path": "./data//expr_go/train-normalized.arff",
    "validation_data_path": "./data//expr_go/dev-normalized.arff",
    "test_data_path": "./data//expr_go/test-normalized.arff",
    "trainer": {
        "type": "gradient_descent_minimax",
        "callbacks": [
            "track_epoch_callback",
            "slurm",
            {
                "save_model_archive": false,
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
        "inner_mode": "score_nn",
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
            "score_nn": 3,
            "task_nn": 1
        },
        "optimizer": {
            "optimizers": {
                "score_nn": {
                    "type": "adamw",
                    "lr": 7.327690541607274e-05,
                    "weight_decay": 1e-05
                },
                "task_nn": {
                    "type": "adamw",
                    "lr": 0.0001444271272068573,
                    "weight_decay": 1e-05
                }
            }
        },
        "patience": 20,
        "validation_metric": "+fixed_f1"
    },
    "type": "train_test_log_to_wandb",
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "evaluate_on_test": true,
    "validation_dataset_reader": {
        "type": "arff",
        "num_labels": 4132
    }
}