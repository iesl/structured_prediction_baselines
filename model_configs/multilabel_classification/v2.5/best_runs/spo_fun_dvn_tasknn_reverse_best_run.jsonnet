// Run ID: 1zmnow18

{
    "dataset_reader": {
        "type": "arff",
        "num_labels": 500
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
                        "num_labels": 500,
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
                        "num_labels": 500,
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
                        "log_key": "neg.dvn_score",
                        "normalize_y": true,
                        "reduction": "none",
                        "type": "multi-label-dvn-score"
                    },
                    {
                        "log_key": "bce",
                        "reduction": "none",
                        "type": "multi-label-bce"
                    }
                ],
                "log_key": "loss",
                "loss_weights": [
                    9.048789321214336,
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
            "type": "multi-label-dvn-bce",
            "log_key": "dvn_bce"
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
                    "hidden_dims": 300,
                    "input_dim": 500,
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
                    "hidden_dims": 300,
                    "input_dim": 86,
                    "num_layers": 4
                },
                "label_embeddings": {
                    "embedding_dim": 300,
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
                "hidden_dims": 300,
                "input_dim": 86,
                "num_layers": 4
            },
            "label_embeddings": {
                "embedding_dim": 300,
                "vocab_namespace": "labels"
            }
        }
    },
    "train_data_path": "./data//spo_fun/train-normalized.arff",
    "validation_data_path": "./data//spo_fun/dev-normalized.arff",
    "test_data_path": "./data//spo_fun/test-normalized.arff",
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
            "score_nn": 12,
            "task_nn": 5
        },
        "optimizer": {
            "optimizers": {
                "score_nn": {
                    "type": "adamw",
                    "lr": 0.0013897170803032388,
                    "weight_decay": 1e-05
                },
                "task_nn": {
                    "type": "adamw",
                    "lr": 0.0001548727897653195,
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
        "num_labels": 500
    }
}