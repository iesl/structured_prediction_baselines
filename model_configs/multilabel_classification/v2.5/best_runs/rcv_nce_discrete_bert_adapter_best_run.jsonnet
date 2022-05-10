// Run ID: dl627csk

{
    "dataset_reader": {
        "type": "rcv",
        "token_indexers": {
            "x": {
                "type": "pretrained_transformer",
                "model_name": "bert-base-uncased"
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "max_length": 512,
            "model_name": "bert-base-uncased"
        }
    },
    "model": {
        "type": "multi-label-classification-with-infnet",
        "inference_module": {
            "type": "multi-label-basic",
            "log_key": "inference_module",
            "loss_fn": {
                "type": "combination-loss",
                "constituent_losses": [
                    {
                        "log_key": "neg_nce_score",
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
                    3,
                    1
                ],
                "reduction": "mean"
            }
        },
        "initializer": {
            "regexes": [
                [
                    ".*feedforward._linear_layers.*weight",
                    {
                        "nonlinearity": "relu",
                        "type": "kaiming_uniform"
                    }
                ],
                [
                    ".*feedforward._linear_layers.*bias",
                    {
                        "type": "zero"
                    }
                ]
            ]
        },
        "loss_fn": {
            "type": "multi-label-nce-ranking-with-discrete-sampling",
            "log_key": "nce",
            "num_samples": 600,
            "sign": "-"
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
                    "hidden_dims": 600,
                    "input_dim": 103,
                    "num_layers": 1
                }
            },
            "task_nn": {
                "type": "multi-label-text-classification",
                "feature_network": {
                    "feedforward": {
                        "activations": [
                            "softplus",
                            "softplus"
                        ],
                        "dropout": [
                            0.1,
                            0
                        ],
                        "hidden_dims": [
                            1536,
                            768
                        ],
                        "input_dim": 768,
                        "num_layers": 2
                    },
                    "final_dropout": 0,
                    "seq2vec_encoder": {
                        "type": "bert_pooler",
                        "pretrained_model": "bert-base-uncased"
                    },
                    "text_field_embedder": {
                        "token_embedders": {
                            "x": {
                                "type": "pretrained_transformer_with_adapter",
                                "model_name": "bert-base-uncased"
                            }
                        }
                    }
                },
                "label_embeddings": {
                    "embedding_dim": 768,
                    "vocab_namespace": "labels"
                }
            }
        },
        "task_nn": {
            "type": "multi-label-text-classification",
            "feature_network": {
                "feedforward": {
                    "activations": [
                        "softplus",
                        "softplus"
                    ],
                    "dropout": [
                        0.1,
                        0
                    ],
                    "hidden_dims": [
                        1536,
                        768
                    ],
                    "input_dim": 768,
                    "num_layers": 2
                },
                "final_dropout": 0,
                "seq2vec_encoder": {
                    "type": "bert_pooler",
                    "pretrained_model": "bert-base-uncased"
                },
                "text_field_embedder": {
                    "token_embedders": {
                        "x": {
                            "type": "pretrained_transformer_with_adapter",
                            "model_name": "bert-base-uncased"
                        }
                    }
                }
            },
            "label_embeddings": {
                "embedding_dim": 768,
                "vocab_namespace": "labels"
            }
        }
    },
    "train_data_path": "data/rcv1/train_val_fold_@(0|1|2|3|4|5).jsonl",
    "validation_data_path": "data/rcv1/train_val_fold_@(6|7|8|9).jsonl",
    "test_data_path": "data/rcv1/test_fold_0.jsonl",
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
                "type": "wandb_allennlp",
                "watch_model": false
            }
        ],
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "cuda_device": 0,
        "grad_norm": {
            "score_nn": 1,
            "task_nn": 1
        },
        "inner_mode": "score_nn",
        "learning_rate_schedulers": {
            "task_nn": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "mode": "max",
                "patience": 1,
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
                    "type": "huggingface_adamw",
                    "lr": 0.0001,
                    "weight_decay": 0.001
                },
                "task_nn": {
                    "type": "huggingface_adamw",
                    "lr": 0.0001,
                    "weight_decay": 0.001
                }
            }
        },
        "patience": 4,
        "validation_metric": "+fixed_f1"
    },
    "vocabulary": {
        "type": "from_files",
        "directory": "data/rcv1/bert_vocab"
    },
    "type": "train_test_log_to_wandb",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 16,
            "sorting_keys": [
                "x"
            ]
        },
        "max_instances_in_memory": 1000,
        "num_workers": 5,
        "start_method": "spawn"
    },
    "evaluate_on_test": true
}