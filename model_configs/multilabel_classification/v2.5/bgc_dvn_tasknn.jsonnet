// Simple tasknn only model for blurb genre collection
local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'bgc';
local dataset_metadata = (import '../datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
// // common
local ff_activation = 'softplus';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
// // score_nn
local transformer_model = 'bert-base-uncased';  // huggingface name of the model
local transformer_dim = 768;
local score_nn_ff_hidden = std.parseJson(std.extVar('score_nn_ff_hidden'));
local score_nn_ff_linear_layers = std.parseJson(std.extVar('score_nn_ff_linear_layers'));
local score_nn_weight_decay = std.parseJson(std.extVar('score_nn_weight_decay'));
local score_nn_ff_hidden = std.parseJson(std.extVar('score_nn_ff_hidden'));
local global_score_hidden_dim = std.parseJson(std.extVar('global_score_hidden_dim'));
// // task_nn
local task_nn_dropout = std.parseJson(std.extVar('task_nn_dropout_10x')) / 10.0;
local task_nn_ff_linear_layers = std.parseJson(std.extVar('task_nn_ff_linear_layers'));
local task_nn_weight_decay = std.parseJson(std.extVar('task_nn_weight_decay'));
local task_nn_transformer_hidden = std.parseJson(std.extVar('task_nn_transformer_hidden'));
local task_nn_embedding = std.floor(task_nn_transformer_hidden / 2);
local task_nn_transformer_layers = std.parseJson(std.extVar('task_nn_transformer_layers'));
local task_nn_transformer_attn_heads = std.parseJson(std.extVar('task_nn_transformer_attn_heads'));
local cross_entropy_loss_weight = std.parseJson(std.extVar('cross_entropy_loss_weight'));
local dvn_score_loss_weight = std.parseJson(std.extVar('dvn_score_loss_weight'));
{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  // Data
  dataset_reader: {
    type: 'bgc',
    [if test == '1' then 'max_instances']: 100,
    token_indexers: {
      x: {
        type: 'pretrained_transformer',
        model_name: transformer_model,
      },
    },
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: transformer_model,
      max_length: 512,
    },

  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),

  // Model
  model: {
    type: 'multi-label-classification-with-infnet',
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    task_nn: {
      type: 'multi-label-text-classification',
      feature_network: {
        text_field_embedder: {
          token_embedders: {
            x: {
              type: 'embedding',
              embedding_dim: task_nn_embedding,
            },
          },
        },
        seq2seq_encoder: {
          type: 'pytorch_transformer',
          input_dim: task_nn_embedding,
          num_layers: task_nn_transformer_layers,
          feedforward_hidden_dim: task_nn_transformer_hidden,
          num_attention_heads: task_nn_transformer_attn_heads,
          positional_encoding: 'sinusoidal',
          positional_embedding_size: 512,  // same as max_len of encoder
          dropout_prob: task_nn_dropout,
        },
        seq2vec_encoder: {
          type: 'cls_pooler',
        },
        final_dropout: 0,
        feedforward: {
          input_dim: task_nn_embedding,
          num_layers: task_nn_ff_linear_layers,
          activations: ([ff_activation for i in std.range(0, task_nn_ff_linear_layers - 2)] + [ff_activation]),
          hidden_dims: ff_hidden,
          dropout: ([task_nn_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
        },
      },
      label_embeddings: {
        embedding_dim: task_nn_embedding,
        vocab_namespace: 'labels',
      },
    },
    inference_module: {
      type: 'multi-label-basic',
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'multi-label-dvn-score',
            log_key: 'neg_dvn_score',
            normalize_y: true,
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'multi-label-bce',
            reduction: 'none',
            log_key: 'bce',
          },
        ],
        loss_weights: [dvn_score_loss_weight, cross_entropy_loss_weight],
        reduction: 'mean',
      },
    },
    oracle_value_function: { type: 'per-instance-f1', differentiable: false },
    score_nn: {
      type: 'multi-label-classification',
      task_nn: {
        type: 'multi-label-text-classification',
        feature_network: {
          text_field_embedder: {
            token_embedders: {
              x: {
                type: 'pretrained_transformer',
                model_name: transformer_model,
              },
            },
          },
          seq2vec_encoder: {
            type: 'bert_pooler',
            pretrained_model: transformer_model,
          },
          final_dropout: score_nn_dropout,
          feedforward: {
            input_dim: transformer_dim,
            num_layers: score_nn_ff_linear_layers,
            activations: ([ff_activation for i in std.range(0, score_nn_ff_linear_layers - 2)] + [ff_activation]),
            hidden_dims: score_nn_ff_hidden,
            dropout: ([score_nn_dropout for i in std.range(0, score_nn_ff_linear_layers - 2)] + [0]),
          },
        },
        label_embeddings: {
          embedding_dim: score_nn_ff_hidden,
          vocab_namespace: 'labels',
        },
      },
      global_score: {
        type: 'multi-label-feedforward',
        feedforward: {
          input_dim: num_labels,
          num_layers: 1,
          activations: ff_activation,
          hidden_dims: global_score_hidden_dim,
        },
      },
    },
    loss_fn: { type: 'multi-label-dvn-bce', log_key: 'dvn_bce' },
    initializer: {
      regexes: [
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*feedforward._linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 2,
    },
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 300,
    grad_norm: { task_nn: 10.0, score_nn: 1.0 },
    num_gradient_accumulation_steps: 16,  // effective batch size = batch_size*num_gradient_accumulation_steps
    patience: 5,
    validation_metric: '+fixed_f1',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 2,
        verbose: true,
      },
    },
    optimizer: {
      optimizers: {  // have only tasknn optmizer
        task_nn: {
          lr: 1e-3,
          weight_decay: task_nn_weight_decay,
          type: 'adamw',
        },
        score_nn: {
          lr: 5e-5,
          weight_decay: score_nn_weight_decay,
          type: 'huggingface_adamw',
        },
      },
    },
    checkpointer: {
      keep_most_recent_by_count: 1,
    },
    callbacks: [
      'track_epoch_callback',
      'slurm',
    ] + (
      if use_wandb then [
        {
          type: 'wandb_allennlp',
          sub_callbacks: [{ type: 'log_best_validation_metrics', priority: 100 }],
          save_model_archive: false,
        },
      ]
      else []
    ),
    inner_mode: 'score_nn',
    num_steps: { task_nn: 1, score_nn: 6 },
  },
}
