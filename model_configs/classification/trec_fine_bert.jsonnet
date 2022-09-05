// Setup
local cuda_device = std.parseInt(std.extVar('CUDA_DEVICE'));
local test = std.extVar('TEST');  // a test run with small dataset
local use_wandb = (if test == '1' then false else true);
// Data
local data_dir = std.extVar('DATA_DIR');
local dataset_name = 'trec';
local label_granularity = 'fine';
local dataset_metadata = (import './datasets.jsonnet')[dataset_name];
// Model
local transformer_model = 'bert-base-uncased';
local transformer_dim = 768;
local n_ff_linear_layers = std.parseJson(std.extVar('n_ff_linear_layers'));
local ff_hidden_dim = std.parseJson(std.extVar('ff_hidden_dim'));
local ff_activation = 'softplus';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
// Training
local n_epochs = std.parseInt(std.extVar('n_epochs'));
local batch_size = std.parseInt(std.extVar('batch_size'));
local task_nn_dropout = std.parseJson(std.extVar('task_nn_dropout'));
local task_nn_lr = std.parseJson(std.extVar('task_nn_lr'));
local task_nn_weight_decay = std.parseJson(std.extVar('task_nn_weight_decay'));

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,

  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' + dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' + dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' + dataset_metadata.test_file),
  vocabulary: {
    type: 'from_files',
    directory: data_dir + '/' + dataset_metadata.dir_name + '/' + 'bert_vocab'
  },
  dataset_reader: {
    type: dataset_name,
    granularity: label_granularity,
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: transformer_model,
      max_length: 512,
    },
    token_indexers: {
      x: { // token_indexer name
        type: 'pretrained_transformer',
        model_name: transformer_model,
      },
    },
  },
  data_loader: {
    "batch_sampler": {
      "type": "bucket", // good for tasks that involve sequences
      "batch_size": batch_size, // effective batch size = batch_size * num_gradient_accumulation_steps
    },
  },

  model: {
    type: 'classification-with-infnet',
    task_nn: {
      type: 'text-classification',
      feature_network: {
        text_field_embedder: {
          token_embedders: {
            x: {  // token_indexer name
              type: 'pretrained_transformer',
              model_name: transformer_model,
            },
          },
        },
        seq2vec_encoder: {
          type: 'bert_pooler',
          pretrained_model: transformer_model,
        },
        final_dropout: task_nn_dropout,
        feedforward: {
          input_dim: transformer_dim,
          num_layers: n_ff_linear_layers,
          activations: ([ff_activation for i in std.range(0, n_ff_linear_layers - 2)] + [ff_activation]),
          hidden_dims: ff_hidden_dim,
          dropout: ([task_nn_dropout for i in std.range(0, n_ff_linear_layers - 2)] + [0]),
        },
      },
      label_embeddings: {
        embedding_dim: ff_hidden_dim,
        vocab_namespace: 'labels',
      },
    },
    score_nn: null,
    oracle_value_function: null,
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    inference_module: {
      type: 'classification-basic',
      log_key: 'inference_module',
      loss_fn: {
        type: 'classification-ce',
        reduction: 'mean',
        log_key: 'cross_entropy',
      },
    },
    loss_fn: {type: 'zero'},
    initializer: {
      regexes: [
        [@'.*feedforward._linear_layers.*weight', (
          if std.member(['tanh', 'sigmoid'], ff_activation) then {
            type: 'xavier_uniform', gain: gain
          } else {
            type: 'kaiming_uniform', nonlinearity: 'relu'
          }
        )],
        [@'.*feedforward._linear_layers.*bias', {type: 'zero'}],
      ],
    },
  },

  trainer: {
    type: 'gradient_descent_minimax',
    cuda_device: cuda_device,
    inner_mode: 'score_nn',

    num_epochs: if test == '1' then 2 else n_epochs,
    num_gradient_accumulation_steps: 1,
    num_steps: {task_nn: 1, score_nn: 0},
    validation_metric: '+accuracy',
    patience: 5,

    grad_norm: {task_nn: 1.0},
    optimizer: {
      optimizers: {
        task_nn: {
          type: 'huggingface_adamw',
          lr: task_nn_lr,
          weight_decay: task_nn_weight_decay,
        },
      },
    },
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 3,
        verbose: true,
      },
    },

    checkpointer: {keep_most_recent_by_count: 1},
    callbacks: [
      'track_epoch_callback',
      'slurm',
    ] + (
      if use_wandb then [
        {
          type: 'wandb_allennlp',
          sub_callbacks: [{type: 'log_best_validation_metrics', priority: 100}],
          save_model_archive: false,
          watch_model: false,
          should_log_parameter_statistics: false,
        },
      ] else []
    ),
  },
}