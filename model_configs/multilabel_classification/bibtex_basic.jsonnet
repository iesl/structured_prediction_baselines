local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'bibtex_original';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
//local label_space_dim= std.parseJson(std.extVar('label_space_dim'));
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_activation = std.parseJson(std.extVar('ff_activation'));
//local ff_activation = 'tanh';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));

local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  // Data
  dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  validation_dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),

  // Model
  model: {
    type: 'multi-label-classification',
    sampler: {
      type: 'multi-label-basic',
      task_nn: {
        feature_network: {
          input_dim: num_input_features,
          num_layers: ff_linear_layers,
          activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + ['linear']),
          hidden_dims: ff_hidden,
          dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
        },
        label_embeddings: {
          embedding_dim: ff_hidden,
          vocab_namespace: 'labels',
        },
      },
    },
    loss_fn: { type: 'multi-label-bce', reduction: 'mean' },
    initializer: {
      regexes: [
        [@'.*feature_network._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    shuffle: true,
    batch_size: 64,
  },
  trainer: {
    num_epochs: if test == '1' then 5 else 200,
    //grad_norm: 10.0,
    patience: 10,
    validation_metric: '+fixed_f1',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 3,
      verbose: true,
    },
    optimizer: {
      parameter_groups: [[[@'.*linear_layers.*weight', @'.*linear_layers.*bias'], { weight_decay: ff_weight_decay }]],
      lr: 0.001,
      weight_decay: 0,
      type: 'adamw',
    },
    checkpointer: {
      num_serialized_models_to_keep: 1,
    },
    callbacks: [
      'track_epoch_callback',
      {
        type: 'tensorboard',
        tensorboard_writer: {
          should_log_learning_rate: true,
        },
      },
    ] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}
