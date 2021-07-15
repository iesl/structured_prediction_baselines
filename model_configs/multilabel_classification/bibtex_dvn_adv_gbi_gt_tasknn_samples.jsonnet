local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'bibtex_original';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation = 'tanh';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
//local global_score_hidden_dim = 150;
local global_score_hidden_dim = std.parseJson(std.extVar('global_score_hidden_dim'));
local inf_lr = std.parseJson(std.extVar('inf_lr'));
//local inf_optim = std.parseJson(std.extVar('inf_optim'));
local inf_optim = 'sgd';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
//local sample_picker = std.parseJson(std.extVar('sample_picker'));
local sample_picker = 'best';
local cross_entorpy_loss_weight = std.parseJson(std.extVar('cross_entorpy_loss_weight'));

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
      type: 'appending-container',
      constituent_samplers: [
        //GBI
        {
          type: 'gradient-based-inference',
          gradient_descent_loop: {
            optimizer: {
              lr: inf_lr,  //0.1
              weight_decay: 0,
              type: inf_optim,
            },
          },
          loss_fn: { type: 'multi-label-dvn-score', reduction: 'none' },  //This loss can be different from the main loss // change this
          output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
          stopping_criteria: 20,
          sample_picker: { type: sample_picker },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },
        // Adversarial
        {
          type: 'gradient-based-inference',
          gradient_descent_loop: {
            optimizer: {
              lr: inf_lr,  //0.1
              weight_decay: 0,
              type: inf_optim,
            },
          },
          loss_fn: {
            type: 'negative',
            constituent_loss: { type: 'multi-label-dvn-bce', reduction: 'none' },
            reduction: 'none',
          },
          output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
          stopping_criteria: 20,
          sample_picker: { type: sample_picker },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },
        // Ground Truth
        { type: 'ground-truth' },
        // Inference Net/ TaskNN
        {
          type: 'multi-label-inference-net-normalized-or-sampled',
          num_samples: 10,
          keep_probs: true,
          optimizer: {
            lr: 0.0001,
            weight_decay: 0,
            type: 'adamw',
          },
          inference_nn: {
            type: 'multi-label-classification',
            feature_network: {
              input_dim: num_input_features,
              num_layers: ff_linear_layers,
              activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
              hidden_dims: ff_hidden,
              dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
            },
            label_embeddings: {
              embedding_dim: ff_hidden,
              vocab_namespace: 'labels',
            },
          },
          loss_fn: {
            type: 'combination-loss',
            constituent_losses: [
              {
                type: 'multi-label-dvn-score',
                normalize_y: true,
                reduction: 'none',
              },  //This loss can be different from the main loss // change this
              {
                type: 'multi-label-bce',
                reduction: 'none',
              },
            ],
            loss_weights: [1.0, cross_entorpy_loss_weight],
            reduction: 'sum',
          },
          stopping_criteria: 2,
        },
      ],
    },
    inference_module: {
      type: 'from-container',
      index: -1,
    },
    oracle_value_function: { type: 'per-instance-f1', differentiable: false },
    score_nn: {
      type: 'multi-label-classification',
      task_nn: {
        type: 'multi-label-classification',
        feature_network: {
          input_dim: num_input_features,
          num_layers: ff_linear_layers,
          activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
          hidden_dims: ff_hidden,
          dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
        },
        label_embeddings: {
          embedding_dim: ff_hidden,
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
    loss_fn: { type: 'multi-label-dvn-bce' },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*_linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    shuffle: true,
    batch_size: 32,
  },
  trainer: {
    num_epochs: if test == '1' then 10 else 300,
    //grad_norm: 10.0,
    patience: 20,
    validation_metric: '+fixed_f1',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 5,
      verbose: true,
    },
    optimizer: {
      lr: 0.001,
      weight_decay: 1e-4,
      type: 'adamw',
    },
    checkpointer: {
      keep_most_recent_by_count: 1,
    },
    callbacks: [
      'track_epoch_callback',
      {
        type: 'tensorboard-custom',
        tensorboard_writer: {
          should_log_learning_rate: true,
        },
        model_outputs_to_log: ['sample_probabilities'],
      },
    ] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}
