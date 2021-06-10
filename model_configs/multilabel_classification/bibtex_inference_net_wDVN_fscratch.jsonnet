local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'bibtex_original';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables 
// score_loss_weight (small to large) & learning rate & cross_entorpy_loss_weight=1
// DVN model variables (hyperparameter sweep on: ff_linear_layers (2,3,4,5), inference_score_weight, cross_entorpy_loss_weight)
local ff_hidden = 150; // std.parseJson(std.extVar('ff_hidden')); // pre-trained 150
local label_space_dim = ff_hidden;
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_dropout_score = 0.3;                // std.parseJson(std.extVar('ff_dropout'));
local ff_activation = 'tanh';                   // std.parseJson(std.extVar('ff_activation'));
local ff_linear_layers_score = 3;                // std.parseJson(std.extVar('ff_linear_layers'));
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
// local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));    --> not used below (also not used in InfNN)
local global_score_hidden_dim = 200;            // local global_score_hidden_dim = std.parseJson(std.extVar('global_score_hidden_dim'));
// local inf_lr = std.parseJson(std.extVar('inf_lr'));                      --> used in orig DVN sampler but not used here.
// local inf_optim = std.parseJson(std.extVar('inf_optim'));                --> used in orig DVN sampler but not used here.
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local score_loss_weight = std.parseJson(std.extVar('score_loss_weight'));
local cross_entorpy_loss_weight = 1;

// ToDo:
// 1. turn off the optimizer score_NN.
// 2. constituent sampler problem --> We cannot have one sampler in test time. 


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
      type: 'inference-network',
      optimizer: {
        lr: 0.001,
        weight_decay: 1e-4,
        type: 'adam',
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
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'multi-label-bce',
            reduction: 'none',
          },
        ],
        loss_weights: [score_loss_weight, cross_entorpy_loss_weight],
        reduction: 'mean',
      },
      stopping_criteria: 10,
    },
    oracle_value_function: { type: 'per-instance-f1' },
    score_nn: {
      type: 'multi-label-classification',
      task_nn: {
        type: 'multi-label-classification',
        feature_network: {
          input_dim: num_input_features,
          num_layers: ff_linear_layers_score,
          activations: ([ff_activation for i in std.range(0, ff_linear_layers_score - 2)] + [ff_activation]),
          hidden_dims: ff_hidden,
          dropout: ([ff_dropout_score for i in std.range(0, ff_linear_layers_score - 2)] + [0]),
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
    loss_fn: { // for maximzing score (in SPEN, min step of energy)
      type: 'multi-label-dvn-bce',
      reduction: 'mean',
    },
    initializer: {
      regexes: [
        // [@'score_nn.*', { type: 'pretrained', weights_file_path: 'wandb/DVN/4mepc65o/dvn.th' }],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    shuffle: true,
    batch_size: 32,
  },
  trainer: {
    num_epochs: if test == '1' then 150 else 300,
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
      type: 'adam',
    },
    checkpointer: {
      num_serialized_models_to_keep: 1,
    },
    callbacks: [
      'track_epoch_callback',
      {
        type: 'tensorboard-custom',
        tensorboard_writer: {
          should_log_learning_rate: true,
        },
        model_outputs_to_log: ['y_hat_extra'],
      },
    ] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}
