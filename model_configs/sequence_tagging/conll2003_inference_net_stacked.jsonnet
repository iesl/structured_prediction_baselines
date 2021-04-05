local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'conll2003ner';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local transformer_model = 'bert-base-uncased';
local transformer_hidden_dim = 768;
local max_length = 256;
local num_tags = 15;

local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation = 'softplus';
//local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local inference_score_weight = std.parseJson(std.extVar('inference_score_weight'));
local cross_entorpy_loss_weight = std.parseJson(std.extVar('cross_entorpy_loss_weight'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local task_nn = {
  type: 'sequence-tagging',
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer_mismatched',
        model_name: transformer_model,
        max_length: max_length,
      },
    },
  },
  dropout: 0.1,
};

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      tokens: {
        type: 'pretrained_transformer_mismatched',
        model_name: transformer_model,
        max_length: max_length,
      },
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
    type: 'sequence-tagging',
    label_encoding: 'BIOUL',
    sampler: {
      type: 'inference-network',
      optimizer: {
        lr: 0.001,
        weight_decay: 1e-4,
        type: 'adam',
      },
      loss_fn: {
        type: 'combination-loss',
        constituent_losses: [
          {
            type: 'sequence-tagging-inference',
            inference_score_weight: inference_score_weight,
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'sequence-tagging-masked-cross-entropy',
            reduction: 'none',
            normalize_y: true,
          },
        ],
        loss_weights: [1.0, cross_entorpy_loss_weight],
        reduction: 'mean',
      },
      inference_nn: task_nn,
      cost_augmented_layer: {
        type: 'sequence-tagging-stacked',
        seq2seq: {
          type: 'feedforward',
          feedforward: {
            input_dim: 2 * num_labels,
            num_layers: ff_linear_layers,
            activations: [ff_activation, 'linear'],
            hidden_dims: [ff_hidden for i in std.range(0, ff_linear_layers - 2)] + [num_labels],
          },
        },

        normalize_y: true,
      },
      stopping_criteria: {
        type: 'number-of-steps',
        number_of_steps: 10,
      },
    },
    oracle_value_function: { type: 'manhattan' },
    score_nn: {
      type: 'sequence-tagging',
      task_nn: task_nn,
      global_score: {
        type: 'linear-chain',
        num_tags: num_labels,
      },
    },
    loss_fn: {
      type: 'sequence-tagging-margin-based',
      reduction: 'mean',
      perceptron_loss_weight: inference_score_weight,
    },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',  // bucket is only good for tasks that involve seq
      batch_size: 32,
    },
  },
  trainer: {
    num_epochs: if test == '1' then 2 else 300,
    //grad_norm: 10.0,
    patience: 8,
    validation_metric: '+f1-measure-overall',
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
      },
    ] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}
