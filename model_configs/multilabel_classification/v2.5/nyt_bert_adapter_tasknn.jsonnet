// Simple tasknn only model for blurb genre collection
local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'nyt10';
local dataset_metadata = (import '../datasets.jsonnet')[dataset_name];
local num_input_features = dataset_metadata.input_features;

// model variables
local task_nn_dropout = std.parseJson(std.extVar('dropout_10x')) / 10.0;
local ff_activation = 'softplus';
local ff_linear_layers = 2;
local task_nn_weight_decay = std.parseJson(std.extVar('weight_decay'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local transformer_model = 'bert-base-uncased';  // huggingface name of the model
local transformer_dim = 768;
{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  // Data
  dataset_reader: {
    type: 'nyt',
    //[if test == '1' then 'max_instances']: 100,
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
  vocabulary: {
        type: 'from_files',
        directory: data_dir + '/' + dataset_metadata.dir_name + '/' + 'bert_vocab'
  },

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
              type: 'pretrained_transformer_with_adapter',
              model_name: transformer_model,
            },
          },
        },
        seq2vec_encoder: {
          type: 'bert_pooler',
          pretrained_model: transformer_model,
        },
        final_dropout: 0,
        feedforward: {
          input_dim: transformer_dim,
          num_layers: ff_linear_layers,
          activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
          hidden_dims: ([transformer_dim * 2 for i in std.range(0, ff_linear_layers - 2)] + [transformer_dim]),
          dropout: ([task_nn_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
        },
      },
      label_embeddings: {
        embedding_dim: transformer_dim,
        vocab_namespace: 'labels',
      },
    },
    inference_module: {
      type: 'multi-label-basic',
      log_key: 'inference_module',
      loss_fn: {
        type: 'multi-label-bce',
        reduction: 'mean',
        log_key: 'bce',
      },
    },
    oracle_value_function: { type: 'per-instance-f1', differentiable: false },
    score_nn: null,  // no score nn for basic tasknn model
    loss_fn: { type: 'zero' },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*feedforward._linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 16,  // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['x'],
    },
    num_workers: 5,
    max_instances_in_memory: if test == '1' then 10 else 1000,
    start_method: 'spawn',
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 300,
    grad_norm: { task_nn: 1.0 },
    //num_gradient_accumulation_steps: 16,  // effective batch size = batch_size*num_gradient_accumulation_steps
    patience: 4,
    validation_metric: '+fixed_f1',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 1,
        verbose: true,
      },
    },
    optimizer: {
      optimizers: {  // have only tasknn optmizer
        task_nn:
          {
            lr: 1e-5,
            weight_decay: task_nn_weight_decay,
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
          watch_model: false,
        },
      ]
      else []
    ),
    inner_mode: 'score_nn',
    num_steps: { task_nn: 1, score_nn: 1 },
  },
}
