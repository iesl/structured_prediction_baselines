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
local transformer_hidden = std.parseJson(std.extVar('transformer_hidden'));
local embedding_dim = std.parseJson(std.extVar('embedding_dim'));
local label_space_dim = embedding_dim;
local dropout = std.parseJson(std.extVar('dropout_10x')) / 10.0;
local transformer_layers = std.parseJson(std.extVar('transformer_layers'));
local transformer_attn_heads = std.parseJson(std.extVar('transformer_attn_heads'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local activation = 'softplus';
local gain = (if activation == 'tanh' then 5 / 3 else 1);

local create_feature_network(
  embedding_dim=300,
  transformer_layers=3,
  transformer_hidden=1024,
  transformer_attn_heads=3,
  dropout=0.1,
  use_glove=true,
      ) = {
  text_field_embedder: {
    token_embedders: {
      x: if (embedding_dim == 300 && use_glove) then {
        type: 'embedding',
        embedding_dim: embedding_dim,
        pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz',
        trainable: true,
      } else {
        type: 'embedding',
        embedding_dim: embedding_dim,
      },
    },
  },
  seq2seq_encoder: {
    type: 'pytorch_transformer',
    input_dim: embedding_dim,
    num_layers: transformer_layers,
    feedforward_hidden_dim: transformer_hidden,
    num_attention_heads: transformer_attn_heads,
    positional_encoding: 'sinusoidal',
    positional_embedding_size: 512,  // same as max_len of encoder
    dropout_prob: dropout,
  },
  seq2vec_encoder: {
    type: 'cls_pooler',
    embedding_dim: embedding_dim,
  },
  final_dropout: 0,  // don't need a dropout here because transformer output ends with a dropout
  // we will drop the extra feed-forward because each layer of transformer ends with
  // a two layer feedforward.
};


{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  // Data
  dataset_reader: {
    type: 'bgc',
    [if test == '1' then 'max_instances']: 100,
    token_indexers: {
      x: {
        type: 'single_id',
        lowercase_tokens: true,
      },
    },
    tokenizer: {
      type: 'spacy',
    },
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),
  vocabulary: { type: 'from_files', directory: data_dir + '/' + dataset_metadata.dir_name + '/vocabulary' },
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
      feature_network: create_feature_network(embedding_dim, transformer_layers, transformer_hidden, transformer_attn_heads, dropout),
      label_embeddings: {
        embedding_dim: embedding_dim,
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
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*feedforward._linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 2,  // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['x'],
    },
    num_workers: 5,
    max_instances_in_memory: if test == '1' then 10 else 1000,
    start_method: 'spawn',
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 1 else 300,
    grad_norm: { task_nn: 5.0 },
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
        task_nn:
          {
            lr: 1e-5,
            weight_decay: weight_decay,
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
    num_steps: { task_nn: 1, score_nn: 1 },
  },
}
