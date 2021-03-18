// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).
// see: https://github.com/allenai/allennlp-models/blob/main/training_config/tagging/ner.jsonnet

local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'conll2003ner';
local dataset = (import 'datasets.jsonnet')[dataset_name];

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
      token_characters: {
        type: 'characters',
        min_padding_length: 3,
      },
    },
  },
  train_data_path: 'data/' + dataset.dir + '/' + dataset.files.train,
  validation_data_path: 'data/' + dataset.dir + '/' + dataset.files.val,
  test_data_path: 'data/' + dataset.dir + '/' + dataset.files.test,
  model: {
    type: 'crf_tagger',
    label_encoding: 'BIOUL',
    constrain_crf_decoding: true,
    calculate_span_f1: true,
    dropout: 0.5,
    include_start_end_transitions: false,
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 50,
          pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz',
          trainable: true,
        },
        token_characters: {
          type: 'character_encoding',
          embedding: {
            embedding_dim: 16,
          },
          encoder: {
            type: 'cnn',
            embedding_dim: 16,
            num_filters: 128,
            ngram_filter_sizes: [3],
            conv_layer_activation: 'relu',
          },
        },
      },
    },
    encoder: {
      type: 'lstm',
      input_size: 50 + 128,
      hidden_size: 200,
      num_layers: 2,
      dropout: 0.5,
      bidirectional: true,
    },
  },
  data_loader: {
    batch_size: 64,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: 0.001,
    },
    checkpointer: {
      num_serialized_models_to_keep: 3,
    },
    validation_metric: '+f1-measure-overall',
    num_epochs: 75,
    grad_norm: 5.0,
    patience: 25,
    callbacks: ['track_epoch_callback'] + (if use_wandb then ['log_metrics_to_wandb'] else []),
  },
}
