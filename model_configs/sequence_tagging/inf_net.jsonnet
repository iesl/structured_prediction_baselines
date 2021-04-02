local dataset_name = 'conll2003ner';
local dataset = (import 'datasets.jsonnet')[dataset_name];

local transformer_model = "bert-base-uncased";
local transformer_hidden_dim = 768;
local max_length = 256;
local num_tags = 15;

{
  evaluate_on_test: true,
  dataset_reader: {
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      tokens: {
        type: "pretrained_transformer_mismatched",
        model_name: transformer_model,
        max_length: max_length,
      },
    },
  },
  train_data_path: 'data/' + dataset.dir + '/' + dataset.files.train,
  validation_data_path: 'data/' + dataset.dir + '/' + dataset.files.val,
  test_data_path: 'data/' + dataset.dir + '/' + dataset.files.test,

  // Model
  model: {
    type: 'sequence-tagging',
    num_tags: num_tags,
    label_encoding: 'BIOUL',
    sampler: {
      type: 'sequence-tagging-inference',
      optimizer: {
        lr: 0.001,
        weight_decay: 1e-4,
        type: 'adam',
      },
      loss_fn: {
        type: 'sequence-tagging-loss',
        reduction: 'mean',
      },
      inference_nn: {
        type: 'seq-tagging-task',
        text_field_embedder: {
          token_embedders: {
            tokens: {
                type: "pretrained_transformer_mismatched",
                model_name: transformer_model,
                max_length: max_length
            },
          },
        },
        num_tags: num_tags,
        dropout: 0.2,
        output_dim: transformer_hidden_dim,
      },
      stopping_criteria: {
        type: 'number-of-steps',
        number_of_steps: 10,
      },
    },
    oracle_value_function: { type: 'manhattan' },
    score_nn: {
      type: 'seq-tagging-score',
      task_nn: {
        type: 'seq-tagging-task',
        text_field_embedder: {
          token_embedders: {
            tokens: {
              type: "pretrained_transformer_mismatched",
              model_name: transformer_model,
              max_length: max_length
            },
          },
        },
        num_tags: num_tags,
        dropout: 0.2,
        output_dim: transformer_hidden_dim,
      },
      global_score: {
        type: 'linear-chain',
        num_tags: num_tags,
      },
    },
    loss_fn: {
      type: 'sequence-tagging-loss',
      reduction: 'mean',
      cross_entropy: false,
      zero_truncation: true,
    },
  },
  data_loader: {
    shuffle: true,
    batch_size: 32,
  },
  trainer: {
    num_epochs: 2,
    //grad_norm: 10.0,
    patience: 20,
    validation_metric: '+f1-measure-overall',
    cuda_device: -1,
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
        model_outputs_to_log: ['y_probabilities'],
      },
    ],
  },
}
