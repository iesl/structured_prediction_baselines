// Setup
local cuda_device = std.extVar('CUDA_DEVICE');
local test = std.extVar('TEST');  // a test run with small dataset
local use_wandb = (if test == '1' then false else true);
// Data
local data_dir = '/mnt/nfs/scratch1/wenlongzhao/SEAL/structured_prediction_baselines/data/Tweet_POS';
local train_data_path = data_dir + '/oct27.traindev.proc.cnn';
local validation_data_path = data_dir + '/oct27.test.proc.cnn';
local test_data_path = data_dir + '/daily547.proc.cnn';
local vocab_dir = '/mnt/nfs/scratch1/wenlongzhao/SEAL/structured_prediction_baselines/data/Tweet_POS/bilstm_vocab';
// Model
local num_labels = 25;
local attention_dim = std.parseJson(std.extVar('attention_dim'));
local attention_dropout = std.parseJson(std.extVar('attention_dropout_10x'))/10.0;
local cross_entropy_loss_weight = 1;
local score_loss_weight = std.parseJson(std.extVar('score_loss_weight'));
local ff_activation = 'softplus';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local task_nn = {
  type: 'sequence-tagging',
  "text_field_embedder": {
    "token_embedders": {
      "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
          "trainable": true,
      },
      "token_characters": {
        "type": "character_encoding",
        "embedding": {
          "embedding_dim": 16,
        },
        "encoder": {
          "type": "cnn",
          "conv_layer_activation": "relu",
          "embedding_dim": 16,
          "ngram_filter_sizes": [3],
          "num_filters": 128,
        },
      },
    },
  },
  "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "dropout": 0.5,
      "hidden_size": 200,
      "input_size": 50+128,
      "num_layers": 2,
  },
};
// Training
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local tasknn_lr = std.parseJson(std.extVar('tasknn_lr'));

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,

  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  test_data_path: test_data_path,
  // vocabulary: {
  //   type: 'from_files',
  //   directory: vocab_dir,
  // },
  dataset_reader: {
    type: 'tweet_pos',
    // coding_scheme: 'BIOUL',
    token_indexers: {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3,
      },
    },
    model_name: "bilstm",
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket', // bucket is only good for tasks that involve seq
      batch_size: 32, // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['tokens'],
    },
    // num_workers: 5,
    // max_instances_in_memory: if test == '1' then 10 else 1000,
    // start_method: 'spawn',
  },

  model: {
    type: 'seal-seq-tag',
    // label_encoding: 'BIOUL',
    task_nn: task_nn,
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    inference_module: {
      type: 'sequence-tagging-inference-net-normalized', // didn't use
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'sequence-tagging-score-loss', // jy: fix here..
            log_key: 'neg.nce_score',
            normalize_y: true,
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'sequence-tagging-masked-cross-entropy',
            log_key: 'ce',
            reduction: 'none',
            normalize_y: false,
          },
        ],
        loss_weights: [score_loss_weight, cross_entropy_loss_weight],
        reduction: 'mean',
      },
    },
    oracle_value_function: { type: 'manhattan', differentiable: true},
    score_nn: {
      type: 'sequence-tagging',
      task_nn: task_nn,
      global_score: {
        type: 'self-attention-full-sequence',
        num_heads: 1,
        num_tags: num_labels,
        attention_dim: attention_dim,
        dropout: attention_dropout,
      },
    },
    loss_fn: {
      type: 'seqtag-nce-ranking-with-discrete-sampling',
      reduction: 'mean',
      log_key: 'seq_nce_loss',
      num_samples: 10,
    },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },

  trainer: {
    type: 'gradient_descent_minimax',
    cuda_device: std.parseInt(cuda_device),
    inner_mode: 'score_nn',

    num_epochs: if test == '1' then 5 else 300,
    num_steps: { task_nn: 1, score_nn: 1 },
    grad_norm: { task_nn: 1.0 },
    patience: 20,
    validation_metric: '+accuracy',

    optimizer: {
      optimizers: {
        task_nn:
          {
            lr: tasknn_lr,
            weight_decay: weight_decay,
            type: 'adamw',
          },
        score_nn: {
          lr: 0.00001,
          weight_decay: weight_decay,
          type: 'adamw',
        },
      },
    },
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 5,
        verbose: true,
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
          should_log_parameter_statistics: false,
        },
      ]
      else []
    ),
  },
}
