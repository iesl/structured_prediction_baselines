// Setup
local cuda_device = std.extVar('CUDA_DEVICE');
local test = std.extVar('TEST');  // a test run with small dataset
local use_wandb = (if test == '1' then false else true);
// Data
local data_dir = '/mnt/nfs/work1/mccallum/LDCdata/CCGBank/ccgbank_1_1/data/AUTO';
local train_data_path = data_dir + '/ccgbank_train.auto'; // Section 02-21
local validation_data_path = data_dir + '/ccgbank_dev.auto'; // Section 00
local test_data_path = data_dir + '/ccgbank_test.auto'; // Section 23
// Model
local transformer_model = 'bert-base-uncased';
local max_length = 512;
local ff_activation = 'softplus';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local task_nn = {
  type: 'sequence-tagging',
  text_field_embedder: {
    token_embedders: {
      transformer_indexer: { // name of the indexer
        type: 'pretrained_transformer_mismatched',
        model_name: transformer_model,
        max_length: max_length,
      },
    },
  },
};
// Training
local weight_decay = std.parseJson(std.extVar('weight_decay')); // TODO
local tasknn_lr = std.parseJson(std.extVar('tasknn_lr')); // TODO

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,

  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  test_data_path: test_data_path,
  // vocabulary: { // TODO
  //   type: 'from_files',
  //   directory: data_dir + '/' + dataset_metadata.dir_name + '/' + 'bert_vocab',
  // },
  dataset_reader: {
    type: 'ccgbank',
    token_indexers: {
      transformer_indexer: {
        type: 'pretrained_transformer_mismatched',
        model_name: transformer_model,
        max_length: max_length,
      },
    },
    tag_label: 'ccg',
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

  model: { // TODO
    type: 'seal-ccgbank',
    task_nn: task_nn,
    // sampler: {
    //   type: 'appending-container',
    //   log_key: 'sampler',
    //   constituent_samplers: [],
    // },
    inference_module: {
      type: 'sequence-tagging-inference-net-normalized',
      log_key: 'inference_module',
      loss_fn: {
        type: 'sequence-tagging-masked-cross-entropy',
        log_key: 'ce',
        reduction: 'mean',  // mean will work fine because seq-tagging-masked-ce will take care of masking
        normalize_y: false,  // don't normalize because ce requires logits
      },
    },
    // oracle_value_function: {},
    // score_nn: {},
    loss_fn: {
      type: 'zero',
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

    num_epochs: if test == '1' then 5 else 50,
    num_steps: { task_nn: 1, score_nn: 1 },
    grad_norm: { task_nn: 1.0 },
    patience: 5,
    validation_metric: '+accuracy',

    optimizer: {
      optimizers: {
        task_nn:
          {
            lr: tasknn_lr,
            weight_decay: weight_decay,
            type: 'huggingface_adamw',
          },
      },
    },
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 2,
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
        },
      ]
      else []
    ),
  },
}
