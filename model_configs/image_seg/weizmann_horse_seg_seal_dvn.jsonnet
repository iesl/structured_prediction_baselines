// Setup
local cuda_device = std.parseInt(std.extVar('CUDA_DEVICE'));
local test = std.extVar('TEST');  // a test run with small dataset
local use_wandb = (if test == '1' then false else true);
// Data
local data_dir = '/mnt/nfs/scratch1/username/SEAL/structured_prediction_baselines/data/weizmann_horse_seg';
local train_data_path = data_dir + '/weizmann_horse_train.npy';
local validation_data_path = data_dir + '/weizmann_horse_val.npy';
local test_data_path = data_dir + '/weizmann_horse_test.npy';
local batch_size = std.parseInt(std.extVar('batch_size'));
// local eval_cropping = std.extVar('eval_cropping'); // uncomment if environment variable
local eval_cropping = std.parseJson(std.extVar('eval_cropping')); // TODO uncomment if from yaml
// Model
local task_nn = {type: 'weizmann-horse-seg',};
// Training
local score_loss_weight = std.parseJson(std.extVar('score_loss_weight'));

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,

  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  test_data_path: test_data_path,
  vocabulary: {type: 'empty',},
  dataset_reader: {type: 'weizmann-horse-seg', cropping: 'random',},
  validation_dataset_reader: {type: 'weizmann-horse-seg', cropping: eval_cropping,},
  data_loader: {
    batch_size: batch_size,
    max_instances_in_memory: batch_size, // so that cropping happens every time
    shuffle: true,
  },

  model: {
    type: 'seal-weizmann-horse-seg',
    task_nn: task_nn,
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    inference_module: {
      type: 'weizmann-horse-seg-inference-net',
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'weizmann-horse-seg-dvn-score-loss',
            normalize_y: true,
            reduction: 'none',
            log_key: 'neg.dvn_score',
          },
          {
            type: 'weizmann-horse-seg-ce',
            normalize_y: false,
            reduction: 'none',
            log_key: 'ce',
          },
        ],
        loss_weights: [score_loss_weight, 1],
        reduction: 'mean',
      },
      eval_loss_fn: (if eval_cropping == 'thirty_six' then {type: 'zero'} else null), // zero loss during eval
    },
    oracle_value_function: {type: 'seg-iou', differentiable: true},
    score_nn: {
      type: 'weizmann-horse-seg',
      task_nn: task_nn,
    },
    loss_fn: {
      type: 'weizmann-horse-seg-dvn-bce',
      log_key: 'dvn_bce',
      normalize_y: true,
      reduction: 'mean',
    },
  },

  trainer: {
    type: 'gradient_descent_minimax',
    cuda_device: cuda_device,
    inner_mode: 'score_nn',

    num_epochs: if test == '1' then 5 else 1000,
    num_steps: { task_nn: 1, score_nn: 1 },
    patience: 20,
    validation_metric: '+seg_iou',

    grad_norm: { task_nn: 10.0, score_nn: 10.0 },
    optimizer: {
      optimizers: {
        task_nn: {
            type: 'adam',
            lr: 0.1,
            weight_decay: 0.0001,
          },
        score_nn: {
          type: 'adam',
          lr: 0.1,
          weight_decay: 0.0001,
        },
      },
    },
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 10,
        verbose: true,
      },
      score_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 10,
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
