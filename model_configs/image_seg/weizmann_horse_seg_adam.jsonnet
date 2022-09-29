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
  validation_data_loader: {
    batch_size: batch_size,
    max_instances_in_memory: batch_size,
  },

  model: {
    type: 'seal-weizmann-horse-seg',
    task_nn: task_nn,
    inference_module: {
      type: 'weizmann-horse-seg-inference-net',
      log_key: 'inference_module',
      loss_fn: {
        type: 'weizmann-horse-seg-ce',
        reduction: 'mean',
        normalize_y: false,
        log_key: 'ce',
      },
      eval_loss_fn: (if eval_cropping == 'thirty_six' then {type: 'zero'} else null), // zero loss during eval
    },
    loss_fn: {type: 'zero'},
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
