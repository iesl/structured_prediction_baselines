// Setup
local cuda_device = std.parseInt(std.extVar('CUDA_DEVICE'));
local test = std.extVar('TEST');  // a test run with small dataset
local use_wandb = (if test == '1' then false else true);
// Data
local data_dir = '/mnt/nfs/scratch1/username/SEAL/structured_prediction_baselines/data/weizmann_horse_seg';
local train_data_path = data_dir + '/weizmann_horse_train.npy'; // TODO trainval
local validation_data_path = data_dir + '/weizmann_horse_val.npy';
local test_data_path = data_dir + '/weizmann_horse_test.npy';
local batch_size = std.parseInt(std.extVar('batch_size'));
local eval_cropping = std.extVar('eval_cropping'); // uncomment if environment variable
// local eval_cropping = std.parseJson(std.extVar('eval_cropping')); // TODO uncomment if from yaml
// Model
local task_nn = {type: 'weizmann-horse-seg',};
// Training and inference
local mix_method = std.extVar('mix_method'); // uncomment if environment variable
// local mix_method = std.parseJson(std.extVar('mix_method')); // TODO uncomment if from yaml
local gbi_optim = 'sgd';
local gbi_lr = std.parseJson(std.extVar('gbi_lr'));
local adv_lr = std.parseJson(std.extVar('adv_lr'));

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
    type: 'weizmann-horse-seg',

    sampler: {
      type: 'random-picking-container', // TODO: appending-container
      log_key: 'sampler',
      constituent_samplers: [
        //GBI
        {
          type: 'gradient-based-inference',
          log_key: 'gbi',
          gradient_descent_loop: {
            optimizer: {
              type: gbi_optim,
              lr: gbi_lr,
              momentum: 0.0,
              weight_decay: 0.0,
            },
          },
          loss_fn: { type: 'multi-label-dvn-score', log_key: 'neg_dvn_score', reduction: 'none' }, // same for weizmann
          output_space: { type: 'weizmann-horse-seg', default_value: 0.0, mix_method: mix_method }, // TODO
          stopping_criteria: 30,
          sample_picker: { type: 'best' },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },
        // Adversarial
        {
          type: 'gradient-based-inference',
          log_key: 'adv',
          gradient_descent_loop: {
            optimizer: {
              type: gbi_optim,
              lr: adv_lr,
              momentum: 0.0,
              weight_decay: 0.0,
            },
          },
          loss_fn: {
            type: 'negative',
            log_key: 'neg',
            constituent_loss: { type: 'multi-label-dvn-bce', log_key: 'dvn_bce', reduction: 'none' },
            reduction: 'none',
          },  // same for weizmann
          output_space: { type: 'weizmann-horse-seg', default_value: 0.0, mix_method: mix_method },
          stopping_criteria: 30,
          sample_picker: { type: 'best' },
          number_init_samples: 1,
          random_mixing_in_init: 0.5,
        },
        // Ground Truth
        { type: 'ground-truth' },
      ],
    },

    score_nn: {
      type: 'weizmann-horse-seg',
      task_nn: task_nn, // not really useful
    },
    oracle_value_function: { type: 'seg-iou', differentiable: true },
    loss_fn: { type: 'multi-label-dvn-bce', log_key: 'dvn_bce', reduction: 'mean' }, // same for weizmann horse seg

    inference_module: {
      type: 'gradient-based-inference-weizmann-horse-seg-36crops',
      log_key: 'inference',
      gradient_descent_loop: {
        optimizer: {
          type: gbi_optim,
          lr: gbi_lr,
          momentum: 0.0,
          weight_decay: 0.0,
        },
      },
      loss_fn: { type: 'multi-label-dvn-score', log_key: 'neg_dvn_score', reduction: 'none' }, // same for weizmann
      output_space: { type: 'weizmann-horse-seg', default_value: 0.0 },
      stopping_criteria: 30,
      sample_picker: { type: 'best' },
      number_init_samples: 1,
      random_mixing_in_init: 1.0, // not used
    },
  },

  trainer: {
    type: 'gradient_descent_minimax',
    cuda_device: cuda_device,
    inner_mode: 'task_nn',

    num_epochs: if test == '1' then 5 else 1000,
    num_steps: { task_nn: 0, score_nn: 1 },
    patience: 20,
    validation_metric: '+seg_iou',

    // grad_norm: { score_nn: 10.0 }, // TODO: commented out 5/16
    optimizer: {
      optimizers: {
        score_nn: {
          type: 'adam',
          lr: 0.01,
          weight_decay: 0.0001,
        },
      },
    },
    learning_rate_schedulers: {
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
