// Setup
local cuda_device = std.parseInt(std.extVar('CUDA_DEVICE'));
local test = std.extVar('TEST');  // a test run with small dataset
local use_wandb = (if test == '1' then false else true);
// Data
local data_dir = '/mnt/nfs/scratch1/wenlongzhao/SEAL/structured_prediction_baselines/data/weizmann_horse';
local train_data_path = data_dir + '/weizmann_horse_trainval.npy';
local validation_data_path = data_dir + '/weizmann_horse_val.npy';
local test_data_path = data_dir + '/weizmann_horse_test.npy';
local batch_size = std.parseInt(std.extVar('batch_size'));
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
  validation_dataset_reader: {type: 'weizmann-horse-seg',},
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
      type: 'weizmann-horse-seg-inference-net', // TODO
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
            type: 'sequence-tagging-masked-cross-entropy', // TODO placeholder, not use
            log_key: 'ce',
            reduction: 'none',
            normalize_y: false,
          },
        ],
        loss_weights: [score_loss_weight, 1],
        reduction: 'mean',
      },
    },
    oracle_value_function: {type: 'seg-iou', differentiable: true},
    score_nn: {
      type: 'weizmann-horse-seg',
      task_nn: task_nn,
    },
    loss_fn: { TODO
      type: 'seqtag-nce-ranking-with-discrete-sampling',
      reduction: 'mean',
      log_key: 'seq_nce_loss',
      num_samples: 10,
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

    optimizer: {
      optimizers: {
        task_nn: {
            type: 'sgd',
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0001,
          },
        score_nn: {
          type: 'sgd',
          lr: 0.1,
          momentum: 0.9,
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