local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'iwslt';
local source_lang = 'en';
local target_lang = 'de';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name][source_lang + '-' + target_lang];
//local num_labels = dataset_metadata.num_labels;
local transformer_model = 'bert-base-multilingual-cased';
local transformer_hidden_dim = 768;
//local max_length = 512;

local ff_activation = 'softplus';
local cross_entropy_loss_weight = 1;
local ff_weight_decay = std.parseJson(std.extVar('weight_decay'));
local num_attn_layers = std.parseJson(std.extVar('attn_layers'));
local num_attn_heads = std.parseJson(std.extVar('attn_heads'));
local attn_dropout = std.parseJson(std.extVar('attn_dropout'));
local ff_hidden_dim = std.parseJson(std.extVar('ff_hidden_dim'));
local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local task_nn = {
  type: 'machine-translation',
  source_embedder: {
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer',  // we don't use mismatched because that is what allennlp srl model does
        model_name: transformer_model,
        //max_length: max_length,
      },
    },
  },
  decoder: {
    type: 'auto_regressive_decoder',
    decoder_net: {
      decoding_dim: transformer_hidden_dim,
      target_embedding_dim: transformer_hidden_dim,
      feedforward_hidden_dim: ff_hidden_dim,
      num_layers: num_attn_layers,
      num_attention_heads: num_attn_heads,
      dropout_prob: ff_dropout,
      residual_dropout_prob: ff_dropout,
      attention_dropout_prob: attn_dropout,
    },
    target_embedder: {
      token_embedders: {
        tokens: {
          type: 'pretrained_transformer',  // we don't use mismatched because that is what allennlp srl model does
          model_name: transformer_model,
          //max_length: max_length,
        },
      },
    },
    beam_search: {
      max_steps: 150,
      beam_size: 5,
      min_steps: 10,
      final_sequence_scorer: {type: 'length-normalized-sequence-log-prob'}
    }
  }
};

local token_indexer = {
  tokens: {
    type: 'pretrained_transformer',
    model_name: transformer_model,
  },
};

local tokenizer = {
  type: 'pretrained_transformer',
  model_name: transformer_model,
  max_length: 150,
};

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'seq2seq',
    source_tokenizer: tokenizer,
    source_token_indexers: token_indexer,
    source_add_start_token: false,
    source_add_end_token: false,
    target_add_start_token: false,
    target_add_end_token: false,
    [if test == '1' then 'max_instances']: 5,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),
  vocabulary: {
    type: 'from_files',
    directory: data_dir + '/' + dataset_metadata.vocab_dir + '/' + 'multilingual-bert-vocab',
    padding_token: '[PAD]',
    oov_token: '[UNK]'
  },
  // Model
  model: {
    type: 'seal-machine-translation',
    task_nn: task_nn,
    inference_module: {
      type: 'machine-translation-inference-net-normalized',
      log_key: 'inference_module',
      loss_fn: {
        type: 'machine-translation-masked-cross-entropy',
        log_key: 'ce',
        reduction: 'mean',  // mean will work fine because seq-tagging-masked-ce will take care of masking
        normalize_y: false,  // don't normalize because ce requires logits
      },
    },
    loss_fn: {
      type: 'zero',  // there is no score_nn so we put a dummy zero loss
    },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 16,  // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['source_tokens'],
    },
    //max_instances_in_memory: if test == '1' then 10 else 1000,
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 5 else 50,
    grad_norm: { task_nn: 1.0 },
    patience: 4,
    validation_metric: '+BLEU',
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
      optimizers: {
        task_nn:
          {
            lr: 0.00001,
            weight_decay: ff_weight_decay,
            type: 'huggingface_adamw',
          },
      },
      parameter_groups: {
        task_nn: [
            [["task_nn.decoder.decoder_net."], {"lr": 1e-3}]
        ],
      }
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
          should_log_parameter_statistics: false,
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
