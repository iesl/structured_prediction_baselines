{
  // Bibtex data set in 10 stratified folds
  bibtex_strat: {
    num_labels: 159,
    input_features: 1836,
    dir_name: 'bibtex_stratified10folds_meka',
    train_file: 'Bibtex-fold@(1|2|3|4|5|6).arff',
    validation_file: 'Bibtex-fold@(7|8).arff',
    test_file: 'Bibtex-fold@(9|10).arff'
  },
  // bibtex dataset in 10 stratified folds with only 1 fold used for training
  bibtex10: {
    num_labels: 159,
    input_features: 1836,
    dir_name: 'bibtex_stratified10folds_meka',
    train_file: 'Bibtex-fold1.arff',
    validation_file: 'Bibtex-fold@(7|8).arff',
    test_file: 'Bibtex-fold@(9|10).arff'
  },
  // bibtex dataset provided in the dvn paper's repo
  bibtex_original: {
    num_labels: 159,
    input_features: 1836,
    dir_name: 'bibtex_original',
    train_file: 'train.arff',
    validation_file: 'val.arff',
    test_file: 'test.arff'
  },
  // text datasets
  bgc: {
    dir_name: 'bgc',
    num_labels: 146,
    train_file: 'train.jsonl',
    validation_file: 'dev.jsonl',
    test_file: 'test.jsonl'
  },
  nyt: {
    dir_name: 'nyt',
    train_file: 'nyt-fold-@(0|1|2|3|4|5).jsonl',
    validation_file: 'nyt-fold-@(6|7).jsonl',
    test_file: 'nyt-fold-@(8|9).jsonl'
  },
  nyt10: {
    dir_name: 'nyt',
    num_labels: 2109,
    train_file: 'nyt-fold-0.jsonl',
    validation_file: 'nyt-fold-6.jsonl',
    test_file: 'nyt-fold-8.jsonl'
  },
  rcv1: {
    dir_name: 'rcv1',
    num_labels: 103,
    train_file: 'train_val_fold_@(0|1|2|3|4|5).jsonl',
    validation_file: 'train_val_fold_@(6|7|8|9).jsonl',
    test_file: 'test_fold_0.jsonl'
  },
  aapd: {
    dir_name: 'AAPD',
    num_labels: 54,
    train_file: 'aapd_train.jsonl',
    validation_file: 'aapd_validation.jsonl',
    test_file: 'aapd_test.jsonl'
  },

  // feature based datasets
  mediamill: {
    num_labels: 101,
    input_features: 120,
    dir_name: 'mediamill-stratified10folds-meka',
    train_file: 'Mediamill-fold@(1|2|3|4|5|6).arff',
    validation_file: 'Mediamill-fold@(7|8).arff',
    test_file: 'Mediamill-fold@(9|10).arff'
  },
  genbase: {
    num_labels: 27,
    input_features: 1186,
    dir_name: 'genbase-stratified10folds-meka',
    train_file: 'Genbase-fold@(1|2|3|4|5|6).arff',
    validation_file: 'Genbase-fold@(7|8).arff',
    test_file: 'Genbase-fold@(9|10).arff'
  },
  corel5k: {
    num_labels: 374,
    input_features: 499,
    dir_name: 'corel5k-stratified10folds-meka',
    train_file: 'Corel5k-fold@(1|2|3|4|5|6).arff',
    validation_file: 'Corel5k-fold@(7|8).arff',
    test_file: 'Corel5k-fold@(9|10).arff'
  },
  cal500: {
    num_labels: 174,
    input_features: 68,
    dir_name: 'cal500-stratified10folds-meka',
    train_file: 'CAL500-fold@(1|2|3|4|5|6)-normalised.arff',
    validation_file: 'CAL500-fold@(7|8)-normalised.arff',
    test_file: 'CAL500-fold@(9|10)-normalised.arff'
  },
  delicious: {
    num_labels: 983,
    input_features: 500,
    dir_name: 'delicious-stratified10folds-meka',
    train_file: 'Delicious-fold@(1|2|3|4|5|6).arff',
    validation_file: 'Delicious-fold@(7|8).arff',
    test_file: 'Delicious-fold@(9|10).arff'
  },
  eurlexev: {
    num_labels: 3993,
    input_features: 5000,
    dir_name: 'eurlex-ev-stratified10folds-meka',
    train_file: 'Eurlex-ev-fold@(1|2|3|4|5|6)-normalised.arff',
    validation_file: 'Eurlex-ev-fold@(7|8)-normalised.arff',
    test_file: 'Eurlex-ev-fold@(9|10)-normalised.arff'
  },
  tmc500: {
    num_labels: 22,
    input_features: 500,
    dir_name: 'tmc2007-500-Stratified10Folds-Meka',
    train_file: 'tmc2007-500-fold@(1|2|3|4|5|6).arff',
    validation_file: 'tmc2007-500-fold@(7|8).arff',
    test_file: 'tmc2007-500-fold@(9|10).arff'
  },
  expr_fun: {
    num_labels: 500,
    input_features: 561,
    dir_name: 'expr_fun',
    train_file: 'train-normalized.arff',
    validation_file: 'dev-normalized.arff',
    test_file: 'test-normalized.arff'
  },
  expr_go: {
    num_labels: 4132,
    input_features: 561,
    dir_name: 'expr_go',
    train_file: 'train-normalized.arff',
    validation_file: 'dev-normalized.arff',
    test_file: 'test-normalized.arff'
  },
  spo_go: {
    num_labels: 4120,
    input_features: 86,
    dir_name: 'spo_go',
    train_file: 'train-normalized.arff',
    validation_file: 'dev-normalized.arff',
    test_file: 'test-normalized.arff'
  },
  spo_fun: {
    num_labels: 500,
    input_features: 86,
    dir_name: 'spo_fun',
    train_file: 'train-normalized.arff',
    validation_file: 'dev-normalized.arff',
    test_file: 'test-normalized.arff'
  }
}
