{
  // Bibtex data set in 10 stratified folds
  bibtex: {
    num_labels: 159,
    input_features: 1836,
    dir_name: 'bibtex_stratified10folds_meka',
    train_file: 'Bibtex-fold@(1|2|3|4|5|6).arff',
    validation_file: 'Bibtex-fold@(7|8).arff',
    test_file: 'Bibtex-fold@(9|10).arff',
  },
  // bibtex dataset in 10 stratified folds with only 1 fold used for training
  bibtex10: {
    num_labels: 159,
    input_features: 1836,
    dir_name: 'multilabel_classification/bibtex_stratified10folds_meka',
    train_file: 'Bibtex-fold1.arff',
    validation_file: 'Bibtex-fold@(7|8).arff',
    test_file: 'Bibtex-fold@(9|10).arff',
  },
}
