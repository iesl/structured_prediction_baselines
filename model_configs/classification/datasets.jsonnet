{
  // text datasets
  trec: {
    dir_name: 'trec',
    num_labels: {
      coarse: 6,
      fine: 50,
    },
    train_file: 'train_val_fold_@(0|1|2|3|4|5).jsonl',
    validation_file: 'train_val_fold_@(6|7|8|9).jsonl',
    test_file: 'test.jsonl',
  },
}