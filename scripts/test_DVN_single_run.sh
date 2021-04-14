mkdir -p test_dir
allennlp train model_configs/multilabel_classification/DVN_config.json -s test_dir  --include-package structured_prediction_baselines >& test_dir/out.log
