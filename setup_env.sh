#!/bin/bash
echo "=======Creating virtual env========="
python3 -m virtualenv  .venv_allennlp-models-structured-prediction
source .venv_allennlp-models-structured-prediction/bin/activate

echo "=======Install test requirements======="
pip install -r test_requirements.txt

echo "=======Install doc requirements======="
pip install -r doc_requirements.txt

echo "=======Install core requirements======"
pip install -r core_requirements.txt

echo "=======Login to wandb (optional)==============="
wandb init

