#!/bin/bash
set -x
echo "=======Creating virtual env========="
python3 -m virtualenv  .venv_seal
source .venv_seal/bin/activate

echo "=======Install test requirements======="
pip install -r test_requirements.txt

echo "=======Install doc requirements======="
pip install -r doc_requirements.txt

echo "=======Install core requirements======"
pip install -r core_requirements.txt

echo "Do 'source .venv_seal/bin/activate' to load the enviroment."
