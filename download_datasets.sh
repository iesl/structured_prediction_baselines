#!/bin/bash
# CONLL 2003 NER
mkdir -p data/conll2003ner
wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/train.txt -O data/conll2003ner/train.txt
wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/valid.txt -O data/conll2003ner/val.txt
wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/test.txt -O data/conll2003ner/test.txt

