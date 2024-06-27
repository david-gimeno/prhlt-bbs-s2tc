#!/bin/bash

set -e

# -- building tokenizers
python src/tokenizers/spm/train_spm_model.py \
  --split-path ./splits/bbs-s2tc/all_clean_data.csv \
  --dst-spm-dir ./src/tokenizers/spm/bbs-s2tc/256vocab/ \
  --spm-name all_clean_data+langs \
  --vocab-size 256

# -- training model

