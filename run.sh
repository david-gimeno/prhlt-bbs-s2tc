#!/bin/bash

set -e

# -- building tokenizers
python src/tokenizers/spm/train_spm_model.py \
  --split-path ./splits/bbs-s2tc/all_clean_data.csv \
  --dst-spm-dir ./src/tokenizers/spm/bbs-s2tc/256vocab/ \
  --spm-name all_clean_data+langs \
  --vocab-size 256

# -- training + inference
python asr_main.py \
  --training-dataset ./splits/bbs-s2tc/all_clean_data.csv \
  --validation-dataset ./splits/bbs-s2tc/dev.csv \
  --test-dataset ./splits/bbs-s2tc/test.csv \
  --config-file ./configs/asr/conformer+hierlidutt_maskctc.yaml \
  --mode both \
  --filter-by-language all-langs \
  --output-dir ./exps/bbs-s2tc/ \
  --output-name test_all-langs \
  --yaml-overrides training_settins:batch_size:16
