#!/bin/bash
PROBLEM=languagemodel_ptb10k
TMP_DIR=/datasets/BigLearning/jinlianw/tmp
DATA_DIR=/datasets/BigLearning/jinlianw/languagemodel_ptb10k

MODEL=attention_lm_moe
HPARAMS=attention_lm_moe_tiny

TRAIN_DIR=/proj/BigLearning/jinlianw/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

nvprof --export-profile ../profile.nvvp \
       -f --print-summary \
       ./tensor2tensor/bin/t2t-trainer \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --train_steps=1000
