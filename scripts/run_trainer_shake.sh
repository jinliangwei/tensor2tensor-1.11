#!/bin/bash
PROBLEM=image_mnist
TMP_DIR=/datasets/BigLearning/jinlianw/tmp
DATA_DIR=/datasets/BigLearning/jinlianw/$PROBLEM

MODEL=shake_shake
HPARAMS=shake_shake_quick
TRAIN_STEPS=1000
EVAL_STEPS=100

TRAIN_DIR=/proj/BigLearning/jinlianw/t2t_train_single/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

DBG_PROFILE=false

USE_NVPROF=false

if [ $USE_NVPROF == "true" ]
then
    /usr/local/cuda/bin/nvprof \
	--export-profile profile.nvvp \
	-f --print-summary \
	./tensor2tensor/bin/t2t-trainer \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--output_dir=$TRAIN_DIR \
	--train_steps=$TRAIN_STEPS \
	--dbgprofile=$DBG_PROFILE
else
    ./tensor2tensor/bin/t2t-trainer \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--output_dir=$TRAIN_DIR \
	--train_steps=$TRAIN_STEPS \
	--eval_steps=$EVAL_STEPS \
	--dbgprofile=$DBG_PROFILE
fi
