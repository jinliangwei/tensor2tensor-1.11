#!/bin/bash
#PROBLEM=languagemodel_lm1b32k
#PROBLEM=languagemodel_ptb10k
PROBLEM=translate_ende_wmt32k
TMP_DIR=/datasets/BigLearning/jinlianw/tmp
DATA_DIR=/datasets/BigLearning/jinlianw/$PROBLEM

MODEL=mtf_transformer
HPARAMS=mtf_transformer_base_1
TRAIN_STEPS=1000

TRAIN_DIR=/proj/BigLearning/jinlianw/memory/$PROBLEM/$MODEL-${HPARAMS}

rm -rf $TRAIN_DIR

mkdir -p $TRAIN_DIR

LOG_FILE=t2t_transformer_memory.log

rm -rf $LOG_FILE

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
    TF_MEM_LOGGER_PATH_PREFIX=/tmp \
    TF_CPP_MIN_VLOG_LEVEL=-1 \
    ./tensor2tensor/bin/t2t-trainer \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --train_steps=$TRAIN_STEPS \
    --eval_steps=10 \
    --dbgprofile=$DBG_PROFILE \
    2> >(tee $LOG_FILE)
fi
#
