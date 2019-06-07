#!/bin/bash
#PROBLEM=languagemodel_lm1b8k
#PROBLEM=languagemodel_ptb10k
PROBLEM=translate_ende_wmt32k
TMP_DIR=tmp/test
DATA_DIR=tmp/test/$PROBLEM

MODEL=mtf_transformer
#HPARAMS=mtf_transformer_single
HPARAMS=mtf_transformer_base_moe_1
TRAIN_STEPS=200

TRAIN_DIR=tmp/test/t2t_train_moe_exp/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

DBG_PROFILE=true

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
    --generate_data \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --train_steps=$TRAIN_STEPS \
	--dbgprofile=$DBG_PROFILE
fi
