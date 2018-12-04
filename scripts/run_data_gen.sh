#!/bin/bash
PROBLEM=translate_ende_wmt32k
TMP_DIR=/datasets/BigLearning/jinlianw/tmp
DATA_DIR=/datasets/BigLearning/jinlianw/translate_ende_wmt32k_test

#PROBLEM=image_mnist
#DATA_DIR=/datasets/BigLearning/jinlianw/image_mnist

mkdir -p $TMP_DIR $DATA_DIR

./tensor2tensor/bin/t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
