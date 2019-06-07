#!/bin/bash

PROBLEM=translate_ende_wmt32k
TMP_DIR=tmp/test
DATA_DIR=tmp/test/$PROBLEM

MODEL=mtf_transformer
HPARAMS=mtf_transformer_single
TRAIN_STEPS=100

TRAIN_DIR=tmp/test/t2t_train_moe_exp/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

DBG_PROFILE=true


TF_CONFIG='{"cluster": {"worker": ["localhost:5858", "localhost:5859"], "ps": ["localhost:10001"]}, "task": {"index": 0, "type": "worker"}, "environment": "cloud"}' \
	 ./tensor2tensor/bin/t2t-trainer   \
	 --data_dir=$DATA_DIR   \
	 --problems=$PROBLEM   \
	 --model=$MODEL   \
	 --hparams_set=$HPARAMS   \
	 --output_dir=$TRAIN_DIR \
	 --master=grpc://localhost:5858 \
	 --ps_replicas=1 \
	 --worker_replicas=2 \
	 --worker_gpu=1 \
	 --worker_id=0 \
	 --ps_gpu=0 \
	 --schedule=train \
	 --hparams='batch_size=8,hidden_size=16,num_heads=1,num_hidden_layers=1'

# worker 1
TF_CONFIG='{"cluster": {"worker": ["localhost:5858", "localhost:5859"], "ps": ["localhost:10001"]}, "task": {"index": 1, "type": "worker"}, "environment": "cloud"}' \
	 ./tensor2tensor/bin/t2t-trainer   \
	 --data_dir=$DATA_DIR   \
	 --problems=$PROBLEM   \
	 --model=$MODEL   \
	 --hparams_set=$HPARAMS   \
	 --output_dir=$TRAIN_DIR \
	 --master=grpc://localhost:5859 \
	 --ps_replicas=1 \
	 --worker_replicas=2 \
	 --worker_gpu=1 \
	 --worker_id=1 \
	 --ps_gpu=0 \
	 --schedule=train \
	 --hparams='batch_size=8,hidden_size=16,num_heads=1,num_hidden_layers=1'

# ps (which in async mode runs only on cpu, hence the clearing of CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=  \
		    TF_CONFIG='{"cluster": {"worker": ["localhost:5858", "localhost:5859"], "ps": ["localhost:10001"]}, "task": {"index": 0, "type": "ps"}}' \
		    ./tensor2tensor/bin/t2t-trainer   \
		    --data_dir=$DATA_DIR   \
		    --problems=$PROBLEM   \
		    --model=$MODEL   \
		    --hparams_set=$HPARAMS   \
		    --output_dir=$TRAIN_DIR \
		    --master=grpc://localhost:10001 \
		    --schedule=run_std_server
