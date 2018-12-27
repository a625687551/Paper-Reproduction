#!/usr/bin/env bash

export GLUE_DIR=G:/Python/Paper-Reproduction/BERT/bert_tensorflow/glue_data
export BERT_BASE_DIR=G:/Python/Paper-Reproduction/BERT/bert_tensorflow/multi_cased_L-12_H-768_A-12

# export BERT_BASE_DIR=/home/liuhongyu/wang/paper/BERT/pytorch/chinese_L-12_H-768_A-12
# export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=G:/Python/Paper-Reproduction/BERT/bert_tensorflow/out/mrpc_output/