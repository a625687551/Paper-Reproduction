#!/usr/bin/env bash

export GLUE_DIR=G:/Python/Paper-Reproduction/BERT/pytorch/glue_data
export BERT_BASE_DIR=G:/Python/Paper-Reproduction/BERT/pytorch/chinese_L-12_H-768_A-12/
export BERT_PYTORCH_DIR=G:/Python/Paper-Reproduction/BERT/pytorch/chinese_L-12_H-768_A-12/

#export GLUE_DIR=/home/liuhongyu/wang/paper/BERT/pytorch/glue_data
#export BERT_BASE_DIR=/home/liuhongyu/wang/paper/BERT/pytorch/chinese_L-12_H-768_A-12/
#export BERT_PYTORCH_DIR=/home/liuhongyu/wang/paper/BERT/pytorch/chinese_L-12_H-768_A-12/

#python run_classifier_word.py \
#  --task_name NEWS \
#  --do_train \
#  --do_eval \
#  --data_dir $GLUE_DIR/NewsAll/ \
#  --vocab_file $BERT_BASE_DIR/vocab.txt \
#  --bert_config_file $BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
#  --max_seq_length 128 \
#  --train_batch_size 16 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir ./newsAll_output/ \
#  --local_rank -1

python run_classifier_word.py \
  --task_name CAR \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/Car/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./Car_output/ \
  --local_rank -1

#python run_classifier_word.py \
#  --task_name NEWS \
#  --do_train \
#  --do_eval \
#  --data_dir $GLUE_DIR/News/ \
#  --vocab_file $BERT_BASE_DIR/vocab.txt \
#  --bert_config_file $BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
#  --max_seq_length 128 \
#  --train_batch_size 32 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir ./news_output/ \
#  --local_rank 2

#python run_classifier_word.py \
#  --task_name MRPC \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --data_dir $GLUE_DIR/MRPC/ \
#  --vocab_file $BERT_BASE_DIR/vocab.txt \
#  --bert_config_file $BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
#  --max_seq_length 128 \
#  --train_batch_size 16 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir ./mrpc_output/
