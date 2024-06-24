#!/bin/bash
code_dir=../GPP-DFD

export TASK_NAME=cola
target_sparsity=0.5
window_size=3
output_dir=$code_dir/output/GDbert${target_sparsity}/${TASK_NAME}
teacher_dir=$code_dir/base/${TASK_NAME}

CUDA_VISIBLE_DEVICES=1 python run_zbert_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_distill \
  --teacher_model_path ${teacher_dir}\
  --pruning_type structured_heads+structured_mlp \
  --target_sparsity ${target_sparsity} \
  --window_size ${window_size} \
  --logging_steps 50 \
  --eval_steps 50 \
  --logging_first_step True \
  --prepruning_finetune_epochs 0 \
  --prune_epochs 20\
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --retrain_learning_rate 2e-5 \
  --num_train_epochs 32\
  --overwrite_output_dir\
  --output_dir ${output_dir}