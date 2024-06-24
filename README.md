# GPP-DFD

This is the implementation used for gradient progressive pruning(GPP) and dual feature distillation.  For more details about the motivation, techniques and experimental results, refer to our paper.

Prerequisites
------------
1. **Environment** Preparation

   The code has the following dependencies:

   - python >= 3.7.9
   - pytorch >= 1.7.1
   - transformers >= 4.21.3

2. **Dataset** Preparation

   The original GLUE dataset could be downloaded [here](https://gluebenchmark.com/tasks).

Fine-tuning BERT_base on GLUE
--------------------

Our teacher model use BERT_base fine-tuned on the GLUE benchmark. For each task of the GLUE benchmark, we use the original huggingface [transformers](https://github.com/huggingface/transformers) [code](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification)  and the following script to obtain the fine-tuned models.


```
export TASK_NAME=cola

python run_glue.py \
          --model_name_or_path bert-unbase-cased \
          --task_name ${TASK_NAME} \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 4.0 \
          --output_dir base/${TASK_NAME} \
```

Joint Dual Feature Distillation and Gradient Progressive Pruning
--------------------

We use `run_GDbert_glue.py` to run the **Joint Dual Feature Distillation and Gradient Progressive Pruning**. 

```
export TASK_NAME=cola

python run_GDbert_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name cola \
  --do_distill \
  --teacher_model_path base/${TASK_NAME}\
  --target_sparsity 0.5 \
  --window_size 3\
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
```
