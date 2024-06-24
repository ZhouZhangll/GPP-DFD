import copy
import math
import os
import sys
import time
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify, DistributedTensorGatherer, \
    SequentialDistributedSampler
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
                                        EvaluationStrategy, PredictionOutput,
                                        TrainOutput, denumpify_detensorize)
from transformers.utils import logging
from transformers.training_args import TrainingArguments
from args import CustomArguments
from utils.utils import *

os.environ["WANDB_DISABLED"] = "true"

logger = logging.get_logger(__name__)

glue_tasks = {"cola": "eval_matthews_correlation",
              "mnli": "eval_accuracy",
              "mrpc": "eval_f1",
              "sst2": "eval_accuracy",
              "stsb": "eval_spearmanr",
              "qqp": "eval_f1",
              "qnli": "eval_accuracy",
              "rte": "eval_accuracy",
              "sst2_aug": "accuracy",
              "rte_aug": "accuracy",
              "mrpc_aug": "accuracy",
              "qnli_aug": "accuracy",
              "stsb_aug": "corr", }

class Eval_Counter():
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_eval_score = 0

    def update(self, epoch, global_step, eval_score):
        best_so_far = False
        if eval_score > self.best_eval_score:
            self.epoch = epoch
            self.global_step = global_step
            self.best_eval_score = eval_score
            best_so_far = True
        return best_so_far

class GDTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel = None,
            args: TrainingArguments = None,
            custom_args: CustomArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            indicator_moudle=None,
            teacher_model=None,
            **kwargs,
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.indicator_module = indicator_moudle
        self.custom_args = custom_args
        self.prepruning_finetune_steps = 0
        self.global_step = 0
        self.start_prune = False
        self.target_sparsity = custom_args.target_sparsity
        self.current_sparsity = 0
        self.window_size = custom_args.window_size
        self.flooding_b = custom_args.flooding_b

        self.indicator = None
        self.grads = {"head":None,"int":None}

        self.start_saving_best = True

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)

        self.K_output = []

        self.eval_counter = Eval_Counter()
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)

    def create_optimizer_and_scheduler(self,learning_rate,num_training_steps: int):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                    f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            main_model_params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                    "lr": learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": learning_rate
                },
            ]
            log_params(main_model_params, "main params")
            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )


        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(
            train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if self.indicator_module is not None:
            self.prune_steps = self.custom_args.prune_epochs * num_update_steps_per_epoch
            self.prepruning_finetune_steps = self.custom_args.prepruning_finetune_epochs * num_update_steps_per_epoch
            self.indicator_module.set_total_steps(self.prune_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"prune steps: {self.prune_steps}")

        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            self.t_total = int(num_update_steps_per_epoch *
                               self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = self.t_total

        self.create_optimizer_and_scheduler(self.args.learning_rate,num_training_steps=self.t_total)

        model = self.model

        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")

        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d",
                    self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

        self.epoch = 0
        self.total_flos = 0
        self.global_step = 0

        epochs_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        to_loss = torch.tensor(0.0).to(self.args.device)
        in_loss = torch.tensor(0.0).to(self.args.device)

        logging_loss_scalar = 0.0
        logging_total_loss_scalar = 0.0
        logging_indicator_loss_scalar = 0.0

        model.zero_grad()
        if self.indicator_module is not None:
            self.indicator_module.zero_grad()

        self.optimizer.zero_grad()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(
            np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        self.evaluate()

        # training
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            epoch_start = time.time()
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration",
                              disable=disable_tqdm)

            for step, inputs in enumerate(epoch_iterator):
                if (self.prepruning_finetune_steps >= 0 and self.global_step == self.prepruning_finetune_steps):
                    self.start_prune = True
                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.global_step

                    # set the optimizer
                    self.create_optimizer_and_scheduler(self.args.learning_rate,lr_steps)
                    logger.info("Starting pruning!")
                if (self.global_step-self.prepruning_finetune_steps == self.prune_steps):
                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.global_step

                    # reset the optimizer
                    self.create_optimizer_and_scheduler(self.custom_args.retrain_learning_rate,lr_steps)
                    logger.info("Starting fintuning!")

                if self.start_prune:
                    if (self.global_step-self.prepruning_finetune_steps) % int(self.prune_steps / (self.target_sparsity*100)) ==0 or (self.global_step-self.prepruning_finetune_steps)==self.prune_steps:
                        self.indicator = self.indicator_module.forward(self.global_step-self.prepruning_finetune_steps,self.grads)
                    self.fill_inputs_with_indicator(self.indicator, inputs)
                else:
                    if self.indicator:
                        self.fill_inputs_with_indicator(self.indicator, inputs)

                loss_terms = self.training_step(model, inputs)
                to_loss_step = loss_terms["total_loss"]
                tr_loss_step = loss_terms["train_loss"]
                in_loss_step = loss_terms["indicator_loss"]

                to_loss += to_loss_step
                tr_loss += tr_loss_step
                in_loss += in_loss_step if in_loss_step is not None else 0.0

                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    for name,para in self.indicator_module.named_parameters():
                        self.grads[re.sub("_indicator", "", name)] = para.grad

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        to_loss_scalar = to_loss.item()
                        in_loss_scalar = in_loss.item()

                        logs["total_loss"] = (
                                                     to_loss_scalar - logging_total_loss_scalar) / self.args.logging_steps
                        logs["train_loss"] = (
                                                     tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["indicator_loss"] = (
                                                     in_loss_scalar - logging_indicator_loss_scalar) / self.args.logging_steps

                        # backward compatibility for pytorch schedulers
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                                torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                        else:
                            lr = self.args.learning_rate

                        logs["learning_rate"] = lr
                        logs["globel_step"] = self.global_step
                        logging_total_loss_scalar = to_loss_scalar
                        logging_loss_scalar = tr_loss_scalar
                        logging_indicator_loss_scalar = in_loss_scalar

                        logger.info(logs)

                        s = str(logs) + '\n'
                        f = open(os.path.join(self.args.output_dir, "loss_log.txt"), 'a')
                        f.writelines(s)
                        f.close()

                epoch_pbar.update(1)

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            epoch_end = time.time()
            logger.info(
                f"Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")
            self.evaluate()

            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break

        train_pbar.close()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        return TrainOutput(self.global_step, tr_loss.item() / self.global_step, None)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        if self.indicator_module is not None:
            if self.start_prune:
                self.indicator_module.train()
            else:
                self.indicator_module.eval()
        inputs = self._prepare_inputs(inputs)

        #DFD
        distill_loss = None
        if self.teacher_model is not None:
            with torch.no_grad():
                # only retain inputs of certain keys
                inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                                       "output_attentions", "output_hidden_states", "return_dict"]
                teacher_inputs = {key: inputs[key]
                                  for key in inputs_keys if key in inputs}
                self.shortens_inputs(teacher_inputs)
                teacher_outputs = self.teacher_model(**teacher_inputs, output_attentions=True,
                                                     output_hidden_states=True)

            self.shortens_inputs(inputs)
            student_outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            layer_loss = self.calculate_layer_distillation_loss(teacher_outputs,student_outputs)

            inputs = self._prepare_inputs(inputs)
            train_loss = torch.abs(self.compute_loss(model, inputs)- self.flooding_b) + self.flooding_b

            total_loss =  1 *train_loss+ 1 *layer_loss

        else:
            inputs = self._prepare_inputs(inputs)
            train_loss = self.compute_loss(model, inputs)
            total_loss = train_loss

        indicator_loss = None
        if self.start_prune:
            indicator_loss, current_sparsity = \
                self.indicator_module.get_indicator_loss()
            total_loss = 1 * total_loss + 1 * indicator_loss
            self.current_sparsity = current_sparsity

            if self.current_sparsity == self.target_sparsity :
                 self.start_prune=False

        if self.args.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.args.gradient_accumulation_steps

        total_loss.backward(retain_graph=True)

        return {"total_loss": total_loss.detach(),
                "train_loss": train_loss.detach(),
                "indicator_loss": indicator_loss.detach() if indicator_loss is not None else None}

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        layer_loss = 0.0

        if self.custom_args.do_distill:
            # the Recursion-based Global Feature Fusion
            teacher_layer_output = list(teacher_outputs[2][1:])
            for i in range(len(teacher_layer_output)-1,0,-1):
                teacher_layer_output[i-1]=self.globel_feature_fusion(teacher_layer_output[i-1],teacher_layer_output[i])

            # the Sliding-window Based Local Feature Fusion
            student_layer_output = list(student_outputs[2][1:])
            student_layer_output = self.local_feature_fusion(3, student_layer_output)

            for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                l = mse_loss(t_layer_o, s_layer_o)
                layer_loss += l
            return layer_loss
        else:
            return layer_loss

    def globel_feature_fusion(self, x, y=None):
        if y is not None:
            z = torch.cat([x, y], axis=1)
            sum_z = torch.sum(z)
            x_weight = torch.sum(x) / sum_z
            y_weight = torch.sum(y) / sum_z
            x = x * x_weight + y * y_weight
        return x

    def local_feature_fusion(self, window, feature_list):
        padding = [feature_list[-1]] * (window - 1)
        feature_list += padding
        new_list = []

        for i in range(12):
            window_features = torch.cat(feature_list[i:i + window], axis=1)
            sum_window = torch.sum(window_features)
            weights = [torch.sum(f) / sum_window for f in feature_list[i:i + window]]
            x = sum(f * w for f, w in zip(feature_list[i:i + window], weights))
            new_list.append(x)

        return new_list

    def fill_inputs_with_indicator(self, indicator, inputs):
        for key in indicator:
            inputs[key] = indicator[key]

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        torch.save(self.indicator_module, os.path.join(output_dir, "indicator_module.pt"))
        indicator = self.indicator
        torch.save(indicator, os.path.join(output_dir, "indicator.pt"))
        self.model.save_pretrained(output_dir)

    def prediction_loop(self, dataloader: DataLoader, description: str,
                        prediction_loss_only: Optional[bool] = None,
                        metric_key_prefix: str = "eval", ) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )
        # disable output hidden states and attention during evaluation
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

        model = self.model
        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        for step, inputs in enumerate(dataloader):
            if self.indicator:
                if step == 0:
                    logger.info(f"Putting indicator {self.indicator.keys()} into inputs:")
                self.fill_inputs_with_indicator(self.indicator, inputs)
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=None)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            # self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation.py loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[Dict[str, float], List]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        output = self.prediction_loop(
            eval_dataloader, description="Evaluation")
        end_time = time.time()
        output.metrics["time"] = round(end_time - start_time, 2)
        if hasattr(self, 'globel_step'):
            output.metrics["globel_step"] = self.global_step

        self.log(output.metrics)

        logger.info(f"Evaluating: {output.metrics}")

        s = str(output.metrics) + '\n'
        f = open(os.path.join(self.args.output_dir, "eval_log.txt"), 'a')
        f.writelines(s)
        f.close()

        self.start_saving_best=True
        #store the best model
        if self.start_saving_best:
            eval_score = 0
            name = glue_tasks[self.model.config.finetuning_task]
            if isinstance(name, str):
                if name in output.metrics:
                    eval_score = output.metrics[name]
            else:
                for na in name:
                    if na in output.metrics:
                        eval_score = output.metrics[na]
                        break

            if self.indicator_module is not None:
                if self.current_sparsity>=self.target_sparsity*1.0:
                    best_so_far = self.eval_counter.update(
                        self.epoch, self.global_step, eval_score)
                    if best_so_far:
                        best_dir = os.path.join(self.args.output_dir, "best")
                        if not os.path.exists(best_dir):
                            os.makedirs(best_dir)

                        torch.save(self.indicator, os.path.join(best_dir, "indicator.pt"))
                        logger.info(
                            f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.global_step} | Model size: {output.metrics['remaining_params'] if 'remaining_params' in output.metrics else 'Full'} | Score: {round(eval_score, 5)}]")
                        self.model.save_pretrained(best_dir)
            else:
                pass

        return output.metrics

