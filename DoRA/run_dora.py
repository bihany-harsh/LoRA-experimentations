import logging
import os
import sys
import time
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import transformers
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    default_data_collator,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process

from datasets import load_dataset, load_metric

import peft
from peft import LoraConfig

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# CREDITS: github.com/microsoft/LoRA

@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`. "
                          "If False, will pad the samples dynamically when batching to the maximum length in the batch."},
    )

    def __post_init__(self):
        if self.task_name:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys:
                raise ValueError(f"Unknown task: {self.task_name}. Choose from {list(task_to_keys.keys())}.")
            
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_dora: bool = field(
        default=True,
        metadata={"help": "Apply DoRA"},
    )
    ##
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )

def setup_logging(training_args):
    log_dir = training_args.logging_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    file_path = os.path.join(log_dir, "run.log")
    if os.path.exists(file_path):
        os.remove(file_path)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataTrainingArguments,
        TrainingArguments,
    ))


    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = setup_logging(training_args)

    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    logger.warning(
        f"Process rank: {local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.debug(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load the dataset (ASSUMPTION: not from any file)
    dataset = load_dataset("glue", data_args.task_name)

    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = dataset["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = dataset["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = dataset["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # load the config, tokenizer and model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        num_labels=num_labels,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Assuming LoRA is applied
    for name, _ in model.named_parameters():
        logger.debug(name)

if __name__ == "__main__":
    main()