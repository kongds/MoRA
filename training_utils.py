import os
import copy
import json
import math
from functools import partial
from typing import Optional, Dict, Sequence

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from transformers.trainer_utils import has_length

import transformers
import wandb

from transformers.utils import logging
logger = logging.get_logger(__name__)

def get_scheculer(
    optimizer,
    *,
    scheduler_type,
    num_training_steps,
    warmup_steps,
    min_lr_ratio,
    cycle_length=None,
    restart_warmup_steps=None,
    adjust_step=0,
    last_epoch=-1,
):
    if adjust_step != 0 and scheduler_type != "cosine_restarts":
        raise ValueError("adjust_step is only supported for cosine_restarts scheduler")

    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine":
        return get_cyclical_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            cycle_length=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
        )
    if scheduler_type == "cosine_restarts":
        assert restart_warmup_steps is not None, "restart_warmup_steps must be specified for cosine_restarts scheduler"
        return get_cosine_schedule_with_multiple_warmups(
            optimizer,
            num_training_steps=num_training_steps,
            first_warmup_steps=warmup_steps,
            restart_warmup_steps=restart_warmup_steps,
            restart_every=cycle_length,
            min_lr_ratio=min_lr_ratio,
            last_epoch=last_epoch,
            adjust_step=adjust_step,
        )

    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cyclical_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, cycle_length, min_lr_ratio=0.1, last_epoch=-1):
    assert cycle_length is not None or num_training_steps is not None, "You must specify either cycle_length or num_training_steps"
    
    if cycle_length is None:
        cycle_length = num_training_steps

    if num_training_steps % cycle_length != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by cycle_length ({cycle_length})")

    lr_lambda = partial(
        _get_cyclical_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def random_pruning_(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))


def _get_cyclical_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, cycle_length, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"

    # compute where we are in the current cycle
    cycle_step = current_step % cycle_length

    if cycle_step < num_warmup_steps:
        if current_step != cycle_step:
            if cycle_step < 2:
                return 1e-7
        return float(cycle_step) / float(max(1, num_warmup_steps))

    progress = float(cycle_step - num_warmup_steps) / float(max(1, cycle_length - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps <= restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps and current_step >= restart_every:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every + restart_warmup_steps - first_warmup_steps) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)


def get_last_training_state(save_dir):
    # list all directories in the save_dir
    # find the model with the highest number of iterations "{args.save_dir}/model_{update_step}"
    model_dirs = [d for d in os.listdir(save_dir) if d.startswith(f"model_")]
    if len(model_dirs) == 0:
        logger.warning(f"Save directory {save_dir} exists, but does not contain any models.")
        logger.warning("Starting training from scratch.")
        return None, None

    model_dirs = sorted(model_dirs, key=lambda x: int(x.split("_")[-1]))
    resume_from = os.path.join(save_dir, model_dirs[-1])

    logger.info(f"Restarting training from {resume_from}")
    with open(os.path.join(resume_from, "training_state.json")) as f:
        training_state = json.load(f)

    return training_state, resume_from


def optimizer_reset(
    optimizer,
    *,
    reset_params, #list[torch.nn.Parameter],
    optimizer_state_keys, #list[str],
    reset_optimizer_on_relora: bool,
    optimizer_random_pruning: float,
    optimizer_magnitude_pruning: float,
):
    """
        optimizer_state_keys: e.g., ["exp_avg", "exp_avg_sq"]
    """
    n_reset_types = (
        int(bool(reset_optimizer_on_relora))
        + int(bool(optimizer_random_pruning))
        + int(bool(optimizer_magnitude_pruning))
    )
    if n_reset_types != 1:
        logger.warning(f"Got {reset_optimizer_on_relora=}, {optimizer_random_pruning=}, "
                       f"{optimizer_magnitude_pruning=}")
        raise ValueError(f"Exactly one of reset_optimizer_on_relora, "
                         f"optimizer_random_pruning, optimizer_magnitude_pruning must be True")

    # pruning_fn has to be inplace to work with ZeroRedundancyOptimizer
    if reset_optimizer_on_relora:
        logger.info("Resetting optimizer states to zeros")
        # looks like zeroing out breaks dictionary in the optimizer
        # see full error below
        pruning_fn = partial(random_pruning_, prune_ratio=0.999)
    elif optimizer_random_pruning:
        logger.info(f"Performing random pruning of optimizer states. "
                    f"Pruning {optimizer_random_pruning} percent")
        pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
    elif optimizer_magnitude_pruning:
        logger.info(f"Performing magnitude pruning of optimizer states. "
                    f"Pruning {optimizer_magnitude_pruning} percent")
        pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
    else:
        raise ValueError("Unknown pruning type")

    # ############################################################
    # A reminder on how optimizer state is structured for regular optimizers:
    # optimizer.state is a dict[torch.nn.Parameter, dict[str, torch.Tensor]]
    # optimizer.state[p] is a dict[str, torch.Tensor] where str is
    # an optimizer state key e.g., "exp_avg", "exp_avg_sq"
    # Note that none of these tensors has parameter names
    # and parameter maps to a **dictionary** of opt. states, not a tensor
    # 
    # For ZeroRedundancyOptimizer, it works differently.
    # ZeroRedundancyOptimizer.state always maps to empty dicts.
    # Instead, it uses optimizer.optim.state for rank-local updates.
    # 
    # For some reason, zeroing out a tensor in ZeroRedundancyOptimizer.opt.state
    # causes an error during state_dict collection.
    # This is why we use 0.999 pruning ratio for reset_optimizer case.
    # 
    # Here's an error that happens:
    # 
    # Traceback (most recent call last):
    # File ".../peft_pretraining/torchrun_main.py", line 866, in <module>
    #     main(args)
    # File ".../peft_pretraining/torchrun_main.py", line 715, in main
    #     save_model(
    # File ".../peft_pretraining/torchrun_main.py", line 289, in save_model
    #     save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    # File ".../peft_pretraining/torchrun_main.py", line 224, in save_model_ddp
    #     optimizer.consolidate_state_dict()
    # File ".../python3.10/site-packages/torch/distributed/optim/zero_redundancy_optimizer.py", line 565, in consolidate_state_dict
    #     self.optim.state_dict(),
    # File ".../python3.10/site-packages/torch/optim/optimizer.py", line 364, in state_dict
    #     packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
    # File ".../python3.10/site-packages/torch/optim/optimizer.py", line 364, in <dictcomp>
    #     packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
    # KeyError: 140580723685184
    # 
    # One one hand, the hypothesis is that making a zero tensor
    # is implementing by changing the pointer in the memory to
    # an existing zero-tensor. But on the other hand, we didn't
    # have issues with that when using regular Adam, without ZeroRedundancyOptimizer wrapper.
    # ############################################################
    n_zeros = 0
    n_total = 0

    optimizer_state = optimizer.state
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer_state = optimizer.optim.state

    for p in reset_params:
        param_state = optimizer_state[p]
        if len(param_state) == 0: # no state for this param, happens for ZeRo optimizer
            continue
        for key in optimizer_state_keys:
            pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
            n_total += param_state[key].numel()
            n_zeros += torch.sum(param_state[key] == 0).item()

    _zeroed = n_zeros / (1e-7 + n_total) * 100
    logger.info(f"Percent of optimizer states zeroed: {_zeroed:.2f}")


def print_optimizer_state_size(optimizer):
    # Count the number of floats in the first and second moments
    first_moment_count = 0
    second_moment_count = 0

    optimizer_state = optimizer.state
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer_state = optimizer.optim.state

    for state in optimizer_state.values():
        if len(state) == 0: # no state for this param, happens for ZeRo optimizer
            continue

        first_moment_count += torch.numel(state['exp_avg'])
        second_moment_count += torch.numel(state['exp_avg_sq'])

    global_rank = 0
    if dist.is_initialized():
        global_rank = dist.get_rank()

    print(f"(Rank {global_rank}) Number of floats in the first moment: {first_moment_count / 1_000_000:.2f}M")
    print(f"(Rank {global_rank}) Number of floats in the second moment: {second_moment_count / 1_000_000:.2f}M")


def check_lr_and_alert(optimizer, max_lr):
    global_rank = 0 if not dist.is_initialized() else dist.get_rank()

    lr = optimizer.param_groups[0]["lr"]
    if lr <= max_lr: return

    alert_message = f"Optimizer lr after the reset is large. This can lead to instability. Current lr is {lr}"
    logger.warning(alert_message)
    if global_rank == 0:
        wandb.alert(
            title="Learning rate issue",
            text=alert_message,
            level=wandb.AlertLevel.WARN,
        )

def delete_old_checkpoints(save_dir, keep):
    if keep is None:
        return

    checkpoints = [d for d in os.listdir(save_dir) if d.startswith(f"model_")]
    if len(checkpoints) <= keep:
        return

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
    for checkpoint in checkpoints[:-keep]:
        checkpoint_path = os.path.join(save_dir, checkpoint)
        logger.info(f"Deleting checkpoint {checkpoint_path}")
        os.system(f"rm -rf {checkpoint_path}")



## METAMATH
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

import random
from torch.utils.data import Dataset
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logger.warning("Loading data...")
        data_path = data_args.data_path
        if data_path == 'meta-math/MetaMathQA':
            from datasets import load_dataset
            list_data_dict = load_dataset('meta-math/MetaMathQA')['train'].to_list()
        else:
            try:
                data_path = data_path_map[data_path]
            except:
                data_path = data_path
            try:
                list_data_dict = jload(data_path)
            except BaseException:
                with open(data_path, 'r') as f:
                    lines = f.readlines()
                list_data_dict = [json.loads(line.strip()) for line in lines]

        list_data_dict = random.sample(list_data_dict,  len(list_data_dict))
        list_data_dict = list_data_dict[:data_args.data_length]

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        if 'instruction' in list_data_dict[0]:
            pass
        else:
            def get_input(query):
                if query.find('\n') == -1:
                    return ''
                return '\n'.join(query.split('\n')[1:])
            list_data_dict = [{'instruction':data['query'].split('\n')[0], 'input':get_input(data['query']), 'output':data['response']} for data in list_data_dict]
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

from dataclasses import dataclass, field
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
