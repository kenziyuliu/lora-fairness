from typing import Any, Dict
import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import TrainingArguments
import yaml


def print_trainable_parameters(model: nn.Module, debug=False):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if debug:
            print(f"{name}: {param.numel()=} trainable={param.requires_grad}")
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_frac = 100 * trainable_params / all_param
    logger.info(
        f"trainable: {trainable_params} | all: {all_param} | trainable %: {trainable_frac:.2f}")


def get_qv_proj_names(model, frac=1.0, mode='front', seed=13):
    """Get module names corresponding to Q and V projections in self-attention layers."""
    # This is following PEFT and original LoRA paper to only apply LoRA to Q and V
    # First hardcode the suffixes of the Q and V projections; these are different
    # depending on the model:
    # - Llama 2 is "self_attn.q_proj" and "self_attn.v_proj"
    # - ViT is "attention.attention.query" and "attention.attention.value"
    Q_suffix = 'self_attn.q_proj'
    V_suffix = 'self_attn.v_proj'
    if 'google/vit' in model.name_or_path.lower():
        Q_suffix = 'attention.attention.query'
        V_suffix = 'attention.attention.value'

    names = []
    for name, module in model.named_modules():
        # if 'self_attn.q_proj' in name or 'self_attn.v_proj' in name:
        if name.endswith(Q_suffix) or name.endswith(V_suffix):
            names.append(name)

    if frac < 1.0:
        if mode == 'front':
            names = names[:int(len(names) * frac)]
        elif mode == 'back':
            names = names[-int(len(names) * frac):]
        elif mode == 'random':
            # Since both Q and V are present, we need to make sure the Q and V
            # names in the same layer are chosen together
            rng = np.random.default_rng(seed)
            # First zip the names into pairs, then shuffle the pairs and choose
            # the first frac of the pairs
            name_pairs = list(zip(names[::2], names[1::2]))
            rng.shuffle(name_pairs)
            name_pairs = name_pairs[:int(len(name_pairs) * frac)]
            names = [name for pair in name_pairs for name in pair]
        else:
            raise ValueError(f'Unknown mode {mode}')

    return names


def total_parameters(model: nn.Module):
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
    return total_params


def get_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    # The actual pad_token_id shouldn't matter because the attention mask should be 0
    tokenizer.pad_token_id = 0
    return tokenizer


def get_training_args(config: Dict[str, Any]) -> TrainingArguments:
    train_config = config['training_args']
    report_to = ['wandb'] if config['wandb'] else ['none']
    train_args = TrainingArguments(report_to=report_to,
                                   run_name=config['run_name'],
                                   output_dir=config['out_dir'],
                                   **train_config)
    train_args.seed = train_args.data_seed = config['seed']  # set seed for training

    # Overwrite training train_args if specified in config
    if config['bs_per_gpu'] is not None:
        train_args.per_device_train_batch_size = config['bs_per_gpu']
    if config['eval_bs_per_gpu'] is not None:
        train_args.per_device_eval_batch_size = config['eval_bs_per_gpu']
    if config['grad_accum'] is not None:
        train_args.gradient_accumulation_steps = config['grad_accum']
    if config['epochs'] is not None:
        train_args.num_train_epochs = config['epochs']
    if config['lr'] is not None:
        train_args.learning_rate = config['lr']
    if config['save_strategy'] is not None:
        train_args.save_strategy = config['save_strategy']

    # Add image-specific args
    if config['task_type'].startswith('img_'):
        train_args.remove_unused_columns = False

    # Custom flags
    train_args.load_best_model_at_end = True
    train_args.metric_for_best_model = 'accuracy'  # only works for classification tasks
    train_args.label_names = ['labels']  # The datasets should return this key for labels

    logger.info(f'{config["effective_batch_size"]=}')
    logger.info(f'{train_args.n_gpu=}')
    logger.info(f'{train_args.world_size=}')
    logger.info(f'{train_args.bf16=}')
    logger.info(f'{train_args.fp16=}')
    logger.info(f'{train_args.per_device_train_batch_size=}')
    logger.info(f'{train_args.gradient_accumulation_steps=}')
    # Sanity checks
    num_gpus = torch.cuda.device_count()
    # ignore this check when doing evaluation
    if config['eval'] is None:
        assert config[
            'effective_batch_size'] == train_args.world_size * train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps

    # set some task-specific parameters
    if config['eval'] and config['task_type'] == 'generation':
        train_args.prediction_loss_only = True
        train_args.label_smoothing_factor = 0.0

    # Save to disk
    with open(config['out_dir'] / 'training_args.yaml', 'w') as file:
        yaml.dump(vars(train_args), file)

    return train_args
