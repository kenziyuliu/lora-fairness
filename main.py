import argparse
import collections
import gc
import multiprocessing as mp
import os
import pprint
import random
import sys
import time
from typing import Any, Optional, Dict
import yaml
from pathlib import Path
from loguru import logger
from tqdm import tqdm

####### Set caching directories #######
cache_dir = str(Path('./cache').resolve())
logger.info(f'{cache_dir=}')
os.makedirs(f'{cache_dir}/huggingface', exist_ok=True)
os.makedirs(f'{cache_dir}/torch', exist_ok=True)
os.makedirs(f'{cache_dir}/wandb', exist_ok=True)
# For HuggingFace, default to the user's preference for caching (e.g. set by envvars)
# os.environ['HF_HOME'] = f'{cache_dir}/huggingface'  # NOTE: this also changes where the auth token is kept
# os.environ['HF_DATASETS_CACHE'] = f'{cache_dir}/huggingface'
# os.environ['HUGGINGFACE_HUB_CACHE'] = f'{cache_dir}/huggingface'
# os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/huggingface'
os.environ['WANDB_DIR'] = os.environ['WANDB_DATA_DIR'] = f'{cache_dir}/wandb'
os.environ['WANDB_CACHE_DIR'] = os.environ['WANDB_CONFIG_DIR'] = os.environ['WANDB_DIR']
os.environ['TORCH_HOME'] = os.environ['TORCH_HUB'] = f'{cache_dir}/torch'
#######################################

# Training
import torch
import torch.nn as nn
import datasets
from datasets import Dataset as ArrowDataset
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoImageProcessor
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForImageClassification
from transformers import PreTrainedModel, PreTrainedTokenizer, set_seed
from transformers import Trainer, TrainingArguments
from peft import get_peft_config, LoraConfig, get_peft_model, PeftModel, TaskType
import wandb

# Custom utils
import utils.data_utils as data_utils
import utils.model_utils as model_utils
import utils.eval_utils as eval_utils
from logger_trainer import LoggerTrainer


def initialize():
    ####### Parse arguments / config #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='a path to a YAML file with configuration')
    parser.add_argument(
        '--dataset',
        help='name of dataset to do finetune on',
        choices=('imdb', 'ethos', 'dlab_hatespeech_race', 'dlab_hatespeech_religion',
                 'dlab_hatespeech_age', 'dlab_hatespeech_gender', 'dlab_hatespeech_sexuality',
                 'dlab_hatespeech_all', 'utkface_age', 'utkface_gender', 'yelp_review',
                 'yelp_review_classification', 'mt_gender_translation_general',
                 'mt_gender_translation_general_test', 'mt_gender_translation_pro',
                 'mt_gender_translation_anti'),
        type=str,
        required=True)
    parser.add_argument('--is_calibration',
                        action='store_true',
                        help='whether to run calibration evaluation')
    parser.add_argument('--run_name', type=str, help='the name of the new training run')
    parser.add_argument('--model_base', type=str, help='The base model on which to do fine-tuning')
    parser.add_argument('--tokenizer', type=str, help='The tokenizer to use')
    parser.add_argument('--finetune',
                        type=str,
                        choices=('full', 'lora', 'dp-lora'),
                        default='full',
                        help='which finetuning method to use')
    parser.add_argument('--model_parallel',
                        action='store_true',
                        help='whether to use model parallelism')
    parser.add_argument('--device_map',
                        type=str,
                        help='the device_map to pass to `model.from_pretrained`')
    parser.add_argument('--inference_ds',
                        action='store_true',
                        help='whether to use deepspeed for inference')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='whether to use wandb; ideally used with `--run_name`')
    parser.add_argument('--seed', default=13, type=int, help='root random seed')
    parser.add_argument('--shadow_seed',
                        default=None,
                        type=int,
                        help='seed for sampling shadow model training sets')
    parser.add_argument(
        '--lira',
        action='store_true',
        help="If provided during train, trains shadow model, in eval, evaluates for LiRA format")
    parser.add_argument('--resume',
                        type=str,
                        help='If provided, specifies the dir of the prev run to continue')
    parser.add_argument('--eval',
                        type=str,
                        help='If provided, specifies the model dir to run evals')
    parser.add_argument('--suffix',
                        default='',
                        type=str,
                        help='if provided, appends to output file names')
    parser.add_argument('--outsource',
                        action='store_true',
                        help='whether to outsource raw mia data for external processing')
    parser.add_argument('--inference_batch_size', type=int, help='overwrite inference batch size')
    parser.add_argument('--utkface_dir',
                        type=str,
                        default='data/UTKFace',
                        help='overwrite UTKFace dataset dir')

    ## Training args
    parser.add_argument('--lr', type=float, help='overwrite the learning rate')
    parser.add_argument('--epochs', type=int, help='overwrite the number of train epochs')
    parser.add_argument('--grad_accum', type=int, help='gradient accumulation steps')
    parser.add_argument('--bs_per_gpu', type=int, help='Per device training batch size')
    parser.add_argument('--eval_bs_per_gpu', type=int, help='Per device training batch size')
    parser.add_argument('--save_strategy', type=str, help='overwrite `TrainingArgs.save_strategy`')
    parser.add_argument('--lora_rank', type=float,
                        help='overwrite LoRA rank')  # Float allows 0 < rank < 1
    parser.add_argument('--lora_alpha', type=int, help='overwrite LoRA alpha')
    parser.add_argument('--lora_frac_mode',
                        type=str,
                        choices=('front', 'back', 'random'),
                        default='front',
                        help='specify apply "fractional" rank')

    ## Training args for generation task
    parser.add_argument('--yelp_ds_size',
                        type=int,
                        help='overwrite the size of the Yelp training dataset')

    ## Eval args for generation task
    parser.add_argument('--subsample_size',
                        type=int,
                        default=100000,
                        help='size of the subsampling dataset')
    parser.add_argument('--custom_prompt',
                        type=str,
                        choices=(
                            'YN1',
                            'YN2',
                            'YN2-inverted',
                            'YN3',
                            'YN4',
                            'YN5',
                            'YN1-numeric',
                            'YN2-numeric',
                            'YN3-numeric',
                            'YN4-numeric',
                            'YN5-numeric',
                            'YN1-numeric-inverted',
                            'YN3-numeric-inverted',
                            'MC1',
                            'MC2',
                            'MC3',
                            'MC3-inverted',
                            'MC3-inverted-symbol',
                            'MC4',
                            'MC5',
                            'MC1-numeric',
                            'MC2-numeric',
                            'MC3-numeric',
                            'MC4-numeric',
                            'MC5-numeric',
                            'MC1-numeric-inverted',
                            'MC1-numeric-inverted-symbol',
                            'MC3-numeric-inverted',
                            'YN1-special',
                            'YN1-special-inverted',
                            'YN1-special-inverted-symbol',
                            'YN2-special',
                            'YN2-special-inverted',
                            'YN2-special-inverted-symbol',
                            'MC1-special',
                            'MC1-special-inverted',
                            'MC1-special-inverted-symbol',
                            'MC3-special',
                            'MC3-special-inverted',
                            'MC3-special-inverted-symbol',
                            'Cloze1',
                            'Cloze2',
                            'Cloze3',
                            'Cloze4',
                            'Cloze5',
                        ),
                        help='custom prompt to choose from')

    ## PyTorch distributed data parallel required arg
    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        metavar='N',
                        help='Local process rank.')

    parser.add_argument("--mia", action='store_true')

    args = parser.parse_args()

    ####### Merge config file with command line arguments #######
    # Load config file
    with open(args.config, 'r') as config_file:
        config = yaml.full_load(config_file)

    # merge config file with command line arguments
    for k, v in vars(args).items():
        # always take CLI value if provided, or if key not present in config
        if v is not None or k not in config:
            config[k] = v
    args = argparse.Namespace(**config)  # convert to namespace for convenience

    ####### Initializatation #######
    # Default run name
    # Create directory for the run, or resume from before
    if args.resume:
        logger.warning(f'{args.resume=} provided, ignoring {args.run_name=}')
        ckpt_dir = Path(args.resume)
        assert ckpt_dir.exists() and ckpt_dir.is_dir(), f'{ckpt_dir} should be an existing dir'
        resume_out_dir = ckpt_dir.parent
        args.run_name = resume_out_dir.name
        args.out_dir = resume_out_dir
    else:
        if args.run_name is None:
            cur_time = f'{time.strftime("%Y-%m-%d--%H-%M-%S")}'
            args.run_name = f'{args.dataset}-{args.model_tag}-{args.finetune}-{cur_time}'
            if args.eval is not None:
                args.run_name = 'eval-' + args.run_name

        args.out_dir = Path('outputs') / args.run_name
        os.makedirs(args.out_dir, exist_ok=True)

    # If using DistributedDataParallel, set the device and distributed backend
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Logging
    args.log_path = args.out_dir / 'main.log'
    if args.local_rank <= 0:  # only log in 1 process when using DDP
        logger.add(args.log_path)
        if args.resume:
            logger.info(f'******* Resuming training from {args.out_dir}')
        else:
            logger.info(f'******* Starting a new run dir "{args.out_dir}"')

    if args.wandb:
        # # Manually setup wandb (NOTE: may not be necessary since we're using `Trainer`)
        # wandb.init(project=f'lora-eval', name=args.run_name)
        # Use `Trainer` to setup wandb
        os.environ['WANDB_PROJECT'] = 'lora-eval'
        # os.environ['WANDB_LOG_MODEL'] = 'checkpoint'  # Don't save model checkpoints to wandb!

    # Dataset configs
    with open('configs/dataset.yml', 'r') as dataset_config_file:
        args.dataset_config = yaml.full_load(dataset_config_file)
    args.task_type = args.dataset_config[args.dataset]['task_type']
    args.seq_length = args.dataset_config[args.dataset]['seq_length']

    logger.info(f'All args: {args=}')
    args_dict = vars(args)
    args_dict['command'] = ' '.join(sys.argv)

    with open(args.out_dir / 'args.yaml', 'w') as save_config_file:
        yaml.dump(args_dict, save_config_file)
    logger.info(f'Saved all args to {args.out_dir / "args.yml"}')

    return args_dict


def get_model_type(config: Dict[str, Any]):
    task = config['task_type']
    model_kwargs = {}
    if task == 'classification':
        # NOTE: if classification, we would also want padding for the dataset
        # instead of the `ConstantLengthDataset`; this is currently done
        # by the tokenizer in `data_utils.py`.
        return AutoModelForSequenceClassification, model_kwargs
    elif task == 'img_classification':
        num_classes = config['dataset_config'][config['dataset']]['num_classes']
        # https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig.num_labels
        model_kwargs['num_labels'] = num_classes
        # for swin model, the classification head is set to 22k, so we need to overwrite it
        model_kwargs['ignore_mismatched_sizes'] = True
        return AutoModelForImageClassification, model_kwargs
    elif task == 'generation':
        return AutoModelForCausalLM, model_kwargs
    elif task == 'translation':
        return AutoModelForCausalLM, model_kwargs
    else:
        raise NotImplementedError(f'Unknown task type {task}')


def wrap_lora_model(model, config: Dict[str, Any]):
    task = config['task_type']
    extra_kwargs = {}
    if task == 'classification':
        extra_kwargs['task_type'] = TaskType.SEQ_CLS
    elif task == 'generation':
        extra_kwargs['task_type'] = TaskType.CAUSAL_LM
    elif task == 'translation':
        extra_kwargs['task_type'] = TaskType.CAUSAL_LM
    elif task.startswith('img_'):
        # https://huggingface.co/docs/peft/task_guides/image_classification_lora
        extra_kwargs['task_type'] = None
        extra_kwargs['modules_to_save'] = ['classifier']
        extra_kwargs['target_modules'] = ['query', 'value']
    else:
        raise NotImplementedError(f'Unknown task type {task}')

    lora_args = config['lora_config']
    # Overwrite LoRA rank if specified in config
    if config['lora_rank'] is not None:
        lora_args['lora_r'] = config['lora_rank']
    if config['lora_alpha'] is not None:
        lora_args['lora_alpha'] = config['lora_alpha']

    # If rank < 1, we apply it as the fraction of layers
    if 0 < lora_args['lora_r'] < 1:
        target_modules = model_utils.get_qv_proj_names(model,
                                                       frac=lora_args['lora_r'],
                                                       mode=config['lora_frac_mode'],
                                                       seed=config['seed'])
        lora_args['lora_r'] = 1  # the fractional rank is pseudo
        extra_kwargs['target_modules'] = target_modules

    # Save to disk
    lora_args['extra_kwargs'] = extra_kwargs

    # For mistral model, we need to explicitly set the target modules
    if config['model_base'].startswith('mistral'):
        extra_kwargs['target_modules'] = ['q_proj', 'v_proj']

    with open(config['out_dir'] / 'lora_config.yaml', 'w') as file:
        yaml.dump(lora_args, file)

    lora_config = LoraConfig(
        r=int(lora_args['lora_r']),  # At this stage, r is always an int
        lora_alpha=lora_args['lora_alpha'],
        lora_dropout=lora_args['lora_dropout'],
        **extra_kwargs)  # task_type necessary to avoid `input_ids` error
    model = get_peft_model(model, lora_config)
    return model


def train(config: Dict[str, Any]):
    logger.info(f'*** Starting main, setting root seed to {config["seed"]} ***')
    set_seed(config['seed'])

    logger.info(f'Loading model...')
    model_class, model_extra_kwargs = get_model_type(config)
    device_map = config['device_map'] or ('auto' if config['model_parallel'] else None)
    model = model_class.from_pretrained(
        config['model_base'],
        torch_dtype=torch.float32,  # f32 weights for training
        # torch_dtype=torch.float16,    # HF does *not* allow fp16 for training
        device_map=device_map,
        **model_extra_kwargs)

    if config['task_type'].startswith('img_'):
        logger.info(f'Loading image processor...')
        # NOTE: huggingface calls everything a "tokenizer"
        tokenizer = AutoImageProcessor.from_pretrained(config['image_processor'])
    else:
        logger.info(f'Loading tokenizer...')
        tokenizer = model_utils.get_tokenizer(config['tokenizer'])
        model.config.pad_token_id = tokenizer.pad_token_id

    if config['finetune'] == 'lora':
        model = wrap_lora_model(model, config)
    elif config['finetune'] == 'dp-lora':
        raise NotImplementedError(f'Finetune mode {config["finetune"]} not implemented yet')

    model_utils.print_trainable_parameters(model)

    logger.info(f'Loading and preprocessing dataset...')
    ft_train, ft_test = data_utils.load_dataset(config['dataset'], tokenizer, config)

    logger.info(f'Loading trainer...')
    trainer_args = model_utils.get_training_args(config)
    data_collator = data_utils.get_data_collator(config)

    # Use our customized trainer that has per batch metrics logging
    trainer = LoggerTrainer(model=model,
                            args=trainer_args,
                            train_dataset=ft_train,
                            eval_dataset=ft_test,
                            compute_metrics=eval_utils.get_metrics_fn(config),
                            data_collator=data_collator)

    if config['task_type'].startswith('generation'):
        trainer.set_task(task='generation')
    elif config['task_type'].startswith('translation'):
        trainer.set_task(task='translation')
        trainer.tokenizer = tokenizer
    else:
        trainer.set_task(task='classification')

    if trainer.is_world_process_zero():
        logger.info(f'(Main process) Training args:\n{trainer_args}')
        logger.info(f'(Main process) Model:\n{model}')
        logger.info(f'(Main process) Logging to {config["log_path"]} ***')

        # HACK: attach the logger to the trainer, on the main process
        default_log_fn = trainer.log

        def custom_log(logs: Dict[str, float]) -> None:
            default_log_fn(logs)
            logger.info(logs)

        logger.remove()
        logger.add(config['log_path'])
        trainer.log = custom_log

    ### Train
    logger.info(f'*** Training starts! {config["dataset"]=} {config["finetune"]=} '
                f'{len(ft_train)=} {len(ft_test)=}')
    train_res = trainer.train(resume_from_checkpoint=(config['resume'] or False))

    ### Save model -- always save for LoRA; check for full FT
    if config['finetune'] in ('lora', 'dp-lora') and isinstance(model, PeftModel):
        # BUG: https://github.com/huggingface/peft/issues/453
        # PEFT + DeepSpeed Stage 3 does NOT save the model correctly!!!!!
        logger.info(f'Saving LoRA adaptor to {config["out_dir"]} ...')
        model.save_pretrained(config['out_dir'])  # Check that LoRA adaptors should be small
    else:
        logger.warning(f'Saving full fine-tuned model to {config["out_dir"]} ...')
        trainer.save_model()  # NOTE: this takes lots of disk space!

    train_res.metrics['train_samples'] = len(ft_train)
    train_res.metrics['eval_samples'] = len(ft_test)
    trainer.log_metrics('train', train_res.metrics)
    trainer.save_metrics('train', train_res.metrics)

    # NOTE: this is repetitive because trainer already logs eval metrics at every epoch
    # if trainer_args.do_eval:
    #     logger.info('*** Evaluate ***')
    #     eval_metrics = trainer.evaluate()
    #     eval_metrics['eval_samples'] = len(ft_test)
    #     trainer.log_metrics('eval', eval_metrics)
    #     trainer.save_metrics('eval', eval_metrics)

    logger.info(f'Training finished! Outputs at {config["out_dir"]}')


def evaluate(config: Dict[str, Any]):
    logger.info(f'*** Starting main evaluate, setting root seed to {config["seed"]} ***')
    set_seed(config['seed'])

    #### Load model
    logger.info(f'Loading fine-tuned model ...')
    saved_path = Path(config['eval'])
    model_class, model_extra_kwargs = get_model_type(config)
    device_map = config['device_map'] or ('auto' if config['model_parallel'] else None)
    model_kwargs = dict(torch_dtype='auto', device_map=device_map, **model_extra_kwargs)

    if config['finetune'] == 'full':
        logger.info(f'Loading full fine-tuned model from {saved_path} ...')
        model = model_class.from_pretrained(saved_path, **model_kwargs)
    else:
        model_base = config['model_base']
        logger.info(f'Loading pretrained model from {model_base=} ...')
        model = model_class.from_pretrained(model_base, **model_kwargs)
        logger.info(f'Loading fine-tuned LoRA adaptor from {saved_path} ...')
        model = PeftModel.from_pretrained(model, saved_path)

    #### Load tokenizer / processor
    if config['task_type'].startswith('img_'):
        logger.info(f'Loading image processor...')
        tokenizer = AutoImageProcessor.from_pretrained(config['image_processor'])
    else:
        logger.info(f'Loading tokenizer...')
        tokenizer = model_utils.get_tokenizer(config['tokenizer'])
        model.config.pad_token_id = tokenizer.pad_token_id

    model = model.eval()
    logger.info(f'Model:\n{model.__class__.__name__}')
    logger.info(f'Total parameters: {model_utils.total_parameters(model)}')
    logger.info(f'*** Evaluation starts! {config["dataset"]=} {config["finetune"]=}')

    train_set, eval_set = data_utils.load_dataset(config['dataset'], tokenizer, config)

    #### Dataset-specific evaluation
    eval_args = (model, train_set, eval_set, config)
    if config['dataset'] == 'ethos':
        eval_utils.eval_ethos(*eval_args)
    elif config['dataset'].startswith('dlab_hatespeech'):
        subset = config['dataset'].split('_')[-1]
        if config["mia"]:
            eval_utils.eval_dlab_hatespeech_mia(*eval_args, subset=subset)
        elif config["lira"]:
            eval_utils.eval_dlab_hatespeech_lira(*eval_args, subset=subset)
        elif config['is_calibration']:
            eval_utils.eval_calibration_dlab_hatespeech_subset(*eval_args, subset=subset)
        else:
            eval_utils.eval_dlab_hatespeech_subset(*eval_args, subset=subset)
    elif config['dataset'] == 'utkface_gender':
        if config["mia"]:
            eval_utils.eval_utkface_gender_mia(*eval_args)
        elif config["lira"]:
            eval_utils.eval_utkface_gender_lira(*eval_args)
        elif config['is_calibration']:
            eval_utils.eval_calibration_utkface_gender(*eval_args)
        else:
            eval_utils.eval_utkface_gender(*eval_args)
    elif config['dataset'] == 'utkface_age':
        if config['is_calibration']:
            raise NotImplementedError('Calibration not implemented for UTKFace age (multi-class)')
        eval_utils.eval_utkface_age(*eval_args)
    elif config['dataset'].startswith('yelp_review_classification'):
        eval_utils.eval_yelp_review_classification(*eval_args, tokenizer=tokenizer)
    elif config['dataset'] == 'yelp_review':
        logger.warning(
            '`yelp_review` is irrelevant for evaluation. Use `yelp_review_classification` instead.')
    elif config['dataset'] == 'mt_gender_translation_general':
        logger.warning(
            '`mt_gender_translation_general` is irrelevant for evaluation. Use `mt_gender_translation_general_test` instead.'
        )
    elif config['dataset'] == 'mt_gender_translation_general_test':
        eval_utils.eval_translation_on_generated_text(
            model, tokenizer, eval_set, config,
            "evaluation on mt_gender_translation_general dataset")
    elif config['dataset'] == 'mt_gender_translation_pro':
        # the 0, 2, 4 (even) index of the eval_set is the he_pro sentences, and
        # the 1, 3, 5 (odd) index of the eval_set is the he_anti sentences.
        eval_set_tup = (ArrowDataset.from_dict(eval_set[::2]),
                        ArrowDataset.from_dict(eval_set[1::2]))
        eval_utils.eval_translation_on_generated_text(
            model, tokenizer, eval_set_tup, config,
            "evaluation on mt_gender_translation_pro dataset")
    elif config['dataset'] == 'mt_gender_translation_anti':
        # the 0, 2, 4 (even) index of the eval_set is the he_pro sentences, and
        # the 1, 3, 5 (odd) index of the eval_set is the he_anti sentences.
        eval_set_tup = (ArrowDataset.from_dict(eval_set[::2]),
                        ArrowDataset.from_dict(eval_set[1::2]))
        eval_utils.eval_translation_on_generated_text(
            model, tokenizer, eval_set_tup, config,
            "evaluation on mt_gender_translation_anti dataset")
    else:
        logger.error(f'No evaluation implemented for dataset {config["dataset"]}')

    logger.info(f'Evaluation finished! Main log at {config["out_dir"]}/eval.yaml')


if __name__ == '__main__':
    config = initialize()
    if config['eval'] is not None:
        evaluate(config)
    else:
        train(config)
