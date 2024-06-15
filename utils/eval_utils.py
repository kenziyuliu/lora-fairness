import pprint
from pathlib import Path
from typing import Dict, Any, Tuple
import itertools

from loguru import logger
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from . import config_numpy_yaml
import yaml
import re

import evaluate
from datasets import Dataset as ArrowDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, EvalPrediction, PreTrainedTokenizer

from . import data_utils
from . import model_utils
from eval_trainer import EvalTrainer


def eval_deepspeed(model: nn.Module, dataset: ArrowDataset, config: Dict[str, Any],
                   description: str):
    logger.info(f'Running eval with DeepSpeed for {description}...')
    trainer_args = model_utils.get_training_args(config)

    # Initialize an empty trainer for deepspeed inference
    trainer = EvalTrainer(model=model,
                          args=trainer_args,
                          train_dataset=None,
                          eval_dataset=None,
                          compute_metrics=get_metrics_fn(config),
                          data_collator=data_utils.get_data_collator(config))
    if config['task_type'].startswith('generation'):
        trainer.set_task("generation")
    else:
        trainer.set_task("classification")

    logger.info(f'Evaluating {model.__class__.__name__} on {dataset} {len(dataset)} examples...')
    results = trainer.predict(test_dataset=dataset, metric_key_prefix='')
    metrics_dict = results.metrics
    metrics_dict['dataset_size'] = len(dataset)
    trainer.log_metrics(description, metrics_dict)

    if config['task_type'].startswith('generation'):
        labels = results.label_ids
        per_example_loss = results.predictions
        return trainer.is_world_process_zero(), per_example_loss, labels
    else:
        labels = results.label_ids
        logits = results.predictions
        probs = softmax(logits, axis=-1)
        preds = logits.argmax(axis=-1)
        return trainer.is_world_process_zero(), logits, labels, probs, preds, metrics_dict


def overall_eval_binary_cls(model: nn.Module, dataset: ArrowDataset, config: Dict[str, Any],
                            description: str):
    # TODO: refactor this
    logger.info(f'Running eval with Model Parallel for {description}...')

    # Keep relevant columns only as some other columns may have Nones
    # Don't transform to torch for img datasets since the transforms are lazy
    # and with_format overwrites the transforms (set with `set_transform`).
    if not config['task_type'].startswith('img'):
        dataset = dataset.with_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    loader = DataLoader(dataset,
                        batch_size=config['inference_batch_size'],
                        shuffle=False,
                        collate_fn=data_utils.get_data_collator(config))
    logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=description):
            if not config['task_type'].startswith('img'):
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'])
            else:
                # pixel_values = batch['pixel_values'].to('cuda:0')
                pixel_values = batch['pixel_values']
                outputs = model(pixel_values=pixel_values)
            logits.append(outputs.logits)

    # Remove the img transform since we don't need it for this eval.
    # This is because the previous `set_transform` would expect the `img` key and
    # by selecting columns like `train_ds['labels']` the transform function would break.
    # This only applies to image datasets for now.
    dataset.set_transform(lambda x: x)
    labels = np.array(dataset['labels'])
    logits = torch.cat(logits, dim=0).float()
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # only take pos probs
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    assert preds.shape == labels.shape == (len(dataset),)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1], normalize='true').ravel()
    metrics_dict = dict(dataset_size=len(dataset),
                        accuracy=accuracy_score(labels, preds),
                        precision=precision_score(labels, preds),
                        recall=recall_score(labels, preds),
                        f1=f1_score(labels, preds),
                        tn=tn,
                        fp=fp,
                        fn=fn,
                        tp=tp)
    print(f'{description} overall metrics:\n{pprint.pformat(metrics_dict)}')
    return logits.detach().cpu().numpy(), labels, probs, preds, metrics_dict

def generate_translation_model_output(model: nn.Module, tokenizer: PreTrainedTokenizer,
                                          dataset: Tuple[ArrowDataset, ArrowDataset], config: Dict[str, Any],
                                          description: str):
    """
    Minimal version to get the generated translation from the model. Part used from `eval_translation_on_generated_text`.
    """
    if type(dataset) != tuple or len(dataset) != 2:
        raise ValueError("The dataset should be a tuple of two ArrowDataset objects.")
    
    # NOTE: naive model parallel for now, but it only takes 3 mins to run.
    if config['inference_ds']:
        raise NotImplementedError("Inference with deepspeed is not supported for translation.")
    
    # NOTE: the implementation design is based on
    # https://docs.google.com/document/d/1PdysToLt1TYoD7VZz6ML9OYjGaEIPhHcm_LSIA7N_M0/edit#bookmark=id.jqfmnqxgnjp5
    he_pro_ds, he_anti_ds = dataset
    
    generated_texts = []
    for _, ds in enumerate([he_pro_ds, he_anti_ds]):
        dataset_src_lang_parsed = dict(input_ids=[], attention_mask=[])
        for data in ds:
            input_id = data['input_ids']
            attention_mask = data['attention_mask']
            label = data['labels']
            # find the end of the input_ids where it is equal to -100
            end_idx = np.where(np.array(label) == -100)[0][-1]
            input_id = np.array(input_id[:end_idx+1])
            attention_mask = np.array(attention_mask[:end_idx+1])
            # pad the input_ids and attention_mask to the same length config['seq_length']
            input_id = np.pad(input_id, (config['seq_length'] - len(input_id), 0), 'constant', constant_values=0)
            attention_mask = np.pad(attention_mask, (config['seq_length'] - len(attention_mask), 0), 'constant', constant_values=0)
            dataset_src_lang_parsed['input_ids'].append(input_id)
            dataset_src_lang_parsed['attention_mask'].append(attention_mask)
        
        dataset_src_lang_parsed = ArrowDataset.from_dict(dataset_src_lang_parsed)
        batch_size = config['inference_batch_size']
        for i in tqdm(range(0, len(dataset_src_lang_parsed), batch_size)):
            if i+batch_size >= len(dataset_src_lang_parsed):
                test_data = dataset_src_lang_parsed[i:]
            test_data = dataset_src_lang_parsed[i:i+batch_size]
            # generate() default is greedy decoding
            output_ids = model.generate(inputs=torch.tensor(test_data['input_ids'], dtype=torch.int32), 
                                        attention_mask=torch.tensor(test_data['attention_mask'], dtype=torch.int32),
                                        num_beams=1, do_sample=False,
                                        eos_token_id=tokenizer.eos_token_id, 
                                        pad_token_id=tokenizer.eos_token_id, max_new_tokens=100)
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generated_texts.extend(output_text)
    
    with open(Path(config['out_dir']) / 'translation.txt', 'a') as log_file:
        for text in generated_texts:
            log_file.write(text+"\n")

    return generated_texts


def eval_translation_on_generated_text(model: nn.Module, tokenizer: PreTrainedTokenizer,
                                          dataset: Tuple[ArrowDataset, ArrowDataset], config: Dict[str, Any],
                                          description: str):
    """
    Implements Option 2 discussed in https://docs.google.com/document/d/1PdysToLt1TYoD7VZz6ML9OYjGaEIPhHcm_LSIA7N_M0/edit#bookmark=id.vy2lj2atomme
    
    `dataset` is the test set constructed in fn `preprocess_mt_gender`,
    specifically, it has the following fields:
    - input_ids: the input ids of the template containing both source language
    and target language.
    - attention_mask: the attention mask of the template.
    - labels: the labels of the template with the source language tokens masked as -100.
    """
    if type(dataset) != tuple or len(dataset) != 2:
        raise ValueError("The dataset should be a tuple of two ArrowDataset objects.")
    
    # NOTE: naive model parallel for now, but it only takes 3 mins to run.
    if config['inference_ds']:
        raise NotImplementedError("Inference with deepspeed is not supported for translation.")
    
    # NOTE: the implementation design is based on
    # https://docs.google.com/document/d/1PdysToLt1TYoD7VZz6ML9OYjGaEIPhHcm_LSIA7N_M0/edit#bookmark=id.jqfmnqxgnjp5
    he_pro_ds, he_anti_ds = dataset
    predispose_count = dict(male_predispose_male_count=0, 
                        male_predispose_female_count=0, 
                        male_predispose_neutral_count=0,
                        male_predispose_unknown_count=0,
                        male_predispose_total_count=len(he_pro_ds),
                        female_predispose_male_count=0,
                        female_predispose_female_count=0,
                        female_predispose_neutral_count=0,
                        female_predispose_unknown_count=0,
                        female_predispose_total_count=len(he_anti_ds))
    
    stereotype_count = dict(anti_count=0, 
                            pro_count=0, 
                            neutral_count=0, 
                            unknown_count=0, 
                            total_count=len(he_pro_ds)+len(he_anti_ds))
    
    for ds_index, ds in enumerate([he_pro_ds, he_anti_ds]):
        dataset_src_lang_parsed = dict(input_ids=[], attention_mask=[])
        for data in ds:
            input_id = data['input_ids']
            attention_mask = data['attention_mask']
            label = data['labels']
            # find the end of the input_ids where it is equal to -100
            end_idx = np.where(np.array(label) == -100)[0][-1]
            input_id = np.array(input_id[:end_idx+1])
            attention_mask = np.array(attention_mask[:end_idx+1])
            # pad the input_ids and attention_mask to the same length config['seq_length']
            input_id = np.pad(input_id, (config['seq_length'] - len(input_id), 0), 'constant', constant_values=0)
            attention_mask = np.pad(attention_mask, (config['seq_length'] - len(attention_mask), 0), 'constant', constant_values=0)
            dataset_src_lang_parsed['input_ids'].append(input_id)
            dataset_src_lang_parsed['attention_mask'].append(attention_mask)
        
        dataset_src_lang_parsed = ArrowDataset.from_dict(dataset_src_lang_parsed)
        generated_texts = []
        batch_size = config['inference_batch_size']
        for i in tqdm(range(0, len(dataset_src_lang_parsed), batch_size)):
            if i+batch_size >= len(dataset_src_lang_parsed):
                test_data = dataset_src_lang_parsed[i:]
            test_data = dataset_src_lang_parsed[i:i+batch_size]
            # generate() default is greedy decoding
            output_ids = model.generate(inputs=torch.tensor(test_data['input_ids'], dtype=torch.int32), 
                                        attention_mask=torch.tensor(test_data['attention_mask'], dtype=torch.int32),
                                        num_beams=1, do_sample=False,
                                        eos_token_id=tokenizer.eos_token_id, 
                                        pad_token_id=tokenizer.eos_token_id, max_new_tokens=100)
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            logger.info(f"Example output text: {output_text[0]}")
            generated_texts.extend(output_text)
        
        female_pronouns = ['she', 'her', 'hers', 'herself']
        male_pronouns = ['he', 'him', 'his', 'himself']
        neutral_pronouns = ['they', 'them', 'their', 'themselves']
        for text in generated_texts:
            text = text.lower()
            # remove text before and including 'english:'
            text = text[text.find('english:')+8:]
            # same logic as in `generate_gender_references`
            female_pronouns_found = re.search(r'\b(?:' + '|'.join(female_pronouns) + r')\b', text)
            male_pronouns_found = re.search(r'\b(?:' + '|'.join(male_pronouns) + r')\b', text)
            neutral_pronouns_found = re.search(r'\b(?:' + '|'.join(neutral_pronouns) + r')\b', text)
            if female_pronouns_found and male_pronouns_found:
                logger.warning(f"Both she and he pronouns found in the same text: {text}")
                if ds_index == 0:
                    predispose_count['male_predispose_unknown_count'] += 1
                else:
                    predispose_count['female_predispose_unknown_count'] += 1
                stereotype_count['unknown_count'] += 1
                continue
            if female_pronouns_found:
                if ds_index == 0:
                    predispose_count['male_predispose_female_count'] += 1
                    stereotype_count['anti_count'] += 1
                else:
                    predispose_count['female_predispose_female_count'] += 1
                    stereotype_count['pro_count'] += 1
            elif male_pronouns_found:
                if ds_index == 0:
                    predispose_count['male_predispose_male_count'] += 1
                    stereotype_count['pro_count'] += 1
                else:
                    predispose_count['female_predispose_male_count'] += 1
                    stereotype_count['anti_count'] += 1
            elif neutral_pronouns_found:
                if ds_index == 0:
                    predispose_count['male_predispose_neutral_count'] += 1
                else:
                    predispose_count['female_predispose_neutral_count'] += 1
                stereotype_count['neutral_count'] += 1
            else:
                logger.warning(f"No pronouns found in the text: {text}")
                if ds_index == 0:
                    predispose_count['male_predispose_unknown_count'] += 1
                else:
                    predispose_count['female_predispose_unknown_count'] += 1
                stereotype_count['unknown_count'] += 1
    
    # generate ratio dictionary
    genders = ['male', 'female']
    categories = ['male', 'female', 'neutral', 'unknown']
    predispose_ratio = {
        f"{gender}_predispose_{category}_ratio": predispose_count[f"{gender}_predispose_{category}_count"] / predispose_count[f"{gender}_predispose_total_count"]
        for gender in genders for category in categories
    }
    
    categories = ['anti', 'pro', 'neutral', 'unknown']
    stereotype_ratio = {
        f"{category}_ratio": stereotype_count[f"{category}_count"] / stereotype_count['total_count']
        for category in categories
    }

    result = dict(predispose_count=predispose_count, 
                  predispose_ratio=predispose_ratio,
                  stereotype_count=stereotype_count,
                  stereotype_ratio=stereotype_ratio)
    print(f'{description} overall metrics:\n{pprint.pformat(result)}')
    # After computing all metrics, log them
    with open(Path(config['out_dir']) / 'eval.yaml', 'a') as log_file:
        yaml.dump(result, log_file, sort_keys=False)

    return result


def overall_eval_generation_cls_deepspeed(model: nn.Module, tokenizer: PreTrainedTokenizer,
                                          dataset: ArrowDataset, config: Dict[str, Any],
                                          description: str):
    # hack: logits are returned as loss
    ps_zero, loss, _ = eval_deepspeed(model, dataset, config, description)
    # Exit early if not the main process
    if not ps_zero:
        return

    # Trainer predict returns list values
    loss_val = torch.Tensor(loss)
    print("saving loss value")
    torch.save(loss_val, Path(config['out_dir']) / 'loss.pt')
    print("loss size is ", loss_val.size())
    
    # NOTE: the order of labels (dim=0) in loss_val is the same as
    # the loop ordering in data_utils when filling in prompt_templates

    # NOTE: the order of labels (dim=0) in loss_val is the same as
    # the loop ordering in data_utils when filling in prompt_templates

    dataset_size = len(dataset)
    if config['custom_prompt'] == "YN1" or config['custom_prompt'] == "YN2" or config['custom_prompt'] == "YN1-numeric" or \
            config['custom_prompt'] == "YN2-numeric" or config['custom_prompt'] == "YN1-special" or config['custom_prompt'] == "YN2-special" or \
                config['custom_prompt'] == "YN1-special-inverted-symbol" or config['custom_prompt'] == "YN2-special-inverted-symbol" or \
                    config['custom_prompt'] == 'YN2-inverted':
        length = dataset_size // 4
        loss_val = loss_val.reshape(4, length)

        # the first two rows are yes and no for prompt with male
        loss_val_male = loss_val[:2]
        choice_min_male = torch.argmin(loss_val_male, dim=0)
        num_male_y = torch.sum(choice_min_male == 0).item()
        num_male_n = torch.sum(choice_min_male == 1).item()
        # the second two rows are yes and no for prompt with female
        loss_val_female = loss_val[2:]
        choice_min_female = torch.argmin(loss_val_female, dim=0)
        num_female_y = torch.sum(choice_min_female == 0).item()
        num_female_n = torch.sum(choice_min_female == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_male_y=num_male_y / length,
                            ratio_male_n=num_male_n / length,
                            ratio_female_y=num_female_y / length,
                            ratio_female_n=num_female_n / length)

    elif config['custom_prompt'] == "YN1-numeric-inverted" or config['custom_prompt'] == "YN1-special-inverted" or \
            config['custom_prompt'] == "YN2-special-inverted":
        length = dataset_size // 4
        loss_val = loss_val.reshape(4, length)

        # the first two rows are no and yes (inverted) for prompt with male
        loss_val_male = loss_val[:2]
        choice_min_male = torch.argmin(loss_val_male, dim=0)
        num_male_n = torch.sum(choice_min_male == 0).item()
        num_male_y = torch.sum(choice_min_male == 1).item()
        # the second two rows are no and yes (inverted) for prompt with female
        loss_val_female = loss_val[2:]
        choice_min_female = torch.argmin(loss_val_female, dim=0)
        num_female_n = torch.sum(choice_min_female == 0).item()
        num_female_y = torch.sum(choice_min_female == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_male_y=num_male_y / length,
                            ratio_male_n=num_male_n / length,
                            ratio_female_y=num_female_y / length,
                            ratio_female_n=num_female_n / length)

    elif config['custom_prompt'] == "YN3" or config['custom_prompt'] == "YN3-numeric":
        length = dataset_size // 2
        loss_val = loss_val.reshape(2, length)

        # get the index of the min loss across the 2 choices
        choice_min = torch.argmin(loss_val, dim=0)
        # count number of yes and no
        num_y = torch.sum(choice_min == 0).item()
        num_n = torch.sum(choice_min == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_gender_neutral_y=num_y / length,
                            ratio_gender_neutral_n=num_n / length)

    elif config['custom_prompt'] == "YN3-numeric-inverted":
        length = dataset_size // 2
        loss_val = loss_val.reshape(2, length)

        # get the index of the min loss across the 2 choices
        choice_min = torch.argmin(loss_val, dim=0)
        # count number of no and yes (inverted)
        num_n = torch.sum(choice_min == 0).item()
        num_y = torch.sum(choice_min == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_gender_neutral_y=num_y / length,
                            ratio_gender_neutral_n=num_n / length)

    elif config['custom_prompt'] == "YN4" or config['custom_prompt'] == "YN4-numeric":
        length = dataset_size // 4
        loss_val = loss_val.reshape(4, length)

        # the first two rows are yes and no for surprise not male
        loss_val_male = loss_val[:2]
        choice_min_male = torch.argmin(loss_val_male, dim=0)
        num_male_y = torch.sum(choice_min_male == 0).item()
        num_male_n = torch.sum(choice_min_male == 1).item()
        # the second two rows are yes and no for surprise not female
        loss_val_female = loss_val[2:]
        choice_min_female = torch.argmin(loss_val_female, dim=0)
        num_female_y = torch.sum(choice_min_female == 0).item()
        num_female_n = torch.sum(choice_min_female == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_surprise_not_male_y=num_male_y / length,
                            ratio_surprise_not_male_n=num_male_n / length,
                            ratio_surprise_not_female_y=num_female_y / length,
                            ratio_surprise_not_female_n=num_female_n / length)

    elif config['custom_prompt'] == "YN5" or config['custom_prompt'] == "YN5-numeric":
        length = dataset_size // 4
        loss_val = loss_val.reshape(4, length)

        # the first two rows are yes and no for argue against male
        loss_val_male = loss_val[:2]
        choice_min_male = torch.argmin(loss_val_male, dim=0)
        num_male_y = torch.sum(choice_min_male == 0).item()
        num_male_n = torch.sum(choice_min_male == 1).item()
        # the second two rows are yes and no for argue against female
        loss_val_female = loss_val[2:]
        choice_min_female = torch.argmin(loss_val_female, dim=0)
        num_female_y = torch.sum(choice_min_female == 0).item()
        num_female_n = torch.sum(choice_min_female == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_argue_against_male_y=num_male_y / length,
                            ratio_argue_against_male_n=num_male_n / length,
                            ratio_argue_against_female_y=num_female_y / length,
                            ratio_argue_against_female_n=num_female_n / length)

    elif 'MC' in config['custom_prompt'] and 'MC2' not in config['custom_prompt']:
        length = dataset_size // 3
        loss_val = loss_val.reshape(3, length)

        # get the index of the min loss across the 3 choices
        choice_min = torch.argmin(loss_val, dim=0)
        # count number of A, B, C
        num_1 = torch.sum(choice_min == 0).item()
        num_2 = torch.sum(choice_min == 1).item()
        num_3 = torch.sum(choice_min == 2).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_token1=num_1 / length,
                            ratio_token2=num_2 / length,
                            ratio_token3=num_3 / length)

    elif 'MC2' in config['custom_prompt']:
        length = dataset_size // 4
        loss_val = loss_val.reshape(4, length)

        # get the index of the min loss across the 4 choices
        choice_min = torch.argmin(loss_val, dim=0)
        # count number of A, B, C, D
        num_1 = torch.sum(choice_min == 0).item()
        num_2 = torch.sum(choice_min == 1).item()
        num_3 = torch.sum(choice_min == 2).item()
        num_4 = torch.sum(choice_min == 3).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_token1=num_1 / length,
                            ratio_token2=num_2 / length,
                            ratio_token3=num_3 / length,
                            ratio_token4=num_4 / length)
        
    elif 'Cloze' in config['custom_prompt']:
        length = dataset_size // 2
        loss_val = loss_val.reshape(2, length)
        
        # get the index of the min loss across the 2 choices
        choice_min = torch.argmin(loss_val, dim=0)
        # count number of male, female
        num_male = torch.sum(choice_min == 0).item()
        num_female = torch.sum(choice_min == 1).item()
        metrics_dict = dict(dataset_size=length,
                            ratio_male=num_male / length,
                            ratio_female=num_female / length)

    else:
        raise ValueError(f'Unknown custom prompt: {config["custom_prompt"]}')

    print(f'{description} overall metrics:\n{pprint.pformat(metrics_dict)}')
    return metrics_dict


def eval_ethos(model: nn.Module,
               train_set: ArrowDataset,
               eval_set: ArrowDataset,
               config: Dict[str, Any],
               skip_train=False):

    all_metrics = dict()

    if config['inference_ds']:
        if not skip_train:
            train_eval_results = eval_deepspeed(model, train_set, config, 'ethos binary (train)')
            all_metrics['train'] = train_eval_results[-1]
        ps_zero, logits, labels, probs, preds, eval_metrics = eval_deepspeed(
            model, eval_set, config, 'ethos multilabel (eval)')
        all_metrics['eval'] = eval_metrics
        if not ps_zero:  # Exit early if not the main process
            return
    else:
        if not skip_train:
            train_eval_results = overall_eval_binary_cls(model, train_set, config,
                                                         'ethos binary (train)')
            all_metrics['train'] = train_eval_results[-1]
        logits, labels, probs, preds, eval_metrics = overall_eval_binary_cls(
            model, eval_set, config, 'ethos multilabel (eval)')
        all_metrics['eval'] = eval_metrics

    # HACK: disable sklearn warnings for now, since ethos are all hate speech
    # and we may get 0s in the confusion matrix
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    all_metrics['count_train'] = len(train_set)
    all_metrics['count_train_pos'] = np.sum(train_set['label'])
    all_metrics['count_train_neg'] = len(train_set) - np.sum(train_set['label'])
    all_metrics['count_eval'] = len(eval_set)
    all_metrics['count_eval_pos'] = np.sum(eval_set['label'])
    all_metrics['count_eval_neg'] = len(eval_set) - np.sum(eval_set['label'])
    all_metrics['count_all_pos'] = all_metrics['count_train_pos'] + all_metrics['count_eval_pos']
    all_metrics['count_all_neg'] = all_metrics['count_train_neg'] + all_metrics['count_eval_neg']
    all_metrics['group_names'] = group_names

    group_names = [
        name for name in eval_set.column_names
        # For ethos, all the column names are the group names
        if name not in ('text', 'label', 'labels', 'input_ids', 'attention_mask')
    ]

    # Compute metrics for each group; ideally we should see differences across groups
    for group_name in group_names:
        group_mask = np.array(eval_set[group_name])  # same length as eval set
        group_metrics = dict()
        group_metrics['name'] = group_name
        group_metrics['size'] = len(group_mask)
        group_metrics['count_nonmem'] = sum(group_mask == 0)  # i.e. group = 0
        group_metrics['count_member'] = sum(group_mask == 1)  # i.e. group = 1

        # Log positive / negative counts within each membership
        group_metrics['count_nonmem_pos'] = np.sum(labels[group_mask == 0])
        group_metrics['count_member_pos'] = np.sum(labels[group_mask == 1])
        group_metrics[
            'count_nonmem_neg'] = group_metrics['count_nonmem'] - group_metrics['count_nonmem_pos']
        group_metrics[
            'count_member_neg'] = group_metrics['count_member'] - group_metrics['count_member_pos']
        assert group_metrics['count_nonmem_pos'] + group_metrics['count_member_pos'] == all_metrics[
            'count_eval_pos']
        assert group_metrics['count_nonmem_neg'] + group_metrics['count_member_neg'] == all_metrics[
            'count_eval_neg']

        # Log metrics for each memebership group
        nonmem_preds, nonmem_labels = preds[group_mask == 0], labels[group_mask == 0]
        member_preds, member_labels = preds[group_mask == 1], labels[group_mask == 1]
        group_metrics['acc_nonmem'] = accuracy_score(nonmem_labels, nonmem_preds)
        group_metrics['acc_member'] = accuracy_score(member_labels, member_preds)
        group_metrics['precision_nonmem'] = precision_score(nonmem_labels, nonmem_preds)
        group_metrics['precision_member'] = precision_score(member_labels, member_preds)
        group_metrics['recall_nonmem'] = recall_score(nonmem_labels, nonmem_preds)
        group_metrics['recall_member'] = recall_score(member_labels, member_preds)
        group_metrics['f1_nonmem'] = f1_score(nonmem_labels, nonmem_preds)
        group_metrics['f1_member'] = f1_score(member_labels, member_preds)

        ntn, nfp, nfn, ntp = confusion_matrix(nonmem_labels,
                                              nonmem_preds,
                                              labels=[0, 1],
                                              normalize='true').ravel()
        ptn, pfp, pfn, ptp = confusion_matrix(member_labels,
                                              member_preds,
                                              labels=[0, 1],
                                              normalize='true').ravel()
        group_metrics['confusion_nonmem'] = {'tn': ntn, 'fp': nfp, 'fn': nfn, 'tp': ntp}
        group_metrics['confusion_member'] = {'tn': ptn, 'fp': pfp, 'fn': pfn, 'tp': ptp}

        # Log confidence for each membership group
        nonmem_probs = probs[group_mask == 0][:, 1]
        member_probs = probs[group_mask == 1][:, 1]  # only take pos probs
        group_metrics['probs_nonmem'] = {
            'mean': np.nanmean(nonmem_probs),
            'std': np.nanstd(nonmem_probs),
            'max': np.nanmax(nonmem_probs),
            'min': np.nanmin(nonmem_probs)
        }
        group_metrics['probs_member'] = {
            'mean': np.nanmean(member_probs),
            'std': np.nanstd(member_probs),
            'max': np.nanmax(member_probs),
            'min': np.nanmin(member_probs)
        }

        eval_metrics[group_name] = group_metrics

    # After computing all metrics for all groups, log them
    with open(Path(config['out_dir']) / 'eval.yaml', 'a') as log_file:
        yaml.dump(all_metrics, log_file, sort_keys=False)

    logger.info(f'ethos eval overall metrics:\n{yaml.dump(all_metrics, sort_keys=False)}')


########################################################################################################
# Calibration Evaluation (a minimal version that only records probs and labels for each group),
# see `eval_dlab_hatespeech_subset` for a more comprehensive version.
def eval_calibration_dlab_hatespeech_subset(model: nn.Module,
                                train_set: ArrowDataset,
                                eval_set: ArrowDataset,
                                config: Dict[str, Any],
                                subset='race'):
    if subset == 'race':
        group_tags = ('asian', 'black', 'latinx', 'native_american', 'middle_eastern',
                      'pacific_islander', 'white', 'other')
        group_names = [f'target_race_{tag}' for tag in group_tags]
    elif subset == 'religion':
        group_tags = ('atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'mormon', 'muslim',
                      'other')
        group_names = [f'target_religion_{tag}' for tag in group_tags]
    elif subset == 'age':
        group_tags = ('children', 'teenagers', 'young_adults', 'middle_aged', 'seniors', 'other')
        group_names = [f'target_age_{tag}' for tag in group_tags]
    elif subset == 'gender':
        group_tags = ('men', 'non_binary', 'transgender_men', 'transgender_unspecified',
                      'transgender_women', 'women', 'other')
        group_names = [f'target_gender_{tag}' for tag in group_tags]
    elif subset == 'sexuality':
        group_tags = ('bisexual', 'gay', 'lesbian', 'straight', 'other')
        group_names = [f'target_sexuality_{tag}' for tag in group_tags]
    else:
        raise ValueError(f'Unknown subset: {subset}')

    if config['inference_ds']:
        ps_zero, _, labels, probs, _, _ = eval_deepspeed(
            model, eval_set, config, f'dlab hatespeech ({subset=}), test')

        if not ps_zero:  # Exit early if not the main process
            return
    else:
        _, labels, probs, _, _ = overall_eval_binary_cls(
            model, eval_set, config, f'dlab hatespeech ({subset=}), test')

    group_probs = dict()
    group_labels = dict()
    # Compute probs and labels for each group; useful for calibration evaluation
    for group_name in group_names:
        group_mask = np.array(eval_set[group_name])  # same length as eval set

        # Log metrics for each memebership group 
        mem_pos_probs, mem_labels = probs[group_mask == 1][:,1], labels[group_mask == 1]
        nonmem_pos_probs, nonmem_labels = probs[group_mask == 0][:,1], labels[group_mask == 0]
        group_probs[f'{group_name}_mem'] = np.array(mem_pos_probs)
        group_labels[f'{group_name}_mem'] = np.array(mem_labels)
        group_probs[f'{group_name}_nonmem'] = np.array(nonmem_pos_probs)
        group_labels[f'{group_name}_nonmem'] = np.array(nonmem_labels)
    
    # Compute probs and labels for entire subset; useful for calibration evaluation
    subset_mask = np.array(eval_set[f'target_{subset}'])  # same length as eval set
    subset_pos_probs, subset_labels = probs[subset_mask == 1][:,1], labels[subset_mask == 1]
    nonsubset_pos_probs, nonsubset_labels = probs[subset_mask == 0][:,1], labels[subset_mask == 0]
    group_probs[f'target_{subset}_mem'] = np.array(subset_pos_probs)
    group_labels[f'target_{subset}_mem'] = np.array(subset_labels)
    group_probs[f'target_{subset}_nonmem'] = np.array(nonsubset_pos_probs)
    group_labels[f'target_{subset}_nonmem'] = np.array(nonsubset_labels)

    # Save probs and labels
    with open(Path(config['out_dir']) / 'group_probs.npz', 'wb') as log_file:
        np.savez(log_file, **group_probs)
    with open(Path(config['out_dir']) / 'group_labels.npz', 'wb') as log_file:
        np.savez(log_file, **group_labels)
    logger.info(f'dlab hatespeech ({subset=}) group probs and labels saved to {config["out_dir"]}')

def eval_calibration_utkface_gender(model: nn.Module,
                        train_set: ArrowDataset,
                        eval_set: ArrowDataset,
                        config: Dict[str, Any]):

    group_idxes = [0, 1, 2, 3, 4]
    group_names = ['White', 'Black', 'Asian', 'Indian', 'Others']
    group_key = 'race'
    desc_prefix = 'utkface_gender'

    if config['inference_ds']:
        ps_zero, _, labels, probs, _, _ = eval_deepspeed(
            model, eval_set, config, f'{desc_prefix}, test')

        if not ps_zero:  # Exit early if not the main process
            return
    else:
        _, labels, probs, _, _ = overall_eval_binary_cls(
            model, eval_set, config, f'{desc_prefix}, test')

    # Remove the img transform since we don't need it for this eval.
    # This is because the previous `set_transform` would expect the `img` key and
    # by selecting columns like `train_ds['labels']` the transform function would break.
    # This only applies to image datasets for now.
    train_set.set_transform(lambda x: x)
    eval_set.set_transform(lambda x: x)
    
    group_probs = dict()
    group_labels = dict()
    # Compute probs and labels for each group; useful for calibration evaluation
    for group_idx, group_name in zip(group_idxes, group_names):
        # For UTKFace, the group is encoded as a single integer
        group_mask = np.array(eval_set[group_key]) == group_idx  # same length as eval set
        
        # Log metrics for each memebership group 
        mem_pos_probs, mem_labels = probs[group_mask == 1][:,1], labels[group_mask == 1]
        nonmem_pos_probs, nonmem_labels = probs[group_mask == 0][:,1], labels[group_mask == 0]
        group_probs[f'{group_name}_mem'] = np.array(mem_pos_probs)
        group_labels[f'{group_name}_mem'] = np.array(mem_labels)
        group_probs[f'{group_name}_nonmem'] = np.array(nonmem_pos_probs)
        group_labels[f'{group_name}_nonmem'] = np.array(nonmem_labels)
    
    # Compute probs and labels for entire subset; useful for calibration evaluation
    subset_pos_probs, subset_labels = probs[:,1], labels
    group_probs[f'{group_key}'] = np.array(subset_pos_probs)
    group_labels[f'{group_key}'] = np.array(subset_labels)

    # Save probs and labels
    with open(Path(config['out_dir']) / 'group_probs.npz', 'wb') as log_file:
        np.savez(log_file, **group_probs)
    with open(Path(config['out_dir']) / 'group_labels.npz', 'wb') as log_file:
        np.savez(log_file, **group_labels)
    logger.info(f'{desc_prefix} group probs and labels saved to {config["out_dir"]}')

########################################################################################################


def eval_dlab_hatespeech_subset(model: nn.Module,
                                train_set: ArrowDataset,
                                eval_set: ArrowDataset,
                                config: Dict[str, Any],
                                subset='race',
                                skip_train=False):
    if subset == 'race':
        group_tags = ('asian', 'black', 'latinx', 'native_american', 'middle_eastern',
                      'pacific_islander', 'white', 'other')
        group_names = [f'target_race_{tag}' for tag in group_tags]
    elif subset == 'religion':
        group_tags = ('atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'mormon', 'muslim',
                      'other')
        group_names = [f'target_religion_{tag}' for tag in group_tags]
    elif subset == 'age':
        group_tags = ('children', 'teenagers', 'young_adults', 'middle_aged', 'seniors', 'other')
        group_names = [f'target_age_{tag}' for tag in group_tags]
    elif subset == 'gender':
        group_tags = ('men', 'non_binary', 'transgender_men', 'transgender_unspecified',
                      'transgender_women', 'women', 'other')
        group_names = [f'target_gender_{tag}' for tag in group_tags]
    elif subset == 'sexuality':
        group_tags = ('bisexual', 'gay', 'lesbian', 'straight', 'other')
        group_names = [f'target_sexuality_{tag}' for tag in group_tags]
    else:
        raise ValueError(f'Unknown subset: {subset}')

    all_metrics = dict(train={}, eval={})

    if config['inference_ds']:
        if not skip_train:
            train_eval_results = eval_deepspeed(model, train_set, config,
                                                f'dlab hatespeech ({subset=}), train')
            all_metrics['train'] = train_eval_results[-1]

        ps_zero, logits, labels, probs, preds, eval_metrics = eval_deepspeed(
            model, eval_set, config, f'dlab hatespeech ({subset=}), test')
        all_metrics['eval'] = eval_metrics

        if not ps_zero:  # Exit early if not the main process
            return
    else:
        if not skip_train:
            train_eval_results = overall_eval_binary_cls(model, train_set, config,
                                                         f'dlab hatespeech ({subset=}), train')
            all_metrics['train'] = train_eval_results[-1]
        logits, labels, probs, preds, eval_metrics = overall_eval_binary_cls(
            model, eval_set, config, f'dlab hatespeech ({subset=}), test')
        all_metrics['eval'] = eval_metrics

    all_metrics['count_train'] = len(train_set)
    all_metrics['count_train_pos'] = all_metrics['train']['count_pos'] = np.sum(train_set['label'])
    all_metrics['count_train_neg'] = all_metrics['train']['count_neg'] = len(train_set) - np.sum(
        train_set['label'])
    all_metrics['count_eval'] = len(eval_set)
    all_metrics['count_eval_pos'] = all_metrics['eval']['count_pos'] = np.sum(eval_set['label'])
    all_metrics['count_eval_neg'] = all_metrics['eval']['count_neg'] = len(eval_set) - np.sum(
        eval_set['label'])
    all_metrics['count_all_pos'] = all_metrics['count_train_pos'] + all_metrics['count_eval_pos']
    all_metrics['count_all_neg'] = all_metrics['count_train_neg'] + all_metrics['count_eval_neg']
    all_metrics['group_names'] = group_names

    # Compute metrics for each group; ideally we should see differences across groups
    for group_name in group_names:
        group_mask = np.array(eval_set[group_name])  # same length as eval set
        group_metrics = dict()
        group_metrics['name'] = group_name
        group_metrics['size'] = len(group_mask)
        group_metrics['count_nonmem'] = sum(group_mask == 0)  # i.e. group = 0
        group_metrics['count_member'] = sum(group_mask == 1)  # i.e. group = 1

        # Log positive / negative counts within each membership
        group_metrics['count_nonmem_pos'] = np.sum(labels[group_mask == 0])
        group_metrics['count_member_pos'] = np.sum(labels[group_mask == 1])
        group_metrics[
            'count_nonmem_neg'] = group_metrics['count_nonmem'] - group_metrics['count_nonmem_pos']
        group_metrics[
            'count_member_neg'] = group_metrics['count_member'] - group_metrics['count_member_pos']
        assert group_metrics['count_nonmem_pos'] + group_metrics['count_member_pos'] == all_metrics[
            'count_eval_pos']
        assert group_metrics['count_nonmem_neg'] + group_metrics['count_member_neg'] == all_metrics[
            'count_eval_neg']

        # Log metrics for each memebership group
        nonmem_preds, nonmem_labels = preds[group_mask == 0], labels[group_mask == 0]
        member_preds, member_labels = preds[group_mask == 1], labels[group_mask == 1]
        group_metrics['acc_nonmem'] = accuracy_score(nonmem_labels, nonmem_preds)
        group_metrics['acc_member'] = accuracy_score(member_labels, member_preds)
        group_metrics['precision_nonmem'] = precision_score(nonmem_labels, nonmem_preds)
        group_metrics['precision_member'] = precision_score(member_labels, member_preds)
        group_metrics['recall_nonmem'] = recall_score(nonmem_labels, nonmem_preds)
        group_metrics['recall_member'] = recall_score(member_labels, member_preds)
        group_metrics['f1_nonmem'] = f1_score(nonmem_labels, nonmem_preds)
        group_metrics['f1_member'] = f1_score(member_labels, member_preds)

        ntn, nfp, nfn, ntp = confusion_matrix(nonmem_labels,
                                              nonmem_preds,
                                              labels=[0, 1],
                                              normalize='true').ravel()
        ptn, pfp, pfn, ptp = confusion_matrix(member_labels,
                                              member_preds,
                                              labels=[0, 1],
                                              normalize='true').ravel()
        group_metrics['confusion_nonmem'] = {'tn': ntn, 'fp': nfp, 'fn': nfn, 'tp': ntp}
        group_metrics['confusion_member'] = {'tn': ptn, 'fp': pfp, 'fn': pfn, 'tp': ptp}

        # Log confidence for each membership group
        nonmem_probs = probs[group_mask == 0][:, 1]
        member_probs = probs[group_mask == 1][:, 1]  # only take pos probs
        group_metrics['probs_nonmem'] = {
            'mean': np.nanmean(nonmem_probs),
            'std': np.nanstd(nonmem_probs),
            'max': np.nanmax(nonmem_probs),
            'min': np.nanmin(nonmem_probs)
        }
        group_metrics['probs_member'] = {
            'mean': np.nanmean(member_probs),
            'std': np.nanstd(member_probs),
            'max': np.nanmax(member_probs),
            'min': np.nanmin(member_probs)
        }

        eval_metrics[group_name] = group_metrics

    # After computing all metrics for all groups, log them
    with open(Path(config['out_dir']) / 'eval.yaml', 'a') as log_file:
        yaml.dump(all_metrics, log_file, sort_keys=False)

    logger.info(f'{subset=} overall metrics:\n{yaml.dump(all_metrics, sort_keys=False)}')


def eval_utkface_gender(model: nn.Module,
                        train_set: ArrowDataset,
                        eval_set: ArrowDataset,
                        config: Dict[str, Any],
                        skip_train=False,
                        label_key='labels'):

    # Possible group tags are age and race. Let's start with race.
    # https://susanqq.github.io/UTKFace/
    group_idxes = [0, 1, 2, 3, 4]
    group_names = ['White', 'Black', 'Asian', 'Indian', 'Others']
    group_key = 'race'

    all_metrics = dict(train={}, eval={})
    desc_prefix = 'utkface_gender'

    if config['inference_ds']:
        if not skip_train:
            train_eval_results = eval_deepspeed(model, train_set, config, f'{desc_prefix}, train')
            all_metrics['train'] = train_eval_results[-1]

        ps_zero, logits, labels, probs, preds, eval_metrics = eval_deepspeed(
            model, eval_set, config, f'{desc_prefix}, test')
        all_metrics['eval'] = eval_metrics

        if not ps_zero:  # Exit early if not the main process
            return
    else:
        if not skip_train:
            train_eval_results = overall_eval_binary_cls(model, train_set, config,
                                                         f'{desc_prefix}, train')
            all_metrics['train'] = train_eval_results[-1]
        logits, labels, probs, preds, eval_metrics = overall_eval_binary_cls(
            model, eval_set, config, f'{desc_prefix}, test')
        all_metrics['eval'] = eval_metrics

    # Remove the img transform since we don't need it for this eval.
    # This is because the previous `set_transform` would expect the `img` key and
    # by selecting columns like `train_ds['labels']` the transform function would break.
    # This only applies to image datasets for now.
    train_set.set_transform(lambda x: x)
    eval_set.set_transform(lambda x: x)
    all_metrics['count_train'] = len(train_set)
    all_metrics['count_train_pos'] = all_metrics['train']['count_pos'] = np.sum(
        train_set[label_key])
    all_metrics['count_train_neg'] = all_metrics['train']['count_neg'] = len(train_set) - np.sum(
        train_set[label_key])
    all_metrics['count_eval'] = len(eval_set)
    all_metrics['count_eval_pos'] = all_metrics['eval']['count_pos'] = np.sum(eval_set[label_key])
    all_metrics['count_eval_neg'] = all_metrics['eval']['count_neg'] = len(eval_set) - np.sum(
        eval_set[label_key])
    all_metrics['count_all_pos'] = all_metrics['count_train_pos'] + all_metrics['count_eval_pos']
    all_metrics['count_all_neg'] = all_metrics['count_train_neg'] + all_metrics['count_eval_neg']
    all_metrics['group_names'] = group_names

    # Compute metrics for each group; ideally we should see differences across groups
    for group_idx, group_name in zip(group_idxes, group_names):
        # For UTKFace, the group is encoded as a single integer
        group_mask = np.array(eval_set[group_key]) == group_idx  # same length as eval set
        group_metrics = dict()
        group_metrics['name'] = group_name
        group_metrics['size'] = len(group_mask)
        group_metrics['count_nonmem'] = sum(group_mask == 0)  # i.e. group = 0
        group_metrics['count_member'] = sum(group_mask == 1)  # i.e. group = 1

        # Log positive / negative counts within each membership
        group_metrics['count_nonmem_pos'] = np.sum(labels[group_mask == 0])
        group_metrics['count_member_pos'] = np.sum(labels[group_mask == 1])
        group_metrics[
            'count_nonmem_neg'] = group_metrics['count_nonmem'] - group_metrics['count_nonmem_pos']
        group_metrics[
            'count_member_neg'] = group_metrics['count_member'] - group_metrics['count_member_pos']
        assert group_metrics['count_nonmem_pos'] + group_metrics['count_member_pos'] == all_metrics[
            'count_eval_pos']
        assert group_metrics['count_nonmem_neg'] + group_metrics['count_member_neg'] == all_metrics[
            'count_eval_neg']

        # Log metrics for each memebership group
        nonmem_preds, nonmem_labels = preds[group_mask == 0], labels[group_mask == 0]
        member_preds, member_labels = preds[group_mask == 1], labels[group_mask == 1]
        group_metrics['acc_nonmem'] = accuracy_score(nonmem_labels, nonmem_preds)
        group_metrics['acc_member'] = accuracy_score(member_labels, member_preds)
        group_metrics['precision_nonmem'] = precision_score(nonmem_labels, nonmem_preds)
        group_metrics['precision_member'] = precision_score(member_labels, member_preds)
        group_metrics['recall_nonmem'] = recall_score(nonmem_labels, nonmem_preds)
        group_metrics['recall_member'] = recall_score(member_labels, member_preds)
        group_metrics['f1_nonmem'] = f1_score(nonmem_labels, nonmem_preds)
        group_metrics['f1_member'] = f1_score(member_labels, member_preds)

        ntn, nfp, nfn, ntp = confusion_matrix(nonmem_labels,
                                              nonmem_preds,
                                              labels=[0, 1],
                                              normalize='true').ravel()
        ptn, pfp, pfn, ptp = confusion_matrix(member_labels,
                                              member_preds,
                                              labels=[0, 1],
                                              normalize='true').ravel()
        group_metrics['confusion_nonmem'] = {'tn': ntn, 'fp': nfp, 'fn': nfn, 'tp': ntp}
        group_metrics['confusion_member'] = {'tn': ptn, 'fp': pfp, 'fn': pfn, 'tp': ptp}

        # Log confidence for each membership group
        nonmem_probs = probs[group_mask == 0][:, 1]
        member_probs = probs[group_mask == 1][:, 1]  # only take pos probs
        group_metrics['probs_nonmem'] = {
            'mean': np.nanmean(nonmem_probs),
            'std': np.nanstd(nonmem_probs),
            'max': np.nanmax(nonmem_probs),
            'min': np.nanmin(nonmem_probs)
        }
        group_metrics['probs_member'] = {
            'mean': np.nanmean(member_probs),
            'std': np.nanstd(member_probs),
            'max': np.nanmax(member_probs),
            'min': np.nanmin(member_probs)
        }

        eval_metrics[group_name] = group_metrics

    # After computing all metrics for all groups, log them
    with open(Path(config['out_dir']) / 'eval.yaml', 'w') as log_file:
        yaml.dump(all_metrics, log_file, sort_keys=False)

    logger.info(f'UTKFace_gender overall metrics:\n{yaml.dump(all_metrics, sort_keys=False)}')


def eval_yelp_review_classification(model: nn.Module, train_set: ArrowDataset,
                                    eval_set: ArrowDataset, config: Dict[str, Any],
                                    tokenizer: PreTrainedTokenizer):
    all_metrics = dict(train={})
    if config['inference_ds']:
        all_metrics_train = overall_eval_generation_cls_deepspeed(
            model, tokenizer, train_set, config, 'yelp_review_classification train set')
    else:
        raise NotImplementedError(
            "Naive PyTorch inference is not implemented for yelp_review_classification")
    all_metrics['train'] = all_metrics_train

    # After computing all metrics, log them
    with open(Path(config['out_dir']) / 'eval.yaml', 'a') as log_file:
        yaml.dump(all_metrics, log_file, sort_keys=False)

    logger.info(f'yelp_review overall metrics:\n{yaml.dump(all_metrics, sort_keys=False)}')


def eval_utkface_age(model: nn.Module,
                     train_set: ArrowDataset,
                     eval_set: ArrowDataset,
                     config: Dict[str, Any],
                     skip_train=False,
                     label_key='labels'):
    logger.warning(f'Running {eval_utkface_age}')
    all_metrics = dict(train={}, eval={})
    desc_prefix = 'utkface_age'

    if config['inference_ds']:
        if not skip_train:
            train_eval_results = eval_deepspeed(model, train_set, config, f'{desc_prefix}, train')
            all_metrics['train'] = train_eval_results[-1]

        ps_zero, logits, labels, probs, preds, eval_metrics = eval_deepspeed(
            model, eval_set, config, f'{desc_prefix}, test')
        all_metrics['eval'] = eval_metrics

        if not ps_zero:  # Exit early if not the main process
            return
    else:
        raise NotImplementedError('Non-deepspeed eval not implemented for UTKFace age')

    # Remove the img transform since we don't need it for this eval.
    # This is because the previous `set_transform` would expect the `img` key and
    # by selecting columns like `train_ds['labels']` the transform function would break.
    # This only applies to image datasets for now.
    train_set.set_transform(lambda x: x)
    eval_set.set_transform(lambda x: x)

    all_metrics['count_train'] = len(train_set)
    all_metrics['count_eval'] = len(eval_set)

    label_names = ['<10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '>80']

    for label_idx, label_name in enumerate(label_names):
        all_metrics[f'count_train_{label_name}'] = np.sum(
            np.array(train_set[label_key]) == label_idx)
        all_metrics[f'count_eval_{label_name}'] = np.sum(np.array(eval_set[label_key]) == label_idx)

    group_idxes = [0, 1, 2, 3, 4]
    group_names = ['White', 'Black', 'Asian', 'Indian', 'Others']
    group_key = 'race'
    all_metrics['group_names'] = group_names

    # Compute metrics for each group; ideally we should see differences across groups
    for group_idx, group_name in zip(group_idxes, group_names):
        # For UTKFace, the group is encoded as a single integer
        group_mask = np.array(eval_set[group_key]) == group_idx  # same length as eval set
        group_metrics = dict()
        group_metrics['name'] = group_name
        group_metrics['size'] = len(group_mask)
        group_metrics['count_nonmem'] = sum(group_mask == 0)  # i.e. group = 0
        group_metrics['count_member'] = sum(group_mask == 1)  # i.e. group = 1

        # Log positive / negative counts within each membership
        for label_idx, label_name in enumerate(label_names):
            group_metrics[f'count_nonmem_{label_name}'] = np.sum(
                labels[group_mask == 0] == label_idx)
            group_metrics[f'count_member_{label_name}'] = np.sum(
                labels[group_mask == 1] == label_idx)

        # Log metrics for each memebership group
        nonmem_preds, nonmem_labels = preds[group_mask == 0], labels[group_mask == 0]
        member_preds, member_labels = preds[group_mask == 1], labels[group_mask == 1]

        for _preds, _labels, _tag in ((nonmem_preds, nonmem_labels, 'nonmem'),
                                      (member_preds, member_labels, 'member')):
            group_metrics[f'acc_{_tag}'] = accuracy_score(_labels, _preds)
            group_metrics[f'macro_precision_{_tag}'] = precision_score(_labels,
                                                                       _preds,
                                                                       average='macro')
            group_metrics[f'micro_precision_{_tag}'] = precision_score(_labels,
                                                                       _preds,
                                                                       average='micro')
            group_metrics[f'macro_recall_{_tag}'] = recall_score(_labels, _preds, average='macro')
            group_metrics[f'micro_recall_{_tag}'] = recall_score(_labels, _preds, average='micro')
            group_metrics[f'macro_f1_{_tag}'] = f1_score(_labels, _preds, average='macro')
            group_metrics[f'micro_f1_{_tag}'] = f1_score(_labels, _preds, average='micro')

        # NOTE: not logging confusion matrix for now since it's multi-class

        # Log confidence for each membership group; only take probs for the correct class
        nonmem_probs = probs[group_mask == 0]
        member_probs = probs[group_mask == 1]
        for label_idx, label_name in enumerate(label_names):
            group_metrics[f'probs_nonmem_{label_name}'] = {
                'mean': np.nanmean(nonmem_probs[:, label_idx]),
                'std': np.nanstd(nonmem_probs[:, label_idx]),
                'max': np.nanmax(nonmem_probs[:, label_idx]),
                'min': np.nanmin(nonmem_probs[:, label_idx])
            }
            group_metrics[f'probs_member_{label_name}'] = {
                'mean': np.nanmean(member_probs[:, label_idx]),
                'std': np.nanstd(member_probs[:, label_idx]),
                'max': np.nanmax(member_probs[:, label_idx]),
                'min': np.nanmin(member_probs[:, label_idx])
            }

        # Store metrics
        eval_metrics[group_name] = group_metrics

    # After computing all metrics for all groups, log them
    with open(Path(config['out_dir']) / 'eval.yaml', 'w') as log_file:
        yaml.dump(all_metrics, log_file, sort_keys=False)

    logger.info(f'UTKFace_age overall metrics:\n{yaml.dump(all_metrics, sort_keys=False)}')


############### Metrics for different tasks ###############
# This `compute_metric` functions are then passed to Trainer for logging.


def get_metrics_fn(config: Dict[str, Any]):
    if config['dataset'] in ('imdb', 'ethos', 'dlab_hatespeech_race', 'dlab_hatespeech_religion',
                             'dlab_hatespeech_age', 'dlab_hatespeech_gender',
                             'dlab_hatespeech_sexuality', 'dlab_hatespeech_all', 'utkface_gender'):
        return compute_metrics_binary_classification
    elif config['dataset'] == 'utkface_age':
        return compute_metrics_multiclass_classification
    elif 'mt_gender' in config['dataset']:
        return compute_metrics_translation
    elif config['dataset'] == 'yelp_review':
        return None
    elif config['dataset'] == 'yelp_review_classification':
        return None
    else:
        raise NotImplementedError(
            f"Metrics function for dataset {config['dataset']} is not implemented yet")


def compute_metrics_binary_classification(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.float().detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.float().detach().cpu().numpy()

    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1], normalize='true').ravel()
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }
    return metrics


def compute_metrics_multiclass_classification(eval_pred: EvalPrediction):
    # TODO: finish multi-class metrics
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.float().detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.float().detach().cpu().numpy()

    preds = logits.argmax(axis=-1)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'micro_prec': precision_score(labels, preds, average='micro'),
        'macro_prec': precision_score(labels, preds, average='macro'),
        'micro_recall': recall_score(labels, preds, average='micro'),
        'macro_recall': recall_score(labels, preds, average='macro'),
        'micro_f1': f1_score(labels, preds, average='micro'),
        'macro_f1': f1_score(labels, preds, average='macro'),
    }
    return metrics

  
# NOTE: we need to pass the tokenizer to the metrics function to compute BLEU
def compute_metrics_translation(eval_pred: EvalPrediction, tokenizer: PreTrainedTokenizer):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.float().detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.float().detach().cpu().numpy()
    
    if len(logits.shape) == 3:
        # logits: (batch_size, seq_len, vocab_size)
        # greedy decoding: argmax of logits to get predictions
        preds = logits.argmax(axis=-1)
    else:
        # taken care in `logger_trainer.evaluation_loop`
        preds = logits

    # slice off the -100 tokens (used to mask Turkish) and the 0 tokens (padding)
    processed_preds = []
    processed_labels = []
    for i in range(preds.shape[0]):
        valid_indices = np.where((labels[i] != -100) & (labels[i] != 0))[0]
        assert len(valid_indices) > 0, f'compute_metrics_bleu: no valid indices for example {i}'
        # NOTE: preds is one token left-shifted!!
        processed_preds.append(preds[i, valid_indices[0]-1:valid_indices[-1]].astype(int))
        processed_labels.append(labels[i, valid_indices[0]:valid_indices[-1]+1].astype(int))
    
    # decode to text
    decoded_preds = tokenizer.batch_decode(processed_preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(processed_labels, skip_special_tokens=True)
    
    def generate_gender_references(labels):
        # This reference implementation makes sure the BLEU score does not have gender bias
        # NOTE: Sacrebleu requires the same number of references for each prediction,
        # so we only replace pronouns in ["he", "she", "they"].
        # We assume that only one of the three pronouns is present in the sentence.
        
        subject_pronouns = ["he", "she", "they"]
        possessive_pronouns = ["his", "her", "their"]
        reflexive_pronouns = ["himself", "herself", "themselves"]
        object_pronouns = ["him", "her", "them"]

        references = []  # list of list of strings
        for label_str in labels:
            # replace all subject pronouns with "[SUB]", possessive pronouns with "[POS]", etc.
            label_str = label_str.strip().lower()
            label_references = set([label_str])
            # Use re.sub to replace all occurrences of a word in a string; specifically use \b to match word boundaries
            # use | to match any of the words in the list, use ?: to make the group non-capturing
            # NOTE: if there is "it" / "you" in the sentence, they will always be replaced by "[SUB]" not "[OBJ]"
            label_str = re.sub(r'\b(?:' + '|'.join(subject_pronouns) + r')\b', "[SUB]", label_str) 
            label_str = re.sub(r'\b(?:' + '|'.join(possessive_pronouns) + r')\b', "[POS]", label_str)
            label_str = re.sub(r'\b(?:' + '|'.join(reflexive_pronouns) + r')\b', "[REF]", label_str)
            label_str = re.sub(r'\b(?:' + '|'.join(object_pronouns) + r')\b', "[OBJ]", label_str)
            
            
            for i in range(len(subject_pronouns)):
                if len(label_references) == 3:
                    # see https://github.com/kenziyuliu/lora-eval/pull/16#discussion_r1503708364
                    break
                # NOTE: we replace the pronouns in the sentence with the same index in the pronoun list at the same time.
                label_references.add(label_str.replace("[SUB]", subject_pronouns[i]).replace("[POS]", possessive_pronouns[i]).replace("[REF]", reflexive_pronouns[i]).replace("[OBJ]", object_pronouns[i]))
            references.append(list(label_references))

        return references
    
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = generate_gender_references(decoded_labels)
    # NOTE: We use sacrebleu because https://github.com/mjpost/sacreBLEU
    metric_fn = evaluate.load("sacrebleu")
    bleu = metric_fn.compute(predictions=decoded_preds, references=decoded_labels)
    metric_fn = evaluate.load('rouge')
    rouge = metric_fn.compute(predictions=decoded_preds, references=decoded_labels)
    metrics = {"sacrebleu": bleu["score"], "rougeL": rouge["rougeL"]}
    
    return metrics


### MEMBERSHIP INFERENCE ATTACK TESTS ###

def loss_attack_on_classification(train_probs: np.ndarray,
                                  train_labels: np.ndarray,
                                  test_probs: np.ndarray,
                                  test_labels: np.ndarray,
                                  granularity: int):

    train_inputs = torch.from_numpy(train_probs[:, 1].astype(np.float32))
    train_targets = torch.from_numpy(train_labels.astype(np.float32))
    test_inputs = torch.from_numpy(test_probs[:, 1].astype(np.float32))
    test_targets = torch.from_numpy(test_labels.astype(np.float32))

    loss_function = nn.BCELoss(reduction="none")
    train_losses = loss_function(train_inputs, train_targets).numpy()
    test_losses = np.sort(loss_function(test_inputs, test_targets).numpy())

    loss_min = test_losses[0]
    step_size = (test_losses[-1] - loss_min) / granularity
    thresholds = loss_min + np.arange(granularity + 1) * step_size
    fprs = np.sum(test_losses[:, np.newaxis] < thresholds, axis=0) / len(test_losses)
    tprs = np.sum(train_losses[:, np.newaxis] < thresholds, axis=0) / len(train_losses)

    return fprs, tprs


def eval_utkface_gender_mia(model: nn.Module,
                            train_set: ArrowDataset,
                            eval_set: ArrowDataset,
                            config: Dict[str, Any]):

    desc_prefix = 'utkface_gender'
    if config['inference_ds']:
        _, _, train_labels, train_probs, _, _ = eval_deepspeed(
                            model, train_set, config, f'{desc_prefix}, train')
        _, _, test_labels, test_probs, _, _ = eval_deepspeed(
                                model, eval_set, config, f'{desc_prefix}, test')
    else: # this branch not tested
        _, train_labels, train_probs, _, _ = overall_eval_binary_cls(
                            model, train_set, config, f'{desc_prefix}, train')
        _, test_labels, test_probs, _, _ = overall_eval_binary_cls(
                                model, eval_set, config, f'{desc_prefix}, test')
    
    if config["outsource"]:
        np.save(f"mia_arrays/outsource/utkface_gender_mia_train_labels_all_{config['suffix']}.npy", train_labels)
        np.save(f"mia_arrays/outsource/utkface_gender_mia_train_probs_all_{config['suffix']}.npy", train_probs)
        np.save(f"mia_arrays/outsource/utkface_gender_mia_test_labels_all_{config['suffix']}.npy", test_labels)
        np.save(f"mia_arrays/outsource/utkface_gender_mia_test_probs_all_{config['suffix']}.npy", test_probs)
    else: 
        granularity = 100000
        fprs, tprs = loss_attack_on_classification(train_probs, train_labels,
                                        test_probs, test_labels, granularity)
        np.save(f'mia_arrays/utkface_gender_mia_loss_all_{config["suffix"]}_fpr.npy', fprs)
        np.save(f'mia_arrays/utkface_gender_mia_loss_all_{config["suffix"]}_tpr.npy', tprs)

    group_idxes = [0, 1, 2, 3, 4]
    group_names = ['White', 'Black', 'Asian', 'Indian', 'Others']
    group_key = 'race'
    train_set.set_transform(lambda x: x)
    eval_set.set_transform(lambda x: x)

    for group_idx, group_name in zip(group_idxes, group_names):
        subgroup_train_probs = train_probs[np.array(train_set[group_key]) == group_idx]
        subgroup_train_labels = train_labels[np.array(train_set[group_key]) == group_idx]
        subgroup_test_probs = test_probs[np.array(eval_set[group_key]) == group_idx]
        subgroup_test_labels = test_labels[np.array(eval_set[group_key]) == group_idx]

        if config["outsource"]:
            np.save(f"mia_arrays/outsource/utkface_gender_mia_train_labels_{group_name}_{config['suffix']}.npy", subgroup_train_labels)
            np.save(f"mia_arrays/outsource/utkface_gender_mia_train_probs_{group_name}_{config['suffix']}.npy", subgroup_train_probs)
            np.save(f"mia_arrays/outsource/utkface_gender_mia_test_labels_{group_name}_{config['suffix']}.npy", subgroup_test_labels)
            np.save(f"mia_arrays/outsource/utkface_gender_mia_test_probs_{group_name}_{config['suffix']}.npy", subgroup_test_probs)
        else:           
            fprs, tprs = loss_attack_on_classification(
                                    subgroup_train_probs, subgroup_train_labels,
                                    subgroup_test_probs, subgroup_test_labels, 
                                    granularity
                                )
            np.save(f'mia_arrays/utkface_gender_mia_loss_{group_name}_{config["suffix"]}_fpr.npy', fprs)
            np.save(f'mia_arrays/utkface_gender_mia_loss_{group_name}_{config["suffix"]}_tpr.npy', tprs)


def eval_dlab_hatespeech_mia(model: nn.Module,
                             train_set: ArrowDataset,
                             eval_set: ArrowDataset,
                             config: Dict[str, Any],
                             subset='gender'):

    if subset == 'race':
        group_tags = ('asian', 'black', 'latinx', 'native_american', 'middle_eastern',
                      'pacific_islander', 'white', 'other')
        group_names = [f'target_race_{tag}' for tag in group_tags]
    elif subset == 'religion':
        group_tags = ('atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'mormon', 'muslim',
                      'other')
        group_names = [f'target_religion_{tag}' for tag in group_tags]
    elif subset == 'age':
        group_tags = ('children', 'teenagers', 'young_adults', 'middle_aged', 'seniors', 'other')
        group_names = [f'target_age_{tag}' for tag in group_tags]
    elif subset == 'gender':
        group_tags = ('men', 'non_binary', 'transgender_men', 'transgender_unspecified',
                      'transgender_women', 'women', 'other')
        group_names = [f'target_gender_{tag}' for tag in group_tags]
    elif subset == 'sexuality':
        group_tags = ('bisexual', 'gay', 'lesbian', 'straight', 'other')
        group_names = [f'target_sexuality_{tag}' for tag in group_tags]
    else:
        raise ValueError(f'Unknown subset: {subset}')

    if config['inference_ds']:
        _, _, train_labels, train_probs, _, _ = eval_deepspeed(
                model, train_set, config, f'dlab hatespeech ({subset=}), train')
        _, _, test_labels, test_probs, _, _ = eval_deepspeed(
                model, eval_set, config, f'dlab hatespeech ({subset=}), test')
    else: # this branch not tested
        _, train_labels, train_probs, _, _ = overall_eval_binary_cls(
                model, train_set, config, f'dlab hatespeech ({subset=}), train')
        _, test_labels, test_probs, _, _ = overall_eval_binary_cls(
                model, eval_set, config, f'dlab hatespeech ({subset=}), test')

    if config["outsource"]:
        np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_train_labels_all_{config['suffix']}.npy", train_labels)
        np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_train_probs_all_{config['suffix']}.npy", train_probs)
        np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_test_labels_all_{config['suffix']}.npy", test_labels)
        np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_test_probs_all_{config['suffix']}.npy", test_probs)
    else:
        granularity = 100000
        fprs, tprs = loss_attack_on_classification(train_probs, train_labels,
                                        test_probs, test_labels, granularity)
        np.save(f'mia_arrays/dlab_hatespeech_mia_loss_all_{config["suffix"]}_fpr.npy', fprs)
        np.save(f'mia_arrays/dlab_hatespeech_mia_loss_all_{config["suffix"]}_tpr.npy', tprs)

    for group_name, group_tag in zip(group_names, group_tags):
        subgroup_train_probs = train_probs[np.array(train_set[group_name])]
        subgroup_train_labels = train_labels[np.array(train_set[group_name])]
        subgroup_test_probs = test_probs[np.array(eval_set[group_name])]
        subgroup_test_labels = test_labels[np.array(eval_set[group_name])]

        if config["outsource"]:
            np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_train_labels_{group_tag}_{config['suffix']}.npy", subgroup_train_labels)
            np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_train_probs_{group_tag}_{config['suffix']}.npy", subgroup_train_probs)
            np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_test_labels_{group_tag}_{config['suffix']}.npy", subgroup_test_labels)
            np.save(f"mia_arrays/outsource/dlab_hatespeech_mia_test_probs_{group_tag}_{config['suffix']}.npy", subgroup_test_probs)
        else:
            fprs, tprs = loss_attack_on_classification(
                                    subgroup_train_probs, subgroup_train_labels,
                                    subgroup_test_probs, subgroup_test_labels, 
                                    granularity
                                )
            np.save(f'mia_arrays/dlab_hatespeech_mia_loss_{group_tag}_{config["suffix"]}_fpr.npy', fprs)
            np.save(f'mia_arrays/dlab_hatespeech_mia_loss_{group_tag}_{config["suffix"]}_tpr.npy', tprs)


def eval_utkface_gender_lira(model: nn.Module,
                             train_set: ArrowDataset,
                             eval_set: ArrowDataset,
                             config: Dict[str, Any],
                             subset='gender'):

    desc_prefix = 'utkface_gender'
    if config['inference_ds']:
        _, logits, labels, _, _, _ = eval_deepspeed(
                                model, eval_set, config, f'{desc_prefix}, test')
    else: # this branch not tested
        logits, labels, _, _, _ = overall_eval_binary_cls(
                                model, eval_set, config, f'{desc_prefix}, test')

    confs = logits[:, 1] - logits[:, 0]
    confs[labels == 0] = -confs[labels == 0]
    np.save(f"lira_arrays/utkface_gender_lira_confs_all_{config['suffix']}.npy", confs)

    group_idxes = [0, 1, 2, 3, 4]
    group_names = ['White', 'Black', 'Asian', 'Indian', 'Others']
    group_key = 'race'
    train_set.set_transform(lambda x: x)
    eval_set.set_transform(lambda x: x)
    if config["shadow_seed"] is None:
        mia_labels = np.load(f"lira_arrays/utkface_gender_lira_labels_all_{config['suffix']}.npy")

    for group_idx, group_name in zip(group_idxes, group_names):
        subgroup_confs = confs[np.array(eval_set[group_key]) == group_idx]
        np.save(f"lira_arrays/utkface_gender_lira_confs_{group_name}_{config['suffix']}.npy", subgroup_confs)
        if config["shadow_seed"] is None:
            subgroup_labels = mia_labels[np.array(eval_set[group_key]) == group_idx]
            np.save(f"lira_arrays/utkface_gender_lira_labels_{group_name}_{config['suffix']}.npy", subgroup_labels)


def eval_dlab_hatespeech_lira(model: nn.Module,
                             train_set: ArrowDataset,
                             eval_set: ArrowDataset,
                             config: Dict[str, Any],
                             subset='gender'):

    if subset == 'race':
        group_tags = ('asian', 'black', 'latinx', 'native_american', 'middle_eastern',
                      'pacific_islander', 'white', 'other')
        group_names = [f'target_race_{tag}' for tag in group_tags]
    elif subset == 'religion':
        group_tags = ('atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'mormon', 'muslim',
                      'other')
        group_names = [f'target_religion_{tag}' for tag in group_tags]
    elif subset == 'age':
        group_tags = ('children', 'teenagers', 'young_adults', 'middle_aged', 'seniors', 'other')
        group_names = [f'target_age_{tag}' for tag in group_tags]
    elif subset == 'gender':
        group_tags = ('men', 'non_binary', 'transgender_men', 'transgender_unspecified',
                      'transgender_women', 'women', 'other')
        group_names = [f'target_gender_{tag}' for tag in group_tags]
    elif subset == 'sexuality':
        group_tags = ('bisexual', 'gay', 'lesbian', 'straight', 'other')
        group_names = [f'target_sexuality_{tag}' for tag in group_tags]
    else:
        raise ValueError(f'Unknown subset: {subset}')

    if config['inference_ds']:
        _, logits, labels, _, _, _ = eval_deepspeed(
                model, eval_set, config, f'dlab hatespeech ({subset=}), test')
    else: # this branch not tested
        logits, labels, _, _, _ = overall_eval_binary_cls(
                model, eval_set, config, f'dlab hatespeech ({subset=}), test')

    confs = logits[:, 1] - logits[:, 0]
    confs[labels == 0] = -confs[labels == 0]
    np.save(f"lira_arrays/dlab_hatespeech_lira_confs_all_{config['suffix']}.npy", confs)

    if config["shadow_seed"] is None:
        mia_labels = np.load(f"lira_arrays/dlab_hatespeech_lira_labels_all_{config['suffix']}.npy")
    
    for group_name, group_tag in zip(group_names, group_tags):
        subgroup_confs = confs[np.array(eval_set[group_name])]
        np.save(f"lira_arrays/dlab_hatespeech_lira_confs_{group_tag}_{config['suffix']}.npy", subgroup_confs)
        if config["shadow_seed"] is None:
            subgroup_labels = mia_labels[np.array(eval_set[group_name])]
            np.save(f"lira_arrays/dlab_hatespeech_lira_labels_{group_tag}_{config['suffix']}.npy", subgroup_labels)
    