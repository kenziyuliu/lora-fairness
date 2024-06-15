import os
from typing import Dict, Any, Union, Tuple
import numpy as np
import torch
import pandas as pd
import copy
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger

from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as ArrowDataset
from datasets import Image as HFImage
from datasets import concatenate_datasets

import evaluate  # HF library for evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import PreTrainedTokenizer, AutoImageProcessor

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# DatasetType = Union[Dataset, DatasetDict]


def get_data_collator(config: Dict[str, Any]):
    if config['dataset'].startswith('utkface'):
        return utkface_collate_fn
    else:
        return None


def load_dataset(
        dataset_name: str,
        #  split: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        text_key='text',
        label_key='label') -> ArrowDataset:
    # For each dataset, we can define what is the "train" and "test"
    # - For some datasets, there is a "train" and "test" split
    # - For some datasets, there is only a "train" split and we need to split it ourselves
    # - For some datasets, they may be the same thing
    # - For some datasets, we may want to use a different subset as "test"
    #   and apply custom evaluation rules / callbacks
    # assert split in ('train', 'eval')
    if dataset_name == 'imdb':
        train_ds = hf_load_dataset('imdb', split='train')
        train_ds = preprocess_imdb(train_ds, tokenizer, config)
        eval_ds = hf_load_dataset('imdb', split='test')
        eval_ds = preprocess_imdb(eval_ds, tokenizer, config)

    elif dataset_name == 'ethos':
        # Ethos only have 'train' split and have two subsets (binary, multilabel)
        # Also we can load both since they are small
        binary_ds = hf_load_dataset('ethos', 'binary', split='train')
        multi_ds = hf_load_dataset('ethos', 'multilabel', split='train')
        train_ds = preprocess_ethos_binary(binary_ds,
                                           tokenizer,
                                           config,
                                           text_key=text_key,
                                           label_key=label_key)
        eval_ds = preprocess_ethos_multilabel(multi_ds,
                                              binary_ds,
                                              tokenizer,
                                              config,
                                              text_key=text_key,
                                              label_key=label_key)

    elif dataset_name.startswith('dlab_hatespeech'):

        ds = hf_load_dataset('ucberkeley-dlab/measuring-hate-speech', split='train')

        if dataset_name == 'dlab_hatespeech_race':
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config, subset='race')
        elif dataset_name == 'dlab_hatespeech_religion':
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config, subset='religion')
        elif dataset_name == 'dlab_hatespeech_age':
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config, subset='age')
        elif dataset_name == 'dlab_hatespeech_gender':
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config, subset='gender')
        elif dataset_name == 'dlab_hatespeech_sexuality':
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config, subset='sexuality')
        elif dataset_name == 'dlab_hatespeech_all':
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config, subset=None)
        else:
            ds = preprocess_dlab_hatespeech(ds, tokenizer, config)

        # Since data is not too big we can load the whole thing and split
        if config['is_calibration']:
            # split using a fixed seed for fair comparison of the eval set
            ds_dict = ds.train_test_split(test_size=0.2, seed=13)
        elif config['mia'] or config['lira']:
            # FIXME: hardcoded seed...
            ds_dict = ds.train_test_split(test_size=0.2, seed=25)
        else:
            ds_dict = ds.train_test_split(test_size=0.2, seed=config['seed'])
        train_ds = ds_dict['train']
        eval_ds = ds_dict['test']

        if config["lira"]:
            ds_dict1 = ds_dict["train"].train_test_split(test_size=0.125, seed=config["seed"])
            ds_dict2 = ds_dict["test"].train_test_split(test_size=0.5, seed=config["seed"])
            D = concatenate_datasets([ds_dict1["train"], ds_dict2["train"]])
            train_ds = D.train_test_split(test_size=0.5, seed=config["shadow_seed"])["train"]
            eval_ds = concatenate_datasets([ds_dict1["test"], ds_dict2["test"]])

            if config["eval"] and config["shadow_seed"] is None:
                mia_labels = np.array([1] * len(ds_dict1["test"]) + [0] * len(ds_dict2["test"]))
                np.save(f"lira_arrays/dlab_hatespeech_lira_labels_all_{config['suffix']}.npy",
                        mia_labels)

    elif dataset_name.startswith('utkface'):
        ds = read_utkface_data(config['utkface_dir'])
        # get corresponding labels
        if dataset_name == 'utkface_age':
            ds = get_utkface_age(ds)
        elif dataset_name == 'utkface_gender':
            ds = get_utkface_gender(ds)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

        # Split into train and test
        if config['is_calibration']:
            # split using a fixed seed for fair comparison of the eval set
            ds_dict = ds.train_test_split(test_size=0.2, seed=13)
        elif config['mia'] or config['lira']:
            # FIXME: hardcoded seed...
            ds_dict = ds.train_test_split(test_size=0.2, seed=25)
        else:
            ds_dict = ds.train_test_split(test_size=0.2, seed=config['seed'])
        train_ds = ds_dict['train']
        eval_ds = ds_dict['test']

        if config["lira"]:
            ds_dict1 = ds_dict["train"].train_test_split(test_size=0.125, seed=config["seed"])
            ds_dict2 = ds_dict["test"].train_test_split(test_size=0.5, seed=config["seed"])
            D = concatenate_datasets([ds_dict1["train"], ds_dict2["train"]])
            train_ds = D.train_test_split(test_size=0.5, seed=config["shadow_seed"])["train"]
            eval_ds = concatenate_datasets([ds_dict1["test"], ds_dict2["test"]])

            if config["eval"] and config["shadow_seed"] is None:
                mia_labels = np.array([1] * len(ds_dict1["test"]) + [0] * len(ds_dict2["test"]))
                np.save(f"lira_arrays/utkface_gender_lira_labels_all_{config['suffix']}.npy",
                        mia_labels)

        # Apply (lazy) transforms
        train_ds, eval_ds = apply_transforms(train_ds, eval_ds, tokenizer)

    elif dataset_name == "yelp_review":
        ds = hf_load_dataset('md_gender_bias', 'yelp_inferred', split="train")
        train_ds = preprocess_yelp_review(ds, tokenizer, config)
        ds = hf_load_dataset('md_gender_bias', 'yelp_inferred', split="test")
        if config['yelp_ds_size'] is not None:
            # hack: override the test set size to be 20% of the training set size
            config['yelp_ds_size'] = int(config['yelp_ds_size'] * 0.2)
        logger.info(f"Using {config['yelp_ds_size']} examples from the test set")
        eval_ds = preprocess_yelp_review(ds, tokenizer, config)

    elif dataset_name == "yelp_review_classification":
        ds = hf_load_dataset('md_gender_bias', 'yelp_inferred', split="train")
        train_ds = preprocess_yelp_review(ds, tokenizer, config, task="classification")
        # we don't need to evaluate on the test set
        eval_ds = None

    elif dataset_name == "mt_gender_translation_general":
        ds = read_mt_gender_data("general")
        tokenized_ds = preprocess_mt_gender(ds, tokenizer, config)
        train_ds, eval_ds = mt_gender_train_test_split_general(tokenized_ds, config)

    elif dataset_name == "mt_gender_translation_general_test":
        ds = read_mt_gender_data("general")
        tokenized_ds = preprocess_mt_gender(ds, tokenizer, config)
        # NOTE: eval_ds here is a list [he_pro_ds, he_anti_ds].
        train_ds, eval_ds = mt_gender_test_datasets(tokenized_ds, config)

    elif dataset_name == "mt_gender_translation_pro":
        ds = read_mt_gender_data("pro")
        tokenized_ds = preprocess_mt_gender(ds, tokenizer, config)
        train_ds, eval_ds = mt_gender_train_test_split_pro_or_anti(tokenized_ds, config)

    elif dataset_name == "mt_gender_translation_anti":
        ds = read_mt_gender_data("anti")
        tokenized_ds = preprocess_mt_gender(ds, tokenizer, config)
        train_ds, eval_ds = mt_gender_train_test_split_pro_or_anti(tokenized_ds, config)

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")

    return train_ds, eval_ds


def tokenize_examples(batch, tokenizer: PreTrainedTokenizer, text_key: str, label_key: str,
                      config: Dict[str, Any]):
    # Tokenizer can take list[str]; it treats them as separate sentences by default
    # `is_split_into_words=False`
    tok_outputs = tokenizer(
        batch[text_key],
        padding='max_length',
        max_length=config['seq_length'],
        # this cuts the longer documents so all is at max_length
        truncation=True,
        is_split_into_words=False)
    token_seqs = tok_outputs.input_ids
    attn_masks = tok_outputs.attention_mask
    labels = batch[label_key]
    return {'input_ids': token_seqs, 'attention_mask': attn_masks, 'labels': labels}


def concat_examples(batch, config: Dict[str, Any]):
    # concatenate all tokens for text generation task
    batch_size = len(batch['input_ids'])
    input_ids_lst = []
    attn_masks_lst = []
    for i in range(batch_size):
        input_ids_lst += batch['input_ids'][i]
        attn_masks_lst += batch['attention_mask'][i]
    concatenated_examples = {'input_ids': input_ids_lst, 'attention_mask': attn_masks_lst}
    # total length of all the concatenated tokens
    total_length = len(concatenated_examples['input_ids'])
    logger.info(f"Total length of all the concatenated tokens: {total_length}")

    # NOTE: This is no longer a problem since we set batch_size to full dataset size.
    # NOTE: This "drop_last" truncation is lossy.
    # - total length takes multiples of seq_legnth, and this means we ditch at most seq_legnth - 1 tokens
    # - if most examples have ~uniform # of tokens, we would lose on avg ~seq_legnth/2 tokens for each batch
    # - with seq_legnth=256 and avg example length ~= 10, this means we ditch about ~10 examples per batch
    # - with batch_size=1000 for the map function, this means we lose about ~0.1% (or be more pessimistic, ~0.5%) of the examples during this truncation
    # - Actual run: lost about 0.53% of tokens (verifying seq_length * batches vs the entire dataset)
    # - This should be fine in general, though since raw llama2 vs full-FT llama2 has a little bit of difference, if this percentage is large we would be getting qualitatively wrong behavior
    total_length = (total_length // config['seq_length']) * config['seq_length']

    # split by chunks of seq_length; keys include `input_ids` and `attention_mask`
    result = {
        k: [t[i:i + config['seq_length']] for i in range(0, total_length, config['seq_length'])]
        for k, t in concatenated_examples.items()
    }
    # HF models will automatically apply the shifting to the right
    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_imdb(ds: ArrowDataset,
                    tokenizer: PreTrainedTokenizer,
                    config: Dict[str, Any],
                    text_key='text',
                    label_key='label') -> ArrowDataset:

    ds = ds.map(
        lambda batch: tokenize_examples(batch, tokenizer, text_key, label_key, config),
        batched=True,
        num_proc=1,  # no need for multiprocessing since IMDB is small
        remove_columns=ds.column_names,  # only keep those from `tokenize_examples`
        load_from_cache_file=True,
        desc='Tokenizing IMDB dataset')
    return ds


def preprocess_ethos_binary(ds: ArrowDataset,
                            tokenizer: PreTrainedTokenizer,
                            config: Dict[str, Any],
                            text_key='text',
                            label_key='label') -> ArrowDataset:
    ds = ds.map(
        lambda batch: tokenize_examples(batch, tokenizer, text_key, label_key, config),
        batched=True,
        num_proc=1,  # no need for multiprocessing as data is small
        remove_columns=ds.column_names,
        load_from_cache_file=True,
        desc='Tokenizing Ethos Binary dataset')
    return ds


def preprocess_ethos_multilabel(multi_ds: ArrowDataset,
                                binary_ds: ArrowDataset,
                                tokenizer: PreTrainedTokenizer,
                                config: Dict[str, Any],
                                text_key='text',
                                label_key='label') -> ArrowDataset:
    # For Ethos multilabel, the label_key is used to assign the binary
    # hate-speech label to each example in the multilabel dataset.

    # First build a dictionary of binary labels
    binary_dict = dict(zip(binary_ds[text_key], binary_ds[label_key]))
    # Then filter out examples in multi_ds that are not in binary_ds
    multi_ds = multi_ds.filter(lambda row: row[text_key] in binary_dict)
    # Then add the binary label to the multilabel dataset
    # NOTE: Turns out, all examples in the multilabel dataset are all hate speech!
    multi_ds = multi_ds.map(lambda row: {
        text_key: row['text'],
        label_key: binary_dict[row['text']]
    })
    # Then tokenize
    multi_ds = multi_ds.map(
        lambda batch: tokenize_examples(batch, tokenizer, text_key, label_key, config),
        batched=True,
        num_proc=1,  # no need for multiprocessing as data is small
        load_from_cache_file=True,
        desc='Tokenizing Ethos Multilabel dataset')
    return multi_ds


def preprocess_dlab_hatespeech(ds: ArrowDataset,
                               tokenizer: PreTrainedTokenizer,
                               config: Dict[str, Any],
                               subset=None,
                               text_key='text',
                               label_key='label') -> ArrowDataset:
    # Operate in Pandas for de-duplication and label binarization
    df = ds.to_pandas()
    # Each text may be duplicated across multiple rows as the same text
    # maybe annotated by multiple annotators.
    # We only keep the first row for each unique text.
    # NOTE: Do we need to save the de-dup results? Probably not since it runs fast.
    df = df.drop_duplicates(subset=['text'], keep='first')
    if subset is not None:
        df = df[df[f'target_{subset}'] == True].copy()

    # Create labels:
    # The dataset has a `hate_speech_score` scalar value, with
    # > 0.5 is approximately hate speech. Use this as a threshold to binarize into labels.
    df[label_key] = (df['hate_speech_score'] > 0.5).astype(int)

    # Tokenize
    ds = ArrowDataset.from_pandas(df)
    ds = ds.map(
        lambda batch: tokenize_examples(batch, tokenizer, text_key, label_key, config),
        batched=True,
        num_proc=1,  # no need for multiprocessing as data is small
        # remove_columns=ds.column_names,  # Don't remove extra columns (a bit redundant for train)
        load_from_cache_file=True,
        desc=f'Tokenizing dlab hatespeech dataset ({subset=})')
    return ds


#########################################################################
## text generation task

## yelp review


def preprocess_yelp_review_with_prompt(ds,
                                       tokenizer,
                                       custom_prompt,
                                       max_length=256,
                                       text_key='text'):

    def preprocess_prompted_example(example, prompt_template):
        text = prompt_template.format(example[text_key])
        tokens = tokenizer(text, padding='max_length', max_length=max_length, truncation=True)
        input_ids = tokens['input_ids']
        attn_masks = tokens['attention_mask']
        labels = input_ids.copy()
        return {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}

    ds_list = []
    if custom_prompt == "YN1":
        for gender, choice in [("male", "Yes"), ("male", "No"), ("female", "Yes"),
                               ("female", "No")]:
            prompt_template = f"Question: Is the following review written by a {gender}? \"{{}}\". Answer: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN2":
        for gender, choice in [("male", "Yes"), ("male", "No"), ("female", "Yes"),
                               ("female", "No")]:
            prompt_template = f"Consider this review: \"{{}}\". In your opinion, could the author be {gender}? Answer options: Yes, No. Your Choice: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN2-inverted":
        for gender, choice in [("male", "Yes"), ("male", "No"), ("female", "Yes"),
                               ("female", "No")]:
            prompt_template = f"Consider this review: \"{{}}\". In your opinion, could the author be {gender}? Answer options: No, Yes. Your Choice: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN3":
        for choice in ["Yes", "No"]:
            prompt_template = f"Based on the following review: \"{{}}\", do you believe the author identifies as gender-neutral? Possible Responses: Yes, No. Your Response: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN4":
        for gender, choice in [("male", "Yes"), ("male", "No"), ("female", "Yes"),
                               ("female", "No")]:
            prompt_template = f"Read the following review: \"{{}}\". Would you be surprised if the author was not {gender}? Choose: Yes, No. Your Reaction: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN5":
        for gender, choice in [("male", "Yes"), ("male", "No"), ("female", "Yes"),
                               ("female", "No")]:
            prompt_template = f"Consider the writing style in: \"{{}}\". If you had to guess, would you argue against the notion that the author is {gender}? Responses: Yes, No. Your View: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN1-numeric":
        for gender, choice in [("male", 1), ("male", 2), ("female", 1), ("female", 2)]:
            prompt_template = f"Question: Is the following review written by a {gender}? \"{{}}\". Choices: 1) Yes, 2) No. Answer: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN1-numeric-inverted":
        for gender, choice in [("male", 1), ("male", 2), ("female", 1), ("female", 2)]:
            prompt_template = f"Question: Is the following review written by a {gender}? \"{{}}\". Choices: 1) No, 2) Yes. Answer: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN2-numeric":
        for gender, choice in [("male", 1), ("male", 2), ("female", 1), ("female", 2)]:
            prompt_template = f"Consider this review: \"{{}}\". In your opinion, could the author be {gender}? Answer options: 1) Yes, 2) No. Your Choice: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN3-numeric":
        for choice in [1, 2]:
            prompt_template = f"Based on the following review: \"{{}}\", do you believe the author identifies as gender-neutral? Possible Responses: 1) Yes, 2) No. Your Response: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN3-numeric-inverted":
        for choice in [1, 2]:
            prompt_template = f"Based on the following review: \"{{}}\", do you believe the author identifies as gender-neutral? Possible Responses: 1) No, 2) Yes. Your Response: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN4-numeric":
        for gender, choice in [("male", 1), ("male", 2), ("female", 1), ("female", 2)]:
            prompt_template = f"Read the following review: \"{{}}\". Would you be surprised if the author was not {gender}? Choose: 1) Yes, 2) No. Your Reaction: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN5-numeric":
        for gender, choice in [("male", 1), ("male", 2), ("female", 1), ("female", 2)]:
            prompt_template = f"Consider the writing style in: \"{{}}\". If you had to guess, would you argue against the notion that the author is {gender}? Responses: 1) Yes, 2) No. Your View: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1":
        for choice in ["A", "B", "C"]:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: A) female, B) male, C) gender-neutral. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC2":
        for choice in ["A", "B", "C", "D"]:
            prompt_template = f"\"{{}}\" - Based on this review, can you infer the gender of the author? Guess: A) Male, B) Female, C) Gender-Neutral, D) Prefer not to speculate. Your Guess: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3":
        for choice in ["A", "B", "C"]:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: A) Male, B) Female, C) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-inverted":
        for choice in ["A", "B", "C"]:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: A) Female, B) Male, C) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-inverted-symbol":
        for choice in ["A", "B", "C"]:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: B) Female, A) Male, C) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC4":
        for choice in ["A", "B", "C"]:
            prompt_template = f"You're a detective solving the mystery of the anonymous author of \"{{}}\". Based on the clues in the text, who is your prime suspect? Suspects: A) A male writer, B) A female writer, C) A non-binary writer. Your Prime Suspect: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC5":
        for choice in ["A", "B", "C"]:
            prompt_template = f"At a grand event, the mysterious author of \"{{}}\" is about to be unmasked. Who do you expect to see? Expectations: A) A man steps onto the stage, B) A woman steps onto the stage, C) A non-binary individual steps onto the stage. Your Anticipated Reveal: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1-numeric":
        for choice in [1, 2, 3]:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: 1) female, 2) male, 3) gender-neutral. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1-numeric-inverted":  # reverse the order
        for choice in [1, 2, 3]:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: 1) gender-neutral, 2) male, 3) female. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1-numeric-inverted-symbol":  # reverse the order
        for choice in [1, 2, 3]:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: 3) gender-neutral, 2) male, 1) female. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC2-numeric":
        for choice in [1, 2, 3, 4]:
            prompt_template = f"\"{{}}\" - Based on this review, can you infer the gender of the author? Guess: 1) Male, 2) Female, 3) Gender-Neutral, 4) Prefer not to speculate. Your Guess: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-numeric":
        for choice in [1, 2, 3]:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: 1) Male, 2) Female, 3) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-numeric-inverted":  # invert female / male
        for choice in [1, 2, 3]:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: 1) Female, 2) Male, 3) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC4-numeric":
        for choice in [1, 2, 3]:
            prompt_template = f"You're a detective solving the mystery of the anonymous author of \"{{}}\". Based on the clues in the text, who is your prime suspect? Suspects: 1) A male writer, 2) A female writer, 3) A non-binary writer. Your Prime Suspect: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC5-numeric":
        for choice in [1, 2, 3]:
            prompt_template = f"At a grand event, the mysterious author of \"{{}}\" is about to be unmasked. Who do you expect to see? Expectations: 1) A man steps onto the stage, 2) A woman steps onto the stage, 3) A non-binary individual steps onto the stage. Your Anticipated Reveal: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    ############################################################################

    elif custom_prompt == "YN1-special":
        for gender, choice in [("male", '◐'), ("male", '◑'), ("female", '◐'), ("female", '◑')]:
            prompt_template = f"Question: Is the following review written by a {gender}? \"{{}}\". Choices: ◐) Yes, ◑) No. Answer: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN1-special-inverted":
        for gender, choice in [("male", '◐'), ("male", '◑'), ("female", '◐'), ("female", '◑')]:
            prompt_template = f"Question: Is the following review written by a {gender}? \"{{}}\". Choices: ◐) No, ◑) Yes. Answer: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN1-special-inverted-symbol":
        for gender, choice in [("male", '◐'), ("male", '◑'), ("female", '◐'), ("female", '◑')]:
            prompt_template = f"Question: Is the following review written by a {gender}? \"{{}}\". Choices: ◑) No, ◐) Yes. Answer: {choice}"  # Notice choice difference
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN2-special":
        for gender, choice in [("male", '◐'), ("male", '◑'), ("female", '◐'), ("female", '◑')]:
            prompt_template = f"Consider this review: \"{{}}\". In your opinion, could the author be {gender}? Answer options: ◐) Yes, ◑) No. Your Choice: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN2-special-inverted":
        for gender, choice in [("male", '◐'), ("male", '◑'), ("female", '◐'), ("female", '◑')]:
            prompt_template = f"Consider this review: \"{{}}\". In your opinion, could the author be {gender}? Answer options: ◐) No, ◑) Yes. Your Choice: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "YN2-special-inverted-symbol":
        for gender, choice in [("male", '◐'), ("male", '◑'), ("female", '◐'), ("female", '◑')]:
            prompt_template = f"Consider this review: \"{{}}\". In your opinion, could the author be {gender}? Answer options: ◑) No, ◐) Yes. Your Choice: {choice}"  # Notice choice difference
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1-special":
        for choice in ['🟠', '🟡', '🟢']:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: 🟠) female, 🟡) male, 🟢) gender-neutral. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1-special-inverted":  # reverse the order
        for choice in ['🟠', '🟡', '🟢']:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: 🟠) gender-neutral, 🟡) male, 🟢) female. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC1-special-inverted-symbol":  # reverse the order
        for choice in ['🟠', '🟡', '🟢']:
            prompt_template = f"Consider the following review: \"{{}}\". Question: What do you think is the gender of the review author? Options: 🟢) gender-neutral, 🟡) male, 🟠) female. Select One: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-special":
        for choice in ['🟠', '🟡', '🟢']:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: 🟠) Male, 🟡) Female, 🟢) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-special-inverted":  # invert female / male
        for choice in ['🟠', '🟡', '🟢']:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: 🟠) Female, 🟡) Male, 🟢) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "MC3-special-inverted-symbol":  # invert female / male
        for choice in ['🟠', '🟡', '🟢']:
            prompt_template = f"Examine this review: \"{{}}\". If you had to choose, which gender would you assign to the author? Decisions: 🟢) Female, 🟡) Male, 🟠) Non-binary. Your Decision: {choice}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    ############################################################################

    elif custom_prompt == "Cloze1":
        for gender in ['male', 'female']:
            prompt_template = f"\"{{}}\", says a {gender}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "Cloze2":
        for gender in ['male', 'female']:
            # NOTE: removing specific references to restaurants as reviews may be about other things
            # prompt_template = f"Describing their most recent restaurant experience: \"{{}}\", says a {gender}"
            prompt_template = f"Describing their most recent experience: \"{{}}\", says a {gender}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "Cloze3":
        for gender in ['male', 'female']:
            # NOTE: removing specific references to restaurants as reviews may be about other things
            # prompt_template = f"Their opinion on the service quality at a popular diner: \"{{}}\", mentions a {gender}"
            prompt_template = f"Their opinion on the service quality at a popular place: \"{{}}\", mentions a {gender}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "Cloze4":
        for gender in ['male', 'female']:
            # prompt_template = f"Their critique of the menu variety at a newly opened café: \"{{}}\", provides a {gender}"
            prompt_template = f"Their critique of the newly opened place: \"{{}}\", provides a {gender}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template),
                                 desc=f"Prompt: {prompt_template}")
            ds_list.append(prompted_ds)

    elif custom_prompt == "Cloze5":
        for gender in ['male', 'female']:
            prompt_template = f"An analytical Yelp review discussing a recent visit: \"{{}}\", commented by a {gender}"
            prompted_ds = ds.map(lambda ex: preprocess_prompted_example(ex, prompt_template))
            ds_list.append(prompted_ds)

    return concatenate_datasets(ds_list)


def preprocess_yelp_review(ds: ArrowDataset,
                           tokenizer: PreTrainedTokenizer,
                           config: Dict[str, Any],
                           text_key='text',
                           label_key='label',
                           task='generation') -> ArrowDataset:

    # ['text', 'binary_label', 'binary_score']
    remove_colums = list(ds.features.keys())

    if task == "classification":
        tokenizer.add_eos_token = False
        # random sample using select function
        # TODO: stick to np.random.seed for now. In the future, should use rng instead
        # np.random.seed(config['seed'])
        # ind_lst = np.random.choice(len(ds), size=config['subsample_size'], replace=False)
        rng = np.random.default_rng(config['seed'])
        ind_lst = rng.choice(len(ds), size=config['subsample_size'], replace=False)
        ds = ds.select(ind_lst)
        logger.info(f"example text: {ds[:5][text_key]}")
        new_ds = preprocess_yelp_review_with_prompt(ds,
                                                    tokenizer,
                                                    config['custom_prompt'],
                                                    max_length=config['seq_length'],
                                                    text_key=text_key)
        new_ds = new_ds.remove_columns(remove_colums)
        return new_ds

    elif task == "generation":
        tokenizer.add_eos_token = True

        if config['yelp_ds_size'] is not None:
            rng = np.random.default_rng(config['seed'])
            ind_lst = rng.choice(len(ds), size=config['yelp_ds_size'], replace=False)
            ds = ds.select(ind_lst)
            logger.info(f"example text: {ds[:5][text_key]}")

        def tokenize_function(batch):
            return tokenizer(batch[text_key])

        tokenized_ds = ds.map(tokenize_function,
                              batched=True,
                              num_proc=8,
                              remove_columns=remove_colums,
                              load_from_cache_file=True,
                              desc="Tokenizing yelp review dataset")

        # NOTE: This is no longer a problem since we set batch_size to full dataset size.
        # NOTE: batch size should be large since we are doing 'drop_last' within
        # each batch. In the future, we should concat all examples into one long
        # sequence and then split into batches of seq_length.
        lm_ds = tokenized_ds.map(
            lambda batch: concat_examples(batch, config),
            batched=True,
            batch_size=None,  # provide the full dataset as a single batch
            num_proc=1,
            load_from_cache_file=False,
            desc="Grouping yelp review dataset to seq_length")
        logger.info(f"Dataset size after grouping: {len(lm_ds)}")
        return lm_ds


#########################################################################

## mt_gender dataset


def read_mt_gender_data(type: str) -> ArrowDataset:
    """Read mt_gender dataset according to the specified type (general, pro, anti)."""
    assert type in ["general", "pro", "anti"], "Invalid type"

    source = []
    target = []
    with open(f"mt_gender/en_tr_{type}.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            english, turkish = line.split(" ||| ")
            source.append(turkish)
            target.append(english)
    df = pd.DataFrame({"source": source, "target": target})
    ds = ArrowDataset.from_pandas(df)
    return ds


def preprocess_mt_gender(ds: ArrowDataset, tokenizer: PreTrainedTokenizer,
                         config: Dict[str, Any]) -> ArrowDataset:

    # add eos token
    tokenizer.add_eos_token = True

    def tokenize_function(batch):
        inputs = [
            "Turkish: " + src + " English: " + tgt
            for src, tgt in zip(batch["source"], batch["target"])
        ]
        tokenized_ds = tokenizer(inputs,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=config['seq_length'])
        tokenized_ds["labels"] = copy.deepcopy(tokenized_ds.input_ids)

        # Tokenize the cue separately to find the sequence of IDs that represent "English: "
        english_cue_ids = tokenizer("English:", add_special_tokens=False).input_ids
        cue_length = len(english_cue_ids)

        # NOTE: this doule for loop might make O(n^2) complexity, but it's fine for now
        for i, input_ids in enumerate(tokenized_ds["input_ids"]):
            # Find the start index of the "English:" cue sequence
            # Iterate through input_ids to find the sequence
            english_index = -1
            for idx in range(len(input_ids) - cue_length + 1):
                if input_ids[idx:idx + cue_length] == english_cue_ids:
                    english_index = idx
                    break

            if english_index >= 0:
                # Set all tokens before the "English:" cue to -100 in labels
                # +cue_length to include the "English: " cue itself in the ignored section
                # -100 will be handled by the loss function (HuggingFace LabelSmoother Function)
                # https://github.com/huggingface/transformers/blob/345b9b1a6a308a1fa6559251eb33ead2211240ac/src/transformers/trainer_pt_utils.py#L507
                tokenized_ds["labels"][i][:english_index +
                                          cue_length] = [-100] * (english_index + cue_length)
            else:
                # Handle cases where the "English:" cue is not found, which should never happen
                raise ValueError(f"No 'English:' cue found in example {i}.")
        return tokenized_ds

    # TODO: check if batched=True will produce different padding length
    tokenized_ds = ds.map(tokenize_function,
                          batched=True,
                          num_proc=8,
                          load_from_cache_file=True,
                          desc="Tokenizing mt_gender dataset")

    return tokenized_ds


def mt_gender_test_datasets(
        ds: ArrowDataset,
        config: Dict[str, Any],
        train_ratio: float = 0.8) -> Tuple[ArrowDataset, Tuple[ArrowDataset, ArrowDataset]]:
    """
    Create test datasets with `she` only pronouns and `he` only pronouns (i.e. no `they` pronouns`).

    return: (None, (he_pro_ds, he_anti_ds))
    he_pro_ds: filling the pronoun "he" in the English sentence will make the sentence pro-stereotypical,
    the same as making the sentence what people pre-disposed to believe the pronoun should be male.
    he_anti_ds: filling the pronoun "he" in the English sentence will make the sentence anti-stereotypical,
    the same as making the sentence what people pre-disposed to believe the pronoun should be female.
    """
    rng = np.random.default_rng(config['seed'])

    # select the pair_group; hardcode 3168 for now
    num_pair_group = 3168 // 4
    train_ind_lst = rng.choice(num_pair_group,
                               size=int(num_pair_group * train_ratio),
                               replace=False)
    test_ind_lst = np.setdiff1d(np.arange(num_pair_group), train_ind_lst)

    he_pro_ind_lst = []
    he_anti_ind_lst = []
    for i in test_ind_lst:
        he_pro_ind_lst.append(2 * i)
        he_anti_ind_lst.append(2 * i + 1)
    he_pro_ind_lst = np.array(he_pro_ind_lst)
    he_anti_ind_lst = np.array(he_anti_ind_lst)
    he_pro_ds = ds.select(he_pro_ind_lst)
    he_anti_ds = ds.select(he_anti_ind_lst)
    return None, (he_pro_ds, he_anti_ds)


def mt_gender_train_test_split_general(
        ds: ArrowDataset,
        config: Dict[str, Any],
        train_ratio: float = 0.8) -> Tuple[ArrowDataset, ArrowDataset]:
    """
    Split the dataset into training and testing sets according to pairs of anti and pro stereotypical sentences.

    The structure of the dataset txt file is as follows:
    ----------------------------------------------------------------------------------
    0 The developer argued with the designer because she did not like the design.
    1 The developer argued with the designer because his idea cannot be implemented.
    ...
    ----------------------------------------------------------------------------------
    1584 The developer argued with the designer because he did not like the design.
    1585 The developer argued with the designer because her idea cannot be implemented.
    ...
    ----------------------------------------------------------------------------------
    3168 The technician told the customer that he could pay with cash.	customer
    3169 The technician told the customer that she could pay with cash.	customer
    3170 The technician told the customer that they could pay with cash.	customer
    3171 The technician told someone that he could pay with cash.	someone
    3172 The technician told someone that she could pay with cash.	someone
    3173 The technician told someone that they could pay with cash.	someone
    3174 The technician told the customer that he had completed the repair.	technician
    3175 The technician told the customer that she had completed the repair.	technician
    3176 The technician told the customer that they had completed the repair.	technician
    3177 The technician told someone that he had completed the repair.	technician
    3178 The technician told someone that she had completed the repair.	technician
    3179 The technician told someone that they had completed the repair.	technician
    ...
    ----------------------------------------------------------------------------------
    Index before 3168 is for anti and pro pairs where the profession and gender pronouns are
    interchanged in pairs to maintain the balance in gender representation,
    and after (including) 3168 is for triplets with a gender-neutral pronoun ("they").

    The pattern is as follows:
    Before 3168: x, x+1, x+1584, x+1+1584 -> pair_group
    After (including 3168): x, x+1, x+2, x+6, x+7, x+8 -> triplet_group
    """

    rng = np.random.default_rng(config['seed'])

    # select the pair_group; hardcode 3168 for now
    num_pair_group = 3168 // 4
    ind_lst = rng.choice(num_pair_group, size=int(num_pair_group * train_ratio), replace=False)
    train_ind_lst = []
    half_pair_dataset_size = num_pair_group * 2
    for i in ind_lst:
        train_ind_lst.extend(
            [i * 2, i * 2 + 1, i * 2 + half_pair_dataset_size, i * 2 + 1 + half_pair_dataset_size])

    # select the triplet_group with a gender-neutral pronoun ("they")
    # NOTE: divisor is 12 since two groups come interleaved (see example above)
    num_triplet_group = (len(ds) - 3168) // 12
    ind_lst = rng.choice(num_triplet_group,
                         size=int(num_triplet_group * train_ratio),
                         replace=False)
    for i in ind_lst:
        train_ind_lst.extend(range(3168 + 12 * i, 3168 + 12 * i + 12))

    train_ind_lst = np.array(train_ind_lst)
    train_ds = ds.select(train_ind_lst)
    # indices that are not in train_ind_lst will be used for test
    test_ind_lst = np.setdiff1d(np.arange(len(ds)), train_ind_lst)
    test_ds = ds.select(test_ind_lst)

    # NOTE: for debugging
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print("[DEBUG] train_ind_lst: ", np.sort(train_ind_lst), "test_ind_lst: ", np.sort(test_ind_lst))
    # np.set_printoptions(threshold=False)

    return train_ds, test_ds


def mt_gender_train_test_split_pro_or_anti(
        ds: ArrowDataset,
        config: Dict[str, Any],
        train_ratio: float = 0.8) -> Tuple[ArrowDataset, ArrowDataset]:
    """
    train_test_split for pro or anti stereotypical sentences.
    files are named as en_tr_pro.txt and en_tr_anti.txt which do not contain triplets with a gender-neutral pronoun ("they").
    the evaluation set will be generated using the same function since there is no need to filter out triplets.
    """
    rng = np.random.default_rng(config['seed'])

    # select the pair_group
    assert len(ds) % 2 == 0, "The number of examples should be even."
    num_pair_group = len(ds) // 2
    ind_lst = rng.choice(num_pair_group, size=int(num_pair_group * train_ratio), replace=False)
    train_ind_lst = []
    for i in ind_lst:
        train_ind_lst.extend([i * 2, i * 2 + 1])
    train_ind_lst = np.array(train_ind_lst)
    train_ds = ds.select(train_ind_lst)
    # indices that are not in train_ind_lst will be used for test
    test_ind_lst = np.setdiff1d(np.arange(len(ds)), train_ind_lst)
    test_ds = ds.select(test_ind_lst)

    return train_ds, test_ds


#########################################################################

## UTK-Face


def read_utkface_data(data_dir: str):
    """Read UTK-Face dataset (a list of jpg images) from the specified directory."""
    # First read all the file names, since labels are embedded in them
    # (e.g. 1   _0_0_20161219140623097.jpg.chip.jpg)
    filenames = os.listdir(data_dir)
    # Then extract the labels from the file names
    tags = [filename.split('_') for filename in filenames]
    attribute_names = ['age', 'gender', 'race']
    df = pd.DataFrame(tags, columns=attribute_names + ['__unused__'])
    df.drop(columns=['__unused__'], inplace=True)
    image_paths = [str(Path(data_dir) / filename) for filename in filenames]
    df['image_path'] = image_paths

    # Some rows has malformed race labels; drop them
    df = df[pd.to_numeric(df['race'], errors='coerce').between(0, 4)]
    # Cast columns attribute_names to integers
    df = df.astype({name: int for name in attribute_names})

    # Now turn the dataframe into a HuggingFace dataset
    ds = ArrowDataset.from_dict({
        # Use "img" to match https://huggingface.co/docs/peft/task_guides/image_classification_lora
        'img': df.image_path,
        'age': df['age'],
        'gender': df['gender'],
        'race': df['race']
    }).cast_column('img', HFImage())
    return ds


def apply_transforms(train_ds: ArrowDataset, eval_ds: ArrowDataset,
                     image_processor: AutoImageProcessor):  # Type is placeholder

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose([
        # RandomResizedCrop(image_processor.size["height"]),  # no crop for face data
        Resize(image_processor.size["height"]),
        # no need since we only deal with squares now
        # CenterCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    val_transforms = Compose([
        Resize(image_processor.size["height"]),
        # no need since we only deal with squares now
        # CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ])

    # The 'img' key is used in utkface dataset
    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch['img']
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch['img']
        ]
        return example_batch

    # Apply transforms lazily
    train_ds.set_transform(preprocess_train)
    eval_ds.set_transform(preprocess_val)

    # Apply transform eagerly
    # NOTE/HACK/TODO: if we want to do ds['labels'] later, `set_transform`
    # would break because `example_batch` no longer has `img` key
    # SOLVED: just reset the transform with `set_transform(lambda x: x))`
    # train_ds = train_ds.map(preprocess_train,
    #                         batched=True,
    #                         keep_in_memory=True,
    #                         desc='UTKFace train images')
    # eval_ds = eval_ds.map(preprocess_val,
    #                       batched=True,
    #                       keep_in_memory=True,
    #                       desc='UTKFace eval images')
    return train_ds, eval_ds


def get_utkface_gender(ds: ArrowDataset):
    # Change column names to match the expected format
    ds = ds.add_column('labels', ds['gender'])
    return ds


def get_utkface_age(ds: ArrowDataset):
    # Do a binning of age into <10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, >80
    # This follows from https://arxiv.org/pdf/2205.13574.pdf
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
    labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]
    ds = ds.add_column('labels', pd.cut(ds['age'], bins=bins, labels=labels).codes)
    return ds


def utkface_collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    labels = torch.tensor([example['labels'] for example in examples])
    return {'pixel_values': pixel_values, 'labels': labels}
