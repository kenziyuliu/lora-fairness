from transformers import AutoConfig, AutoTokenizer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer, set_seed

# TODO

def prepare_tokenizer(path_or_name: str, special_tokens: list[str] = None) -> PreTrainedTokenizer:
    # always using a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        logger.info(f'Added control tokens: {tokenizer.additional_special_tokens} to tokenizer '
                    f'with ids {tokenizer.additional_special_tokens_ids}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # avoid issue with position embeddings for prompts in conditional generation
    tokenizer.aligned_prefix = special_tokens[0] if special_tokens else None
    tokenizer.misaligned_prefix = special_tokens[1] if special_tokens else None
    return tokenizer
