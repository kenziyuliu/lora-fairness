# base model
# set subsample_size 10000

# for prompt in Cloze1 Cloze2 Cloze3 Cloze4 YN2-special-inverted-symbol MC1-special-inverted-symbol MC3-special-inverted-symbol
#     accelerate launch --main_process_port 6060 --config_file configs/deepspeed_inference_config.yml main.py \
#         --config configs/llama2.yml --dataset yelp_review_classification --finetune lora \
#         --subsample_size $subsample_size --custom_prompt $prompt --eval_bs_per_gpu 128 \
#         --eval outputs/yelp_review_add_eos-llama2-7b-lora-epoch4-lr0.0005 \
#         --run_name eval-yelp_review_add_eos-prompt$prompt-$subsample_size-llama2-7b-lora-epoch4-lr0.0005 \
#         --inference_ds
# end

# chat model
# set subsample_size 50000

# for prompt in YN1 YN2 YN2-inverted YN3 YN4 YN5 YN1-numeric YN2-numeric YN3-numeric YN4-numeric YN5-numeric YN1-numeric-inverted YN3-numeric-inverted MC1 MC2 MC3 MC3-inverted MC3-inverted-symbol MC4 MC5 MC1-numeric MC2-numeric MC3-numeric MC4-numeric MC5-numeric MC1-numeric-inverted MC1-numeric-inverted-symbol MC3-numeric-inverted YN1-special YN1-special-inverted YN1-special-inverted-symbol YN2-special YN2-special-inverted YN2-special-inverted-symbol MC1-special MC1-special-inverted MC1-special-inverted-symbol MC3-special MC3-special-inverted MC3-special-inverted-symbol Cloze1 Cloze2 Cloze3 Cloze4 Cloze5
#     accelerate launch --main_process_port 6060 --config_file configs/deepspeed_inference_config.yml main.py \
#         --config configs/llama2.yml --dataset yelp_review_classification --finetune lora \
#         --subsample_size $subsample_size --custom_prompt $prompt --eval_bs_per_gpu 128 \
#         --model_base meta-llama/Llama-2-7b-chat-hf --tokenizer meta-llama/Llama-2-7b-chat-hf \
#         --eval outputs/yelp_review_add_eos50000-llama2-7b-chat-lora-epoch4-lr0.001 \
#         --run_name eval-yelp_review_add_eos50000-prompt$prompt-$subsample_size-llama2-7b-chat-lora-epoch4-lr0.001 \
#         --inference_ds
# end

# mistral
set subsample_size 50000

for prompt in YN1-numeric-inverted YN3-numeric-inverted MC1 MC2 MC3 MC3-inverted
# for prompt in YN1-numeric-inverted YN3-numeric-inverted MC1 MC2 MC3 MC3-inverted MC3-inverted-symbol MC4 MC5 MC1-numeric MC2-numeric MC3-numeric MC4-numeric MC5-numeric MC1-numeric-inverted MC1-numeric-inverted-symbol MC3-numeric-inverted YN1-special YN1-special-inverted YN1-special-inverted-symbol YN2-special YN2-special-inverted YN2-special-inverted-symbol MC1-special MC1-special-inverted MC1-special-inverted-symbol MC3-special MC3-special-inverted MC3-special-inverted-symbol Cloze1 Cloze2 Cloze3 Cloze4 Cloze5
    accelerate launch --main_process_port 5050 --config_file configs/deepspeed_inference_config.yml main.py \
        --config configs/mistral.yml --dataset yelp_review_classification --finetune lora \
        --subsample_size $subsample_size --custom_prompt $prompt --eval_bs_per_gpu 128 \
        --eval outputs/yelp_review_add_eos50000-mistral-7b-lora-rank256-epoch12-lr0.001 \
        --run_name eval-yelp_review_add_eos50000-prompt$prompt-$subsample_size-mistral-7b-lora-rank256-epoch12-lr0.001 \
        --inference_ds
end

# mistral chat
# set subsample_size 50000

# for prompt in YN1 YN2 YN2-inverted YN3 YN4 YN5 YN1-numeric YN2-numeric YN3-numeric YN4-numeric YN5-numeric YN1-numeric-inverted YN3-numeric-inverted MC1 MC2 MC3 MC3-inverted MC3-inverted-symbol MC4 MC5 MC1-numeric MC2-numeric MC3-numeric MC4-numeric MC5-numeric MC1-numeric-inverted MC1-numeric-inverted-symbol MC3-numeric-inverted YN1-special YN1-special-inverted YN1-special-inverted-symbol YN2-special YN2-special-inverted YN2-special-inverted-symbol MC1-special MC1-special-inverted MC1-special-inverted-symbol MC3-special MC3-special-inverted MC3-special-inverted-symbol Cloze1 Cloze2 Cloze3 Cloze4 Cloze5
#     accelerate launch --main_process_port 5050 --config_file configs/deepspeed_inference_config.yml main.py \
#         --config configs/mistral.yml --dataset yelp_review_classification --finetune lora \
#         --subsample_size $subsample_size --custom_prompt $prompt --eval_bs_per_gpu 32 \
#         --model_base mistralai/Mistral-7B-Instruct-v0.2 --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
#         --eval outputs/yelp_review_add_eos50000-mistral-7b-chat-lora-rank64-epoch12-lr0.001 \
#         --run_name eval-yelp_review_add_eos50000-prompt$prompt-$subsample_size-mistral-7b-chat-lora-rank64-epoch12-lr0.001 \
#         --inference_ds
# end