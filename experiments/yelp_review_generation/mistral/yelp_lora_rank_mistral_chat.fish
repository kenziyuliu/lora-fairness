set ds_size 50000
set epoch 12

for rank in 32 64 128 256
    for lr in 0.001
        accelerate launch \
            --main_process_port 7070 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/mistral.yml --dataset yelp_review \
            --model_base mistralai/Mistral-7B-Instruct-v0.2 --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
            --yelp_ds_size $ds_size --lora_rank $rank \
            --finetune lora --lr $lr --epochs $epoch \
            --run_name yelp_review_add_eos$ds_size-mistral-7b-chat-lora-rank$rank-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end