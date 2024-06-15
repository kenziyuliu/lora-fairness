set ds_size 50000
set epoch 4

for rank in 16 32 64 128
    for lr in 0.0001 0.0003 0.0005 0.001 0.003
        accelerate launch \
            --main_process_port 8088 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/llama2.yml --dataset yelp_review \
            --model_base meta-llama/Llama-2-7b-chat-hf --tokenizer meta-llama/Llama-2-7b-chat-hf \
            --yelp_ds_size $ds_size --lora_rank $rank \
            --finetune lora --lr $lr --epochs $epoch \
            --run_name yelp_review_add_eos$ds_size-llama2-7b-chat-lora-rank$rank-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end