set ds_size 50000

for epoch in 1 2 3 4 6 8 12
    for lr in 0.0001 0.0003 0.0005 0.001 0.003
        accelerate launch \
            --main_process_port 8088 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/mistral.yml --dataset yelp_review \
            --yelp_ds_size $ds_size \
            --finetune lora --lr $lr --epochs $epoch \
            --run_name yelp_review_add_eos$ds_size-mistral-7b-lora-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end