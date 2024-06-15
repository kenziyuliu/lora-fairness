set ds_size 50000

for epoch in 3 4
    for lr in 0.000003 0.000005 0.00001 0.00003
        accelerate launch \
            --main_process_port 6068 --config_file configs/fsdp_config.yml \
            main.py \
            --config configs/llama2.yml --dataset yelp_review \
            --yelp_ds_size $ds_size \
            --finetune full --lr $lr --epochs $epoch \
            --run_name yelp_review_add_eos$ds_size-llama2-7b-full-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end