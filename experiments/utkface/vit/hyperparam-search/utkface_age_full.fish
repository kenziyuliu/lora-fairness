# Full

for epoch in 1 2 3 4 6 8 12
    for lr in 0.000003 0.00001 0.00003 0.0001 0.0003 0.001
        accelerate launch \
            --main_process_port 29503 --config_file configs/fsdp_config.yml \
            main.py \
            --config configs/vit_base.yml --dataset utkface_age \
            --finetune full --lr $lr --epochs $epoch \
            --run_name utkface_age-vit_base-full-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end