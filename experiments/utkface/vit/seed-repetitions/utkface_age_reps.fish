
for seed in 13 1337 42 666 888

    # Best by acc:
    # utkface_age-vit_base-full-epoch3-lr0.0001
    # utkface_age-vit_base-lora-epoch3-lr0.001
    set full_epoch 3
    set full_lr 0.0001

    set lora_epoch 3
    set lora_lr 0.001

    accelerate launch \
        --main_process_port 29508 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/vit_base.yml --dataset utkface_age \
        --run_name utkface_age-vit_base-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    accelerate launch \
        --main_process_port 29509 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/vit_base.yml --dataset utkface_age \
        --run_name utkface_age-vit_base-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
        --save_strategy no --wandb

end