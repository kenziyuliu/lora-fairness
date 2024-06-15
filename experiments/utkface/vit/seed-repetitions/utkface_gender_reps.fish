
for seed in 13 1337 42 666 888

    # Best by acc:
    # utkface_gender-vit_base-full-epoch2-lr0.00003
    # utkface_gender-vit_base-lora-epoch2-lr0.0003
    set full_epoch 2
    set full_lr 0.00003

    set lora_epoch 2
    set lora_lr 0.0003

    accelerate launch \
        --main_process_port 29506 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/vit_base.yml --dataset utkface_gender \
        --run_name utkface_gender-vit_base-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    accelerate launch \
        --main_process_port 29507 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/vit_base.yml --dataset utkface_gender \
        --run_name utkface_gender-vit_base-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
        --save_strategy no --wandb
end
