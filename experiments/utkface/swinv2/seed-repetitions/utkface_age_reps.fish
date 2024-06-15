
for seed in 13 1337 42 666 888

    # Best by acc:
    # utkface_age-swinv2_large-full-epoch2-lr0.0001
    # utkface_age-swinv2_large-lora-epoch8-lr0.0003
    set full_epoch 2
    set full_lr 0.0001

    set lora_epoch 8
    set lora_lr 0.0003

    accelerate launch \
        --main_process_port 6060 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/swinv2_large.yml --dataset utkface_age \
        --run_name utkface_age-swinv2_large-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    accelerate launch \
        --main_process_port 6060 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/swinv2_large.yml --dataset utkface_age \
        --run_name utkface_age-swinv2_large-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
        --save_strategy no --wandb

end