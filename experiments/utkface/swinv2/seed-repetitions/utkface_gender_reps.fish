
for seed in 13 1337 42 666 888

    # Best by eval acc:
    # utkface_gender-swinv2_large-full-epoch6-lr0.000003
    # utkface_gender-swinv2_large-lora-epoch2-lr0.001
    set full_epoch 6
    set full_lr 0.000003

    set lora_epoch 2
    set lora_lr 0.001

    accelerate launch \
        --main_process_port 5050 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/swinv2_large.yml --dataset utkface_gender \
        --run_name utkface_gender-swinv2_large-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    accelerate launch \
        --main_process_port 5050 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/swinv2_large.yml --dataset utkface_gender \
        --run_name utkface_gender-swinv2_large-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
        --save_strategy no --wandb
end
