

#####################################################################
# UTKFace gender

# Best LoRA run
# utkface_gender-vit_base-lora-epoch2-lr0.0003
set epoch 2
set lr 0.0003

for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    set lora_alpha (math $rank x 2)
    accelerate launch \
        --main_process_port 29505 --config_file configs/deepspeed_config.yml \
        main.py \
        --config configs/vit_base.yml --dataset utkface_gender \
        --finetune lora --lr $lr --epochs $epoch \
        --lora_rank $rank --lora_alpha $lora_alpha \
        --run_name utkface_gender-vit_base-lora-epoch$epoch-lr$lr-rank$rank-alpha2x --wandb \
        --save_strategy no

    accelerate launch \
        --main_process_port 29505 --config_file configs/deepspeed_config.yml \
        main.py \
        --config configs/vit_base.yml --dataset utkface_gender \
        --finetune lora --lr $lr --epochs $epoch \
        --lora_rank $rank \
        --run_name utkface_gender-vit_base-lora-epoch$epoch-lr$lr-rank$rank --wandb \
        --save_strategy no
end

