

#####################################################################
# UTKFace age

# Best LoRA run
# utkface_age-vit_base-lora-epoch3-lr0.001
set epoch 3
set lr 0.001


for mode in front back random
    for rank in 0.25 0.5 0.75
        # NOTE: no need to tune alpha (empirically found not helpful, and original
        # paper did not tune it either)
        # set lora_alpha (math $rank x 2)
        # accelerate launch \
        #     --main_process_port 29505 --config_file configs/deepspeed_config.yml \
        #     main.py \
        #     --config configs/vit_base.yml --dataset utkface_age \
        #     --finetune lora --lr $lr --epochs $epoch \
        #     --lora_rank $rank --lora_alpha $lora_alpha \
        #     --run_name utkface_age-vit_base-lora-epoch$epoch-lr$lr-rank$rank-alpha2x --wandb \
        #     --save_strategy no

        # NOTE the fractional rank
        accelerate launch \
            --main_process_port 29505 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/vit_base.yml --dataset utkface_age \
            --finetune lora --lr $lr --epochs $epoch \
            --lora_rank $rank --lora_frac_mode $mode \
            --run_name utkface_age-vit_base-lora-epoch$epoch-lr$lr-rank$rank-mode$mode --wandb \
            --save_strategy no
    end
end
