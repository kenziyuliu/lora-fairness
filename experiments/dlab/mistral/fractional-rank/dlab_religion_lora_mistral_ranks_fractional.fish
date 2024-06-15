
# LoRA

# The best LoRA run on dlab-religion:
# https://wandb.ai/lora-eval-f23/lora-eval/runs/krbxj6ma?workspace=user-kenziyuliu
set subset religion
set epoch 12
set lr 0.00003

for mode in front back random
    for rank in 0.2 0.4 0.6 0.8
        accelerate launch \
            --main_process_port 29505 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
            --finetune lora --lr $lr --epochs $epoch \
            --lora_rank $rank --lora_frac_mode $mode \
            --run_name dlab-$subset-mistral-7b-lora-epoch$epoch-lr$lr-rank$rank-mode$mode --wandb \
            --save_strategy no
    end
end

