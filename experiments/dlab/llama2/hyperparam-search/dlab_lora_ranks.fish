
# LoRA

# # The best LoRA run on dlab-religion:
# # https://wandb.ai/lora-eval-f23/lora-eval/runs/krbxj6ma?workspace=user-kenziyuliu
# set subset religion
# set epoch 12
# set lr 0.00003

# # The best LoRA run on dlab-religion:
# # https://wandb.ai/lora-eval-f23/lora-eval/runs/tamc9tpq?workspace=user-kenziyuliu
# set subset gender
# set epoch 2
# set lr 0.0003

# # The best LoRA run on dlab-sexuality:
# # https://wandb.ai/lora-eval-f23/lora-eval/runs/oq6h5yly?workspace=user-kenziyuliu
# set subset sexuality
# set epoch 4
# set lr 0.0001

# The best LoRA run on dlab-race:
# https://wandb.ai/lora-eval-f23/lora-eval/runs/0skdvzxi?workspace=user-kenziyuliu
set subset race
set epoch 6
set lr 0.0003

for rank in 0 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
    accelerate launch \
        --main_process_port 29505 --config_file configs/deepspeed_config.yml \
        main.py \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --finetune lora --lr $lr --epochs $epoch \
        --lora_rank $rank \
        --run_name dlab-$subset-llama2-7b-lora-epoch$epoch-lr$lr-rank$rank --wandb \
        --save_strategy no
end
