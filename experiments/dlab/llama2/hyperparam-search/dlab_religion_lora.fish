
# LoRA
set subset religion

for epoch in 2 4 6 8 12
    for lr in 0.00001 0.00005 0.0001 0.0003
        accelerate launch \
            --main_process_port 29502 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
            --finetune lora --lr $lr --epochs $epoch \
            --run_name dlab-$subset-llama2-7b-lora-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end
