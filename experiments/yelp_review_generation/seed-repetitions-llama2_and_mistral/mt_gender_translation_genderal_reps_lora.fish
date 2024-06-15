for seed in 13 1337 42 666 888

    # Best by sacrebleu:
    # mt_gender_translation_general-mistral-7b-lora-epoch2-lr0.001

    set lora_epoch 2
    set lora_lr 0.001

    accelerate launch \
        --main_process_port 6068 --config_file configs/deepspeed_config.yml \
        main.py \
        --config configs/mistral.yml --dataset mt_gender_translation_general \
        --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --run_name mt_gender_translation_general-mistral-7b-lora-best-seed$seed --wandb \
        --save_strategy no
end

for seed in 13 1337 42 666 888

    # Best by sacrebleu:
    # mt_gender_translation_general-llama2-7b-lora-epoch6-lr0.003

    set lora_epoch 6
    set lora_lr 0.003

    accelerate launch \
        --main_process_port 6068 --config_file configs/deepspeed_config.yml \
        main.py \
        --config configs/llama2.yml --dataset mt_gender_translation_general \
        --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --run_name mt_gender_translation_general-llama2-7b-lora-best-seed$seed --wandb \
        --save_strategy no
end