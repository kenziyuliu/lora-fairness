for seed in 13 1337 42 666 888

    # Best by sacrebleu:
    # mt_gender_translation_pro-mistral-7b-full-epoch2-lr0.000003

    set full_epoch 2
    set full_lr 0.000003

    accelerate launch \
        --main_process_port 6068 --config_file configs/fsdp_config.yml \
        main.py \
        --config configs/mistral.yml --dataset mt_gender_translation_pro \
        --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --run_name mt_gender_translation_pro-mistral-7b-full-best-seed$seed --wandb \
        --save_strategy no
end

for seed in 13 1337 42 666 888

    # Best by sacrebleu:
    # mt_gender_translation_pro-llama2-7b-full-epoch4-lr0.00001

    set full_epoch 4
    set full_lr 0.00001

    accelerate launch \
        --main_process_port 6068 --config_file configs/fsdp_config.yml \
        main.py \
        --config configs/llama2.yml --dataset mt_gender_translation_pro \
        --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --run_name mt_gender_translation_pro-llama2-7b-full-best-seed$seed --wandb \
        --save_strategy no
end