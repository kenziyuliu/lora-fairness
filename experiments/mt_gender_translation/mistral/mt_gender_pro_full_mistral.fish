for epoch in 1 2 3 4
    for lr in 0.000003 0.000005 0.00001 0.00003
        accelerate launch \
            --main_process_port 6068 --config_file configs/fsdp_config.yml \
            main.py \
            --config configs/mistral.yml --dataset mt_gender_translation_pro \
            --finetune full --lr $lr --epochs $epoch \
            --run_name mt_gender_translation_pro-mistral-7b-full-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end