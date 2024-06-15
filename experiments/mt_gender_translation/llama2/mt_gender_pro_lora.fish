for epoch in 1 2 3 4 6 8 12
    for lr in 0.0001 0.0003 0.0005 0.001 0.003
        accelerate launch \
            --main_process_port 8088 --config_file configs/deepspeed_config.yml \
            main.py \
            --config configs/llama2.yml --dataset mt_gender_translation_pro \
            --finetune lora --lr $lr --epochs $epoch \
            --run_name mt_gender_translation_pro-llama2-7b-lora-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end