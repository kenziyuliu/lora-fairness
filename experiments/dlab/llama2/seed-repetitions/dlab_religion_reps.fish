
for seed in 13 1337 42 666 888

    set subset religion
    # Best by acc:
    # dlab-religion-llama2-7b-full-epoch2-lr0.00001
    # dlab-religion-llama2-7b-lora-epoch6-lr0.0003

    # set full_epoch 2
    # set full_lr 0.00001

    # accelerate launch \
    #     --main_process_port 29501 --config_file configs/fsdp_config.yml \
    #     main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
    #     --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-llama2-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
    #     --save_strategy no --wandb

    set lora_epoch 6
    set lora_lr 0.0003

    accelerate launch \
        --main_process_port 29502 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
        --save_strategy no --wandb

end