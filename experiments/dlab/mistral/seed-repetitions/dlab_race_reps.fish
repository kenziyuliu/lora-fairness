
for seed in 13 1337 42 666 888

    set subset race
    # Best by acc:
    # dlab-race-mistral-7b-full-epoch4-lr0.00005
    # dlab-race-mistral-7b-lora-epoch4-lr0.0001

    # set full_epoch 4
    # set full_lr 0.00005

    # accelerate launch \
    #     --main_process_port 5050 --config_file configs/fsdp_config.yml \
    #     main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
    #     --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-mistral-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
    #     --save_strategy no --wandb

    set lora_epoch 4
    set lora_lr 0.0001

    accelerate launch \
        --main_process_port 5050 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-mistral-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
        --save_strategy no --wandb

end