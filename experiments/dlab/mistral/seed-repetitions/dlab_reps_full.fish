
for seed in 13 1337 42 666 888

    set subset age
    # Best by acc:
    # dlab-age-mistral-7b-full-epoch8-lr0.0001
    # dlab-age-mistral-7b-lora-epoch8-lr0.0003

    set full_epoch 8
    set full_lr 0.0001

    accelerate launch \
        --main_process_port 20200 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-mistral-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    # set lora_epoch 8
    # set lora_lr 0.0003

    # accelerate launch \
    #     --main_process_port 20200 --config_file configs/deepspeed_config.yml \
    #     main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
    #     --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-mistral-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
    #     --save_strategy no --wandb

end


for seed in 13 1337 42 666 888

    set subset gender
    # Best by acc:
    # dlab-gender-mistral-7b-full-epoch4-lr0.00001
    # dlab-gender-mistral-7b-lora-epoch2-lr0.0003
    
    set full_epoch 4
    set full_lr 0.00001

    accelerate launch \
        --main_process_port 3030 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-mistral-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    # set lora_epoch 2
    # set lora_lr 0.0003

    # accelerate launch \
    #     --main_process_port 3030 --config_file configs/deepspeed_config.yml \
    #     main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
    #     --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-mistral-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
    #     --save_strategy no --wandb

end


for seed in 13 1337 42 666 888

    set subset race
    # Best by acc:
    # dlab-race-mistral-7b-full-epoch4-lr0.00005
    # dlab-race-mistral-7b-lora-epoch4-lr0.0001

    set full_epoch 4
    set full_lr 0.00005

    accelerate launch \
        --main_process_port 5050 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-mistral-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    # set lora_epoch 4
    # set lora_lr 0.0001

    # accelerate launch \
    #     --main_process_port 5050 --config_file configs/deepspeed_config.yml \
    #     main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
    #     --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-mistral-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
    #     --save_strategy no --wandb

end


for seed in 13 1337 42 666 888

    set subset religion
    # Best by acc:
    # dlab-religion-mistral-7b-full-epoch2-lr0.00001
    # dlab-religion-mistral-7b-lora-epoch4-lr0.0003

    set full_epoch 2
    set full_lr 0.00001

    accelerate launch \
        --main_process_port 6060 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-mistral-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    # set lora_epoch 4
    # set lora_lr 0.0003

    # accelerate launch \
    #     --main_process_port 6060 --config_file configs/deepspeed_config.yml \
    #     main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
    #     --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-mistral-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
    #     --save_strategy no --wandb

end


for seed in 13 1337 42 666 888

    set subset sexuality
    # Best by acc:
    # dlab-sexuality-mistral-7b-full-epoch3-lr0.00001
    # dlab-sexuality-mistral-7b-lora-epoch12-lr0.0001

    set full_epoch 3
    set full_lr 0.00001

    accelerate launch \
        --main_process_port 7070 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-mistral-7b-full-epoch$full_epoch-lr$full_lr-seed$seed \
        --save_strategy no --wandb

    # set lora_epoch 12
    # set lora_lr 0.0001

    # accelerate launch \
    #     --main_process_port 7070 --config_file configs/deepspeed_config.yml \
    #     main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
    #     --config configs/mistral.yml --dataset dlab_hatespeech'_'$subset \
    #     --run_name dlab-$subset-mistral-7b-lora-epoch$lora_epoch-lr$lora_lr-seed$seed \
    #     --save_strategy no --wandb

end