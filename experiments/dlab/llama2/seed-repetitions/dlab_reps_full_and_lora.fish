
for seed in 1337 42 666 888

    set subset gender
    # Best by acc:
    # dlab-gender-llama2-7b-full-epoch1-lr0.00001
    # dlab-gender-llama2-7b-lora-epoch2-lr0.0003
    set full_epoch 1
    set full_lr 0.00001

    set lora_epoch 2
    set lora_lr 0.0003

    accelerate launch \
        --main_process_port 6060 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-full-best-seed$seed \
        --save_strategy no --wandb

    accelerate launch \
        --main_process_port 7070 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-lora-best-seed$seed \
        --save_strategy no --wandb

end


for seed in 1337 42 666 888

    set subset race
    # Best by acc:
    # dlab-race-llama2-7b-full-epoch1-lr0.00001
    # dlab-race-llama2-7b-lora-epoch6-lr0.0003

    set full_epoch 1
    set full_lr 0.00001

    accelerate launch \
        --main_process_port 6060 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-full-best-seed$seed \
        --save_strategy no --wandb

    set lora_epoch 6
    set lora_lr 0.0003

    accelerate launch \
        --main_process_port 7070 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-lora-best-seed$seed \
        --save_strategy no --wandb

end


for seed in 1337 42 666 888

    set subset religion
    # Best by acc:
    # dlab-religion-llama2-7b-full-epoch2-lr0.00001
    # dlab-religion-llama2-7b-lora-epoch6-lr0.0003

    # set full_epoch 2
    # set full_lr 0.00001

    accelerate launch \
        --main_process_port 6060 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-full-best-seed$seed \
        --save_strategy no --wandb

    set lora_epoch 6
    set lora_lr 0.0003

    accelerate launch \
        --main_process_port 7070 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-lora-best-seed$seed \
        --save_strategy no --wandb

end


for seed in 1337 42 666 888

    set subset sexuality
    # Best by acc:
    # dlab-sexuality-llama2-7b-full-epoch1-lr0.00001
    # dlab-sexuality-llama2-7b-lora-epoch4-lr0.0001
    set full_epoch 1
    set full_lr 0.00001

    set lora_epoch 4
    set lora_lr 0.0001

    accelerate launch \
        --main_process_port 6060 --config_file configs/fsdp_config.yml \
        main.py --finetune full --lr $full_lr --epochs $full_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-full-best-seed$seed \
        --save_strategy no --wandb

    accelerate launch \
        --main_process_port 7070 --config_file configs/deepspeed_config.yml \
        main.py --finetune lora --lr $lora_lr --epochs $lora_epoch --seed $seed \
        --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
        --run_name dlab-$subset-llama2-7b-lora-best-seed$seed \
        --save_strategy no --wandb

end