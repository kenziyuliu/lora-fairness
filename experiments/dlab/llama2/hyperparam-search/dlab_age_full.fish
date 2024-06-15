
# Full finetuning
set subset age

for epoch in 1 2 3 4 6 8
    for lr in 0.00001 0.00005 0.0001 0.0003
        accelerate launch \
            --main_process_port 29501 --config_file configs/fsdp_config.yml \
            main.py \
            --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
            --finetune full --lr $lr --epochs $epoch \
            --run_name dlab-$subset-llama2-7b-full-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end
