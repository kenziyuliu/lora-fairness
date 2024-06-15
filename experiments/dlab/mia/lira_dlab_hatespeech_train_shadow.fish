huggingface-cli login
set prefix /lfs/ampere1/0/pura/
export HF_HOME=$prefix
export WANDB_CACHE_DIR={$prefix}.cache
export WANDB_CONFIG_DIR=$prefix
export TORCH_HOME=$prefix

for seed in 666 888 #13 1337 42 666 888

    for shadow_seed in 2044 776 1885 2094 55 2142 1895 2157 762 1390 1481 472 1712 2008 1719 101

        accelerate launch --main_process_port 15625 --config_file configs/deepspeed_config.yml main.py \
            --finetune lora --lr 0.0003 --epochs 3 --config configs/mistral.yml \
            --dataset dlab_hatespeech_religion --wandb --shadow_seed $shadow_seed --lira \
            --run_name lira_dlab_hatespeech_mistral-seed{$seed}_shadow$shadow_seed --seed $seed

        accelerate launch --main_process_port 15625 --config_file configs/deepspeed_config.yml main.py \
            --finetune lora --lr 0.0003 --epochs 4 --config configs/llama2.yml \
            --dataset dlab_hatespeech_religion --wandb --shadow_seed $shadow_seed --lira \
            --run_name lira_dlab_hatespeech_llama-seed{$seed}_shadow$shadow_seed --seed $seed

    end
end

