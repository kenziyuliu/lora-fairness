huggingface-cli login
set prefix /lfs/ampere1/0/pura/
export HF_HOME=$prefix
export WANDB_CACHE_DIR={$prefix}.cache
export WANDB_CONFIG_DIR=$prefix
export TORCH_HOME=$prefix

for seed in 13 1337 42 #666 888

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/mistral.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset dlab_hatespeech_religion --finetune full \
        --eval outputs/mia_dlab_hatespeech_mistral_full-seed{$seed}/checkpoint-304/ --run_name loss_eval \
        --suffix mistral_{$seed}_target_full

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/mistral.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset dlab_hatespeech_religion --finetune lora \
        --eval outputs/mia_dlab_hatespeech_mistral_lora-seed{$seed}/checkpoint-456/ --run_name loss_eval \
        --suffix mistral_{$seed}_target_lora

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/llama2.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset dlab_hatespeech_religion --finetune full \
        --eval outputs/mia_dlab_hatespeech_llama_full-seed{$seed}/checkpoint-304/ --run_name loss_eval \
        --suffix llama_{$seed}_target_full

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/llama2.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset dlab_hatespeech_religion --finetune lora \
        --eval outputs/mia_dlab_hatespeech_llama_lora-seed{$seed}/checkpoint-456/ --run_name loss_eval \
        --suffix llama_{$seed}_target_lora

    for shadow_seed in 2044 776 1885 2094 55 2142 1895 2157 762 1390 1481 472 1712 2008 1719 101

        accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
            main.py --inference_ds --config configs/mistral.yml --lira --seed $seed \
            --eval_bs_per_gpu 32 --dataset dlab_hatespeech_religion --finetune lora --shadow_seed $shadow_seed \
            --eval outputs/lira_dlab_hatespeech_mistral-seed{$seed}_shadow{$shadow_seed}/checkpoint-228/ --run_name loss_eval \
            --suffix mistral_{$seed}_shadow$shadow_seed

        accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
            main.py --inference_ds --config configs/llama2.yml --lira --seed $seed \
            --eval_bs_per_gpu 32 --dataset dlab_hatespeech_religion --finetune lora --shadow_seed $shadow_seed \
            --eval outputs/lira_dlab_hatespeech_llama-seed{$seed}_shadow{$shadow_seed}/checkpoint-304/ --run_name loss_eval \
            --suffix llama_{$seed}_shadow$shadow_seed

    end
end