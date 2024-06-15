huggingface-cli login

for seed in 13 1337 42 666 888

    accelerate launch --main_process_port 15625 --config_file configs/deepspeed_config.yml main.py \
        --finetune full --lr 0.00001 --epochs 2 --config configs/mistral.yml \
        --dataset dlab_hatespeech_religion --wandb \
        --run_name mia_dlab_hatespeech_mistral_full-seed$seed --seed $seed

    accelerate launch --main_process_port 15625 --config_file configs/deepspeed_config.yml main.py \
        --finetune lora --lr 0.0003 --epochs 5 --config configs/mistral.yml \
        --dataset dlab_hatespeech_religion --wandb \
        --run_name mia_dlab_hatespeech_mistral_lora-seed$seed --seed $seed

    accelerate launch --main_process_port 15625 --config_file configs/deepspeed_config.yml main.py \
        --finetune full --lr 0.00001 --epochs 2 --config configs/llama2.yml \
        --dataset dlab_hatespeech_religion --wandb \
        --run_name mia_dlab_hatespeech_llama_full-seed$seed --seed $seed

    accelerate launch --main_process_port 15625 --config_file configs/deepspeed_config.yml main.py \
        --finetune lora --lr 0.0003 --epochs 8 --config configs/llama2.yml \
        --dataset dlab_hatespeech_religion --wandb \
        --run_name mia_dlab_hatespeech_llama_lora-seed$seed --seed $seed

end