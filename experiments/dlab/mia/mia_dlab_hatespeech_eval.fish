huggingface-cli login

for seed in 13 1337 42 666 888

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/mistral.yml \
        --eval_bs_per_gpu 32 --mia --dataset dlab_hatespeech_religion --finetune full \
        --run_name loss_eval --suffix mistral_full_$seed --outsource \
        --eval outputs/mia_dlab_hatespeech_mistral_full-seed$seed/checkpoint-304/ 

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/mistral.yml \
        --eval_bs_per_gpu 32 --mia --dataset dlab_hatespeech_religion --finetune lora \
        --run_name loss_eval --suffix mistral_lora_$seed --outsource \
        --eval outputs/mia_dlab_hatespeech_mistral_lora-seed$seed/checkpoint-456/ 

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/llama2.yml \
        --eval_bs_per_gpu 32 --mia --dataset dlab_hatespeech_religion --finetune full \
        --run_name loss_eval --suffix llama_full_$seed --outsource \
        --eval outputs/mia_dlab_hatespeech_llama_full-seed$seed/checkpoint-304/ 
        
    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/llama2.yml \
        --eval_bs_per_gpu 32 --mia --dataset dlab_hatespeech_religion --finetune lora \
        --run_name loss_eval --suffix llama_lora_$seed --outsource \
        --eval outputs/mia_dlab_hatespeech_llama_lora-seed$seed/checkpoint-456/ 

end