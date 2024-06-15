# mistral 7b model
# Best LoRA
for subset in gender race sexuality religion
    for seed in 13 1337 42 666 888
        set finetune lora
        accelerate launch --main_process_port 11000 --config_file configs/ddp_inference_config.yml main.py \
            --inference_ds --config configs/mistral.yml \
            --eval_bs_per_gpu 16 --seed $seed --is_calibration \
            --dataset dlab_hatespeech_$subset --finetune $finetune \
            --eval outputs/dlab-$subset-mistral-7b-$finetune-best-seed$seed \
            --run_name eval-calibration-dlab-$subset-mistral-7b-$finetune-best-seed$seed
    end
end

# Best Full
for subset in gender race sexuality religion
    for seed in 13 1337 42 666 888
        set finetune full
        accelerate launch --main_process_port 11000 --config_file configs/ddp_inference_config.yml main.py \
            --inference_ds --config configs/mistral.yml \
            --eval_bs_per_gpu 16 --seed $seed --is_calibration \
            --dataset dlab_hatespeech_$subset --finetune $finetune \
            --eval outputs/dlab-$subset-mistral-7b-$finetune-best-seed$seed \
            --run_name eval-calibration-dlab-$subset-mistral-7b-$finetune-best-seed$seed
    end
end


# llama2 7b model
# Best LoRA
for subset in gender race sexuality religion
    for seed in 13 1337 42 666 888
        set finetune lora
        accelerate launch --main_process_port 11000 --config_file configs/ddp_inference_config.yml main.py \
            --inference_ds --config configs/llama2.yml \
            --eval_bs_per_gpu 16 --seed $seed --is_calibration \
            --dataset dlab_hatespeech_$subset --finetune $finetune \
            --eval outputs/dlab-$subset-llama2-7b-$finetune-best-seed$seed \
            --run_name eval-calibration-dlab-$subset-llama2-7b-$finetune-best-seed$seed
    end
end

# Best Full
for subset in gender race sexuality religion
    for seed in 13 1337 42 666 888
        set finetune full
        accelerate launch --main_process_port 11000 --config_file configs/ddp_inference_config.yml main.py \
            --inference_ds --config configs/llama2.yml \
            --eval_bs_per_gpu 16 --seed $seed --is_calibration \
            --dataset dlab_hatespeech_$subset --finetune $finetune \
            --eval outputs/dlab-$subset-llama2-7b-$finetune-best-seed$seed \
            --run_name eval-calibration-dlab-$subset-llama2-7b-$finetune-best-seed$seed
    end
end