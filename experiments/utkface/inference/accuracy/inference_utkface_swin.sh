################################### UTKFace gender on swin
# full: epoch6-lr0.000003
# lora: epoch2-lr0.001

# Best
for finetune in lora full
    for subset in age gender
        for seed in 13 1337 42 666 888
            accelerate launch --main_process_port 3030 --config_file configs/deepspeed_inference_config.yml main.py \
                --inference_ds --config configs/swinv2_large.yml \
                --eval_bs_per_gpu 128 --seed $seed\
                --dataset utkface_$subset --finetune $finetune \
                --eval outputs/utkface_$subset-swinv2_large-$finetune-best-seed$seed \
                --run_name eval-utkface_$subset-swinv2_large-$finetune-best-seed$seed
        end
    end
end


# LoRA ranks
set config_tag "epoch2-lr0.001"

for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    accelerate launch --main_process_port 7070 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/swinv2_large.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_gender --finetune lora \
        --eval outputs/utkface_gender-swinv2_large-lora-$config_tag-rank$rank \
        --run_name eval-utkface_gender-swinv2_large-lora-$config_tag-rank$rank
end

################################### UTKFace age on swin
# full: epoch2-lr0.0001
# lora: epoch8-lr0.0003

# Best
for finetune in lora full
    accelerate launch --main_process_port 6060 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/swinv2_large.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_age --finetune $finetune \
        --eval outputs/utkface_age-swinv2_large-$finetune-best \
        --run_name eval-utkface_age-swinv2_large-$finetune-best
end


# LoRA ranks
set config_tag "epoch8-lr0.0003"

for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    accelerate launch --main_process_port 3030 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/swinv2_large.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_age --finetune lora \
        --eval outputs/utkface_age-swinv2_large-lora-$config_tag-rank$rank \
        --run_name eval-utkface_age-swinv2_large-lora-$config_tag-rank$rank
end

################## Inference on repeated runs, UTKFace


set datasets utkface_age utkface_gender

set full_tags utkface_age-swinv2_large-full-epoch2-lr0.0001 \
              utkface_gender-swinv2_large-full-epoch6-lr0.000003

set lora_tags utkface_age-swinv2_large-lora-epoch8-lr0.0003 \
              utkface_gender-swinv2_large-lora-epoch2-lr0.001


for seed in 13 1337 42 666 888

    # Use python to zip the two lists together
    for pair in (python -c "import sys; datasets = sys.argv[1].split(','); full_tags = sys.argv[2].split(','); print('\n'.join(f'{a} {b}' for a, b in zip(datasets, full_tags)))" (string join , $datasets) (string join , $full_tags))

        set -l subset (echo $pair | cut -d ' ' -f1)
        set -l tag (echo $pair | cut -d ' ' -f2)

        accelerate launch --main_process_port 8080 \
            --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/swinv2_large.yml \
            --seed $seed \
            --eval_bs_per_gpu 128 \
            --dataset $subset --finetune full \
            --eval outputs/$tag-seed$seed \
            --run_name eval-$tag-seed$seed
    end

    # LoRA
    # Use python to zip the two lists together
    for pair in (python -c "import sys; datasets = sys.argv[1].split(','); lora_tags = sys.argv[2].split(','); print('\n'.join(f'{a} {b}' for a, b in zip(datasets, lora_tags)))" (string join , $datasets) (string join , $lora_tags))

        set -l subset (echo $pair | cut -d ' ' -f1)
        set -l tag (echo $pair | cut -d ' ' -f2)

        echo $subset $tag
        accelerate launch --main_process_port 9090 \
            --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/swinv2_large.yml \
            --seed $seed \
            --eval_bs_per_gpu 128 \
            --dataset $subset --finetune lora \
            --eval outputs/$tag-seed$seed \
            --run_name eval-$tag-seed$seed
    end
end