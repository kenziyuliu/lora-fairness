# DeepSpeed inference using stage 3
# accelerate launch --config_file configs/deepspeed_inference_config.yml main.py \
#     --config configs/llama2.yml --dataset ethos --finetune lora \
#     --eval /lfs/skampere1/0/d1ng/ethos-llama2-7b-lora-8-epoch \
#     --inference_ds
# 10 seconds for ethos
accelerate launch --config_file configs/deepspeed_inference_config.yml main.py \
    --config configs/llama2.yml --dataset ethos --finetune full \
    --eval /lfs/skampere1/0/d1ng/ethos-llama2-7b-full-2-epoch-fp16 \
    --inference_ds
# naive inference using MP (200 seconds for ethos)
# python main.py --config configs/llama2.yml --dataset ethos \
#     --eval /lfs/skampere1/0/d1ng/ethos-llama2-7b-full-2-epoch-fp16 \
#     --finetune full --model_parallel


# set subset gender
# set finetune full
# accelerate launch --config_file configs/deepspeed_inference_config.yml main.py --inference_ds \
#     --config configs/llama2.yml --dataset dlab_hatespeech_$subset --finetune $finetune \
#     --eval outputs/dlab-$subset-llama2-7b-$finetune-best --eval_bs_per_gpu 16


# Best
set subset gender
for finetune in full
    accelerate launch --main_process_port 29501 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/llama2.yml \
        --eval_bs_per_gpu 128 \
        --dataset dlab_hatespeech_$subset --finetune $finetune \
        --eval outputs/dlab-$subset-llama2-7b-$finetune-best \
        --run_name eval-dlab-$subset-llama2-7b-$finetune-best
end

# LoRA match full
set subset religion
accelerate launch --main_process_port 29503 --config_file configs/deepspeed_inference_config.yml main.py \
    --inference_ds --config configs/llama2.yml \
    --eval_bs_per_gpu 128 \
    --dataset dlab_hatespeech_$subset --finetune lora \
    --eval outputs/dlab-$subset-llama2-7b-lora-match-full \
    --run_name eval-dlab-$subset-llama2-7b-lora-match-full


# Rank sweep for LoRA
# set subset gender
# set config_tag "epoch2-lr0.0003"
# set subset religion
# set config_tag "epoch12-lr0.00003"
# set subset race
# set config_tag "epoch6-lr0.0003"
set subset sexuality
set config_tag "epoch4-lr0.0001"

for rank in 0 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
    accelerate launch --main_process_port 29506 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/llama2.yml \
        --eval_bs_per_gpu 128 \
        --dataset dlab_hatespeech_$subset --finetune lora \
        --eval outputs/dlab-$subset-llama2-7b-lora-$config_tag-rank$rank \
        --run_name eval-dlab-$subset-llama2-7b-lora-$config_tag-rank$rank
end



################################### UTKFace gender on ViT

# Best
for finetune in lora full
    accelerate launch --main_process_port 29501 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_gender --finetune $finetune \
        --eval outputs/utkface_gender-vit_base-$finetune-best \
        --run_name eval-utkface_gender-vit_base-$finetune-best
end


# LoRA ranks
set config_tag "epoch2-lr0.0003"

for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    accelerate launch --main_process_port 29506 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_gender --finetune lora \
        --eval outputs/utkface_gender-vit_base-lora-$config_tag-rank$rank \
        --run_name eval-utkface_gender-vit_base-lora-$config_tag-rank$rank
end

# alpha 2x
for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    accelerate launch --main_process_port 29507 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_gender --finetune lora \
        --eval outputs/utkface_gender-vit_base-lora-$config_tag-rank$rank-alpha2x \
        --run_name eval-utkface_gender-vit_base-lora-$config_tag-rank$rank-alpha2x
end




################################### UTKFace age on ViT

# Best
for finetune in lora full
    accelerate launch --main_process_port 29501 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_age --finetune $finetune \
        --eval outputs/utkface_age-vit_base-$finetune-best \
        --run_name eval-utkface_age-vit_base-$finetune-best
end


# LoRA ranks
set config_tag "epoch3-lr0.001"

for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    accelerate launch --main_process_port 29506 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_age --finetune lora \
        --eval outputs/utkface_age-vit_base-lora-$config_tag-rank$rank \
        --run_name eval-utkface_age-vit_base-lora-$config_tag-rank$rank
end

# alpha 2x
set config_tag "epoch3-lr0.001"

for rank in 0 1 2 4 8 16 32 64 128 256 512 768
    accelerate launch --main_process_port 29507 --config_file configs/deepspeed_inference_config.yml main.py \
        --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 128 \
        --dataset utkface_age --finetune lora \
        --eval outputs/utkface_age-vit_base-lora-$config_tag-rank$rank-alpha2x \
        --run_name eval-utkface_age-vit_base-lora-$config_tag-rank$rank-alpha2x
end


################## Inference on repeated runs, DLAB

set subsets gender religion sexuality race

set full_tags dlab-gender-llama2-7b-full-epoch1-lr0.00001 \
              dlab-religion-llama2-7b-full-epoch2-lr0.00001 \
              dlab-sexuality-llama2-7b-full-epoch1-lr0.00001 \
              dlab-race-llama2-7b-full-epoch1-lr0.00001

set lora_tags dlab-gender-llama2-7b-lora-epoch2-lr0.0003 \
              dlab-religion-llama2-7b-lora-epoch6-lr0.0003 \
              dlab-sexuality-llama2-7b-lora-epoch4-lr0.0001 \
              dlab-race-llama2-7b-lora-epoch6-lr0.0003


for seed in 13 1337 42 666 888

    # Use python to zip the two lists together (full_tags and lora_tags)
    for pair in (python -c "import sys; subsets = sys.argv[1].split(','); full_tags = sys.argv[2].split(','); print('\n'.join(f'{a} {b}' for a, b in zip(subsets, full_tags)))" (string join , $subsets) (string join , $full_tags))

        set -l subset (echo $pair | cut -d ' ' -f1)
        set -l tag (echo $pair | cut -d ' ' -f2)

        echo $subset $tag
        accelerate launch --main_process_port 29510 \
            --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/llama2.yml \
            --eval_bs_per_gpu 128 \
            --dataset dlab_hatespeech_$subset --finetune full \
            --eval outputs/$tag-seed$seed \
            --run_name eval-$tag-seed$seed
    end
end

for seed in 13 1337 42 666 888
    # LoRA
    # Use python to zip the two lists together
    for pair in (python -c "import sys; subsets = sys.argv[1].split(','); lora_tags = sys.argv[2].split(','); print('\n'.join(f'{a} {b}' for a, b in zip(subsets, lora_tags)))" (string join , $subsets) (string join , $lora_tags))

        set -l subset (echo $pair | cut -d ' ' -f1)
        set -l tag (echo $pair | cut -d ' ' -f2)

        echo $subset $tag
        accelerate launch --main_process_port 29511 \
            --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/llama2.yml \
            --eval_bs_per_gpu 128 \
            --dataset dlab_hatespeech_$subset --finetune lora \
            --eval outputs/$tag-seed$seed \
            --run_name eval-$tag-seed$seed
    end
end


################## Inference on repeated runs, UTKFace


set datasets utkface_age utkface_gender

set full_tags utkface_age-vit_base-full-epoch3-lr0.0001 \
              utkface_gender-vit_base-full-epoch2-lr0.00003

set lora_tags utkface_age-vit_base-lora-epoch3-lr0.001 \
              utkface_gender-vit_base-lora-epoch2-lr0.0003


for seed in 13 1337 42 666 888

    # Use python to zip the two lists together
    for pair in (python -c "import sys; datasets = sys.argv[1].split(','); full_tags = sys.argv[2].split(','); print('\n'.join(f'{a} {b}' for a, b in zip(datasets, full_tags)))" (string join , $datasets) (string join , $full_tags))

        set -l subset (echo $pair | cut -d ' ' -f1)
        set -l tag (echo $pair | cut -d ' ' -f2)
        echo Evaluating  outputs/$tag-seed$seed

        accelerate launch --main_process_port 29512 \
            --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/vit_base.yml \
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

        echo Evaluating outputs/$tag-seed$seed

        accelerate launch --main_process_port 29513 \
            --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/vit_base.yml \
            --seed $seed \
            --eval_bs_per_gpu 128 \
            --dataset $subset --finetune lora \
            --eval outputs/$tag-seed$seed \
            --run_name eval-$tag-seed$seed
    end
end