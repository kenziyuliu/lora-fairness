set prefix /lfs/mercury1/0/pura/
export HF_HOME=$prefix
export WANDB_CACHE_DIR={$prefix}.cache
export WANDB_CONFIG_DIR=$prefix
export TORCH_HOME=$prefix

for seed in 888 #13 1337 42 666 888

    for shadow_seed in 2044 776 1885 2094 55 2142 1895 2157 762 1390 1481 472 1712 2008 1719 101

        accelerate launch --config_file configs/deepspeed_config.yml \
            --main_process_port 15625 main.py --lr 0.001 --seed $seed --shadow_seed $shadow_seed \
            --config configs/swinv2_large.yml --dataset utkface_gender --finetune lora --lira \
            --epochs 3 --run_name lira_utkface_gender_swin-seed{$seed}_shadow$shadow_seed --wandb \
            --utkface_dir /lfs/local/0/kzliu/lora-eval/lora-eval/data/UTKFace

        accelerate launch --config_file configs/deepspeed_config.yml \
            --main_process_port 15625 main.py --lr 0.0003 --seed $seed --shadow_seed $shadow_seed \
            --config configs/vit_base.yml --dataset utkface_gender --finetune lora --lira \
            --epochs 3 --run_name lira_utkface_gender_vit-seed{$seed}_shadow$shadow_seed --wandb \
            --utkface_dir /lfs/local/0/kzliu/lora-eval/lora-eval/data/UTKFace

    end

end