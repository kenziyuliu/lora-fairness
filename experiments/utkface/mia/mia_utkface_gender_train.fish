for seed in 13 1337 42 666 888

    accelerate launch --config_file configs/deepspeed_config.yml \
        --main_process_port 15625 main.py --lr 0.000003 --seed $seed \
        --config configs/swinv2_large.yml --dataset utkface_gender --finetune full \
        --epochs 6  --run_name mia_utkface_gender_full_swin-seed$seed --wandb \
        --utkface_dir /lfs/local/0/kzliu/lora-eval/lora-eval/data/UTKFace 

    accelerate launch --config_file configs/deepspeed_config.yml \
        --main_process_port 15625 main.py --lr 0.001 --seed $seed \
        --config configs/swinv2_large.yml --dataset utkface_gender --finetune lora \
        --epochs 6 --run_name mia_utkface_gender_lora_swin-seed$seed --wandb \
        --utkface_dir /lfs/local/0/kzliu/lora-eval/lora-eval/data/UTKFace

    accelerate launch --config_file configs/deepspeed_config.yml \
        --main_process_port 15625 main.py --lr 0.00003 --seed $seed \
        --config configs/vit_base.yml --dataset utkface_gender --finetune full \
        --epochs 2  --run_name mia_utkface_gender_full_vit-seed$seed --wandb \
        --utkface_dir /lfs/local/0/kzliu/lora-eval/lora-eval/data/UTKFace 

    accelerate launch --config_file configs/deepspeed_config.yml \
        --main_process_port 15625 main.py --lr 0.0003 --seed $seed \
        --config configs/vit_base.yml --dataset utkface_gender --finetune lora \
        --epochs 5 --run_name mia_utkface_gender_lora_vit-seed$seed --wandb \
        --utkface_dir /lfs/local/0/kzliu/lora-eval/lora-eval/data/UTKFace

end