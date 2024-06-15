
for seed in 13 1337 42 666 888

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/swinv2_large.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset utkface_gender --finetune full \
        --eval outputs/mia_utkface_gender_full_swin-seed{$seed}/checkpoint-3558 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix swin_{$seed}_target_full

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/swinv2_large.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset utkface_gender --finetune lora \
        --eval outputs/mia_utkface_gender_lora_swin-seed{$seed}/checkpoint-2965 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix swin_{$seed}_target_lora 

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/vit_base.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset utkface_gender --finetune full \
        --eval outputs/mia_utkface_gender_full_vit-seed{$seed}/checkpoint-1186 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix vit_{$seed}_target_full

    accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/vit_base.yml --lira --seed $seed \
        --eval_bs_per_gpu 32 --dataset utkface_gender --finetune lora \
        --eval outputs/mia_utkface_gender_lora_vit-seed{$seed}/checkpoint-2372 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix vit_{$seed}_target_lora 

    for shadow_seed in 2044 776 1885 2094 55 2142 1895 2157 762 1390 1481 472 1712 2008 1719 101

        accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
            main.py --inference_ds --config configs/swinv2_large.yml --lira --seed $seed \
            --eval_bs_per_gpu 32  --dataset utkface_gender --finetune lora --shadow_seed $shadow_seed \
            --eval outputs/lira_utkface_gender_swin-seed{$seed}_shadow{$shadow_seed}/checkpoint-891 \
            --run_name loss_eval --utkface_dir /lfs/local/0/pura/UTKFace --suffix swin_{$seed}_shadow$shadow_seed 

        accelerate launch --main_process_port 15627 --config_file configs/ddp_inference_config.yml \
            main.py --inference_ds --config configs/vit_base.yml --lira --seed $seed \
            --eval_bs_per_gpu 32  --dataset utkface_gender --finetune lora --shadow_seed $shadow_seed \
            --eval outputs/lira_utkface_gender_vit-seed{$seed}_shadow{$shadow_seed}/checkpoint-891 \
            --run_name loss_eval --utkface_dir /lfs/local/0/pura/UTKFace --suffix vit_{$seed}_shadow$shadow_seed 

    end
end