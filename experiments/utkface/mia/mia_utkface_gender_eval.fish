for seed in 13 1337 42 666 888

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/swinv2_large.yml \
        --eval_bs_per_gpu 32 --mia --dataset utkface_gender --finetune full \
        --eval outputs/mia_utkface_gender_full_swin-seed$seed/checkpoint-3558 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix swin_full_$seed --outsource

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/swinv2_large.yml \
        --eval_bs_per_gpu 32 --mia --dataset utkface_gender --finetune lora \
        --eval outputs/mia_utkface_gender_lora_swin-seed$seed/checkpoint-2965 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix swin_lora_$seed --outsource

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 32 --mia --dataset utkface_gender --finetune full \
        --eval outputs/mia_utkface_gender_full_vit-seed$seed/checkpoint-1186 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix vit_full_$seed --outsource

    accelerate launch --main_process_port 15625 --config_file configs/ddp_inference_config.yml \
        main.py --inference_ds --config configs/vit_base.yml \
        --eval_bs_per_gpu 32 --mia --dataset utkface_gender --finetune lora \
        --eval outputs/mia_utkface_gender_lora_vit-seed$seed/checkpoint-2372 --run_name loss_eval \
        --utkface_dir /lfs/local/0/pura/UTKFace --suffix vit_lora_$seed --outsource 

end
