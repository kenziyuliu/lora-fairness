################################### UTKFace gender on swin
# full: epoch6-lr0.000003
# lora: epoch2-lr0.001

# Best
for finetune in lora full
    for subset in gender
        for seed in 13 1337 42 666 888
            accelerate launch --main_process_port 3030 --config_file configs/ddp_inference_config.yml main.py \
                --inference_ds --config configs/swinv2_large.yml \
                --eval_bs_per_gpu 128 --seed $seed --is_calibration \
                --dataset utkface_$subset --finetune $finetune \
                --eval outputs/utkface_$subset-swinv2_large-$finetune-best-seed$seed \
                --run_name eval-calibration-utkface-$subset-swinv2_large-$finetune-best-seed$seed
        end
    end
end

################################### UTKFace gender on vit
# Best
for finetune in lora full
    for subset in gender
        for seed in 13 1337 42 666 888
            accelerate launch --main_process_port 3030 --config_file configs/ddp_inference_config.yml main.py \
                --inference_ds --config configs/vit_base.yml \
                --eval_bs_per_gpu 128 --seed $seed --is_calibration \
                --dataset utkface_$subset --finetune $finetune \
                --eval outputs/utkface_$subset-vit_base-$finetune-best-seed$seed \
                --run_name eval-calibration-utkface-$subset-vit_base-$finetune-best-seed$seed
        end
    end
end