# # Best LoRA
# for subset in age gender race sexuality religion
#     for seed in 13 1337 42 666 888
#         set finetune lora
#         accelerate launch --main_process_port 11000 --config_file configs/deepspeed_inference_config.yml main.py \
#             --inference_ds --config configs/mistral.yml \
#             --eval_bs_per_gpu 128 --seed $seed \
#             --dataset dlab_hatespeech_$subset --finetune $finetune \
#             --eval outputs/dlab-$subset-mistral-7b-$finetune-best-seed$seed \
#             --run_name eval-dlab-$subset-mistral-7b-$finetune-best-seed$seed
#     end
# end

# Best Full
for subset in age gender race sexuality religion
    for seed in 13 1337 42 666 888
        set finetune full
        accelerate launch --main_process_port 11000 --config_file configs/deepspeed_inference_config.yml main.py \
            --inference_ds --config configs/mistral.yml \
            --eval_bs_per_gpu 128 --seed $seed \
            --dataset dlab_hatespeech_$subset --finetune $finetune \
            --eval outputs/dlab-$subset-mistral-7b-$finetune-best-seed$seed \
            --run_name eval-dlab-$subset-mistral-7b-$finetune-best-seed$seed
    end
end

# # LoRA ranks
# set config_tag "epoch2-lr0.001"

# for rank in 0 1 2 4 8 16 32 64 128 256 512 768
#     accelerate launch --main_process_port 7070 --config_file configs/deepspeed_inference_config.yml main.py \
#         --inference_ds --config configs/swinv2_large.yml \
#         --eval_bs_per_gpu 128 \
#         --dataset utkface_gender --finetune lora \
#         --eval outputs/utkface_gender-swinv2_large-lora-$config_tag-rank$rank \
#         --run_name eval-utkface_gender-swinv2_large-lora-$config_tag-rank$rank
# end
