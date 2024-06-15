for seed in 13 1337 42
    for rank in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
        # llama2 lora
        python main.py --config configs/llama2.yml --dataset mt_gender_translation_pro \
            --seed $seed --lora_rank $rank \
            --eval outputs/mt_gender_translation_pro-llama2-7b-lora-best-rank$rank-seed$seed \
            --run_name eval-mt_gender_translation_pro-llama2-7b-lora-best-rank$rank-seed$seed \
            --finetune lora --model_parallel

        # mistral lora
        python main.py --config configs/mistral.yml --dataset mt_gender_translation_pro \
            --seed $seed --lora_rank $rank \
            --eval outputs/mt_gender_translation_pro-mistral-7b-lora-best-rank$rank-seed$seed \
            --run_name eval-mt_gender_translation_pro-mistral-7b-lora-best-rank$rank-seed$seed \
            --finetune lora --model_parallel
    end
end