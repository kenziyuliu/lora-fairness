for seed in 13 1337 42 666 888
    # Best by sacrebleu:
    # mt_gender_translation_pro-mistral-7b-full-epoch2-lr0.000003
    # mt_gender_translation_pro-llama2-7b-full-epoch4-lr0.00001

    # llama2 full
    python main.py --config configs/llama2.yml --dataset mt_gender_translation_pro \
        --seed $seed \
        --eval outputs/mt_gender_translation_pro-llama2-7b-full-best-seed$seed \
        --run_name eval-mt_gender_translation_pro-llama2-7b-full-best-seed$seed \
        --finetune full --model_parallel

    # mistral full
    python main.py --config configs/mistral.yml --dataset mt_gender_translation_pro \
        --seed $seed \
        --eval outputs/mt_gender_translation_pro-mistral-7b-full-best-seed$seed \
        --run_name eval-mt_gender_translation_pro-mistral-7b-full-best-seed$seed \
        --finetune full --model_parallel
end