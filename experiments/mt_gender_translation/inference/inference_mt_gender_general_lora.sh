for seed in 13 1337 42 666 888
    # Best by sacrebleu:
    # mt_gender_translation_general-mistral-7b-lora-epoch2-lr0.001
    # mt_gender_translation_general-llama2-7b-lora-epoch6-lr0.003

    # llama2 lora
    python main.py --config configs/llama2.yml --dataset mt_gender_translation_general_test \
        --seed $seed \
        --eval outputs/mt_gender_translation_general-llama2-7b-lora-best-seed$seed \
        --run_name eval-mt_gender_translation_general-llama2-7b-lora-best-seed$seed \
        --finetune lora --model_parallel

    # mistral lora
    python main.py --config configs/mistral.yml --dataset mt_gender_translation_general_test \
        --seed $seed \
        --eval outputs/mt_gender_translation_general-mistral-7b-lora-best-seed$seed \
        --run_name eval-mt_gender_translation_general-mistral-7b-lora-best-seed$seed \
        --finetune lora --model_parallel
end