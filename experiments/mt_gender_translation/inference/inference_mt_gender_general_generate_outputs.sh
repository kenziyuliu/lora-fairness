set seed 13
# Best by sacrebleu:
# mt_gender_translation_general-mistral-7b-full-epoch1-lr0.000003
# mt_gender_translation_general-llama2-7b-full-epoch2-lr0.00001

# llama2 full
python main.py --config configs/llama2.yml --dataset mt_gender_translation_general_test \
    --seed $seed \
    --eval outputs/mt_gender_translation_general-llama2-7b-full-best-seed$seed \
    --run_name eval-translation_output-llama2-7b-full-best-seed$seed \
    --finetune full --model_parallel

# mistral full
python main.py --config configs/mistral.yml --dataset mt_gender_translation_general_test \
    --seed $seed \
    --eval outputs/mt_gender_translation_general-mistral-7b-full-best-seed$seed \
    --run_name eval-translation_output-mistral-7b-full-best-seed$seed \
    --finetune full --model_parallel

# Best by sacrebleu:
# mt_gender_translation_general-mistral-7b-lora-epoch2-lr0.001
# mt_gender_translation_general-llama2-7b-lora-epoch6-lr0.003

# llama2 lora
python main.py --config configs/llama2.yml --dataset mt_gender_translation_general_test \
    --seed $seed \
    --eval outputs/mt_gender_translation_general-llama2-7b-lora-best-seed$seed \
    --run_name eval-translation_output-llama2-7b-lora-best-seed$seed \
    --finetune lora --model_parallel

# mistral lora
python main.py --config configs/mistral.yml --dataset mt_gender_translation_general_test \
    --seed $seed \
    --eval outputs/mt_gender_translation_general-mistral-7b-lora-best-seed$seed \
    --run_name eval-translation_output-mistral-7b-lora-best-seed$seed \
    --finetune lora --model_parallel