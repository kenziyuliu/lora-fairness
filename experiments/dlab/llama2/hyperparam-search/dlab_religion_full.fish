
# Full finetuning
set subset religion

for epoch in 1 2 3 4 6 8
    for lr in 0.00001 0.00005 0.0001 0.0003
        accelerate launch \
            --main_process_port 29501 --config_file configs/fsdp_config.yml \
            main.py \
            --config configs/llama2.yml --dataset dlab_hatespeech'_'$subset \
            --finetune full --lr $lr --epochs $epoch \
            --run_name dlab-$subset-llama2-7b-full-epoch$epoch-lr$lr --wandb \
            --save_strategy no
    end
end


# Steps to generate results on a subset
# 1. Do a simple hyperparameter sweep for full/lora FT on a subset (as above)
# 2. Go to wandb, and pick 3 runs
#    - the best LoRA run
#    - the best full FT run
#    - the LoRA run that matches full FT run, in terms of eval F1
# 3. Save them and delete the rest (since the trained models are large)
# 4. (Optional) Maintain a copy of the saved runs in a separate folder
# 5. Run evaluation on the runs from step 2 (the distributed eval stuff) which
#    generates the eval text (later we want to change this to yaml files)
# 6. Run spread analysis, and put results on slides
