# Remember to change model_name_or_path and output_dir

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 accelerate launch  src/train.py \
    --deepspeed ds_config/ds_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/Qwen/Qwen2.5-14B-Instruct \
    --trust_remote_code True \
    --flash_attn fa2 \
    --finetuning_type full \
    --template qwen \
    --dataset_dir data \
    --dataset sft-zscore \
    --cutoff_len 10000 \
    --learning_rate 1e-05 \
    --num_train_epochs 1 \
    --max_samples 100000 \
    --overwrite_cache True \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 200 \
    --warmup_ratio 0.1 \
    --neftune_noise_alpha 0 \
    --output_dir /saves/qwen-14B/full/sft-zscore-new5 \
    --fp16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --ddp_timeout 180000000

