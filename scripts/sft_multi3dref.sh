MODEL_NAME_OR_PATH="/home/haibo/haibo_workspace/weights/Qwen3-0.6B"
POINT_DIR="/home/haibo/haibo_workspace/weights/sonata"
OUTPUT_DIR="/home/haibo/haibo_workspace/checkpoints/SpatialLM-Qwen3-0.6B-FT-Multi3DRef"
RESUME_PATH="/home/haibo/haibo_workspace/checkpoints/SpatialLM-Qwen3-0.6B-Pretrain-SpatialLM-Scannet/model.safetensors"

RUN_NAME="SpatialLM-Qwen3-0.6B-FT-Multi3DRef"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 --master_port=32123 training/train.py \
    --unfreeze_point_backbone \
    --deepspeed ./configs/zero2.json \
    --dataset multi3dref \
    --resume_path ${RESUME_PATH} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --point_name_or_path ${POINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --point_lr 1e-4 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --model_max_length 8192 \
    --num_bins 1280 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --report_to "wandb" \
    --run_name ${RUN_NAME} \