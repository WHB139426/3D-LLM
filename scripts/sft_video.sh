MODEL_NAME_OR_PATH="/home/haibo/haibo_workspace/weights/InternVL3_5-1B-HF"
OUTPUT_DIR="/home/haibo/haibo_workspace/checkpoints/SpatialLM-InternVL3_5-1B-HF-FT-ScanRef-Multi3DRef-ReferIt3D"
RUN_NAME="SpatialLM-InternVL3_5-1B-HF-FT-ScanRef-Multi3DRef-ReferIt3D"

# --unfreeze_vision_tower \
# --vision_tower_lr 2e-6 \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 --master_port=32123 training/train.py \
    --deepspeed ./configs/zero3.json \
    --dataset scanref_multi3dref_referit3d \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --model_max_length 16384 \
    --num_bins 1280 \
    --num_frames 20 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --report_to "wandb" \
    --run_name ${RUN_NAME} \