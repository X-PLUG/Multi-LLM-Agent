LLAMA_PATH=""
NNODE=8
PORT=12345
BSZ=6
GA=1


###############################
#  GLPFT stage-1: train backbone
###########################

EXP_NAME=/toolbench/backbone
export PYTHONPATH=./
torchrun --nproc_per_node=$NNODE --master_port=$PORT train_mem.py \
    --model_name_or_path $LLAMA_PATH  \
    --data_path dataset/toolbench/train/train_backbone.json\
    --output_dir saved_models/$EXP_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BSZ \
    --per_device_eval_batch_size $BSZ \
    --gradient_accumulation_steps $GA \
    --evaluation_strategy "no" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed ds_configs/stage3-a100.json \
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True 





###############################
#  GLPFT stage-1: train planner, caller, summarizer
###########################



BB_PATH="saved_models/toolbench/backbone"


EXP_NAME=/toolbench/planner
export PYTHONPATH=./
torchrun --nproc_per_node=$NNODE --master_port=$PORT train_mem.py \
    --model_name_or_path $BB_PATH  \
    --data_path dataset/toolbench/train/train_planner.json \
    --output_dir saved_models/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BSZ \
    --per_device_eval_batch_size $BSZ \
    --gradient_accumulation_steps $GA \
    --evaluation_strategy "no" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True



EXP_NAME=/toolbench/caller
export PYTHONPATH=./
torchrun --nproc_per_node=$NNODE --master_port=$PORT train_mem.py \
    --model_name_or_path $BB_PATH  \
    --data_path dataset/toolbench/train/train_caller.json \
    --output_dir saved_models/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BSZ \
    --per_device_eval_batch_size $BSZ \
    --gradient_accumulation_steps $GA \
    --evaluation_strategy "no" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True


EXP_NAME=/toolbench/summarizer
export PYTHONPATH=./
torchrun --nproc_per_node=$NNODE --master_port=$PORT train_mem.py \
    --model_name_or_path $BB_PATH  \
    --data_path dataset/toolbench/train/train_summarizer.json \
    --output_dir saved_models/$EXP_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BSZ \
    --per_device_eval_batch_size $BSZ \
    --gradient_accumulation_steps $GA \
    --evaluation_strategy "no" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.4 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True

