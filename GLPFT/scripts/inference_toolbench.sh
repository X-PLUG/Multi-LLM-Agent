NNODE=8
PORT=12345

PLAN_PATH="saved_models/planner"
CAL_PATH="saved_models/caller"
SUM_PATH="saved_models/summarizer"


LAB_DIR=output_res/toolbench
P_TYPE_PLAN=toolbench_planner
P_TYPE_CAL=toolbench_caller
P_TYPE_SUM=toolbench_summarizer


for DOMAIN in 'in_domain' 'out_of_domain'
do
    export PYTHONPATH=./
    torchrun --nproc_per_node=$NNODE --master_port=$PORT inference_utils/toolbench/infer_pipeline.py \
        --planner_model_name_or_path $PLAN_PATH  \
        --planner_use_lora False \
        --caller_model_name_or_path $CAL_PATH  \
        --caller_use_lora False \
        --summarizer_model_name_or_path $SUM_PATH  \
        --summarizer_use_lora False \
        --per_device_eval_batch_size 1 \
        --data_path dataset/toolbench/test/$DOMAIN.json \
        --bf16_full_eval \
        --assistant_prompt_type $P_TYPE_PLAN \
        --caller_prompt_type $P_TYPE_CAL \
        --conclusion_prompt_type $P_TYPE_SUM \
        --max_input_length 3750 \
        --output_dir $LAB_DIR/$DOMAIN 

    python inference_utils/toolbench/evaluate-multi_agent.py \
    --input_path $LAB_DIR/$DOMAIN/predictions.json \
    --output_path $LAB_DIR/$DOMAIN/metrics.json 

done