DATA_DIR="data/toolbench/data"
PLAN_PATH="../GLPFT/saved_models/planner"
CAL_PATH="../GLPFT/saved_models/caller"
SUM_PATH="../GLPFT/saved_models/summarizer"
EXP_NAME="multi-llm-agent"
TBKEY=



for TEST_SET in 'G1_category' 'G1_instruction' 'G1_tool' 'G2_category' 'G2_instruction' 'G3_instruction'
do
    export PYTHONPATH=./
    python toolbench/inference/qa_pipeline.py \
        --backbone_model collab_agent_v3 \
        --tool_root_dir $DATA_DIR/toolenv/tools/ \
        --user_agent_collab True \
        --planner_model_path $PLAN_PATH \
        --planner_use_lora False \
        --caller_model_path $CAL_PATH \
        --caller_use_lora False \
        --summarizer_model_path $SUM_PATH \
        --summarizer_use_lora False \
        --use_multi_gpu True \
        --max_observation_length 1024 \
        --observ_compress_method truncate \
        --method DFS_woFilter_w2 \
        --input_query_file $DATA_DIR/test_instructions/$TEST_SET.json \
        --output_answer_file output_res/$EXP_NAME/$TEST_SET \
        --toolbench_key $TBKEY
done

