
export PYTHONPATH=./
export ORI_ANSWER_PATH=output_res/multi-llm-agent
export CONVERTED_ANSWER_PATH=output_res/converted/multi-llm-agent

mkdir ${CONVERTED_ANSWER_PATH}
for test_set in "G1_instruction" "G1_category" "G1_tool" "G2_category" "G2_instruction" "G3_instruction"
do
    answer_dir=$ORI_ANSWER_PATH/$test_set
    output_file=${CONVERTED_ANSWER_PATH}/${test_set}.json
    python toolbench/tooleval/convert_to_answer_format.py\
        --answer_dir ${answer_dir} \
        --method DFS_woFilter_w2 \
        --output ${output_file}
done


export SAVE_PATH=pass_rate_results/multi-llm-agent
export CANDIDATE_MODEL=multi-llm-agent
export DATA_DIR="data/toolbench"
export API_POOL_FILE=path/to/your/openai_key_json_file.json
export PYTHONPATH=./
python toolbench/tooleval/eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids $DATA_DIR/test_query_ids \
    --max_eval_threads 1 \
    --evaluate_times 7


