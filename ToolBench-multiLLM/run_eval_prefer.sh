

# to evaluate win rate, we need to first convert the chatgpt_cot results and compute its pass rate

export REF_ANSWER_PATH=data/toolbench/reproduction_data/model_predictions/chatgpt_cot
export REF_CONVERTED_ANSWER_PATH=data/toolbench/reproduction_data/model_predictions_converted/chatgpt_cot
for test_set in "G1_instruction" "G1_category" "G1_tool" "G2_category" "G2_instruction" "G3_instruction"
do
    answer_dir=$ORI_ANSWER_PATH/$test_set
    output_file=${CONVERTED_ANSWER_PATH}/${test_set}.json
    python toolbench/tooleval/convert_to_answer_format.py\
        --answer_dir ${answer_dir} \
        --method DFS_woFilter_w2 \
        --output ${output_file}
done

export SAVE_PATH=pass_rate_results/chatgpt_cot
export CANDIDATE_MODEL=chatgpt_cot
export DATA_DIR="data/toolbench/data"
export API_POOL_FILE=path/to/your/openai_key_json_file.json
export PYTHONPATH=./
python toolbench/tooleval/eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids $DATA_DIR/test_query_ids \
    --max_eval_threads 1 \
    --evaluate_times 7


# then begin to evalute

export OUTPUT_CONVERTED_ANSWER_PATH=output_res/converted/multi-llm-agent
export SAVE_PATH=win_rate_results
export REF_PASS_TARE_PATH=pass_rate_results/chatgpt_cot
export OUTPUT_PASS_TARE_PATH=pass_rate_results/v9/multi-llm-agent
export REFERENCE_MODEL=chatgpt_cot
export CANDIDATE_MODEL=multi-llm-agent
# export API_POOL_FILE=path/to/your/openai_key_json_file.json


export PYTHONPATH=./
python toolbench/tooleval/eval_preference.py \
    --ref_converted_answer_path ${REF_CONVERTED_ANSWER_PATH} \
    --output_converted_answer_path ${OUTPUT_CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids data/test_query_ids/ \
    --save_path ${SAVE_PATH} \
    --ref_pass_rate_result_path ${REF_PASS_TARE_PATH} \
    --output_pass_rate_result_path ${OUTPUT_PASS_TARE_PATH} \
    --max_eval_threads 1 \
    --use_pass_rate true \
    --evaluate_times 7

