# ✨α-UMi: Small LLMs Are Weak Tool Learners: A Multi-LLM Agent
<div align="center">
Weizhou Shen<sup>1</sup>, Chenliang Li<sup>2</sup>, Hongzhan Chen<sup>1</sup>, Ming Yan<sup>2*</sup>, Xiaojun Quan<sup>1*</sup>, Hehong Chen<sup>2</sup>, Ji Zhang<sup>2</sup>, Fei Huang<sup>2</sup>
</div>
<div align="center">
shenwzh3@mail2.sysu.edu.cn, quanxj3@mail.sysu.edu.cn, ym119608@alibaba-inc.com
</div>
<div align="center">
<sup>1</sup>Sun Yat-sen University <sup>2</sup>Alibaba Group
</div>
<div align="center">
*Corresponding authors
</div>



<div align="center">
    <a href="https://github.com/modelscope/modelscope-agent/tree/alpha_umi"><img src="assets/Demo-ModelScope-brightgreen.svg" alt="Demo ModelScope"></a>
    <!-- <a href="https://replicate.com/joehoover/mplug-owl"><img src="https://replicate.com/replicate/mplug-owl/badge" alt="Run with Replicate"></a>
    <a href="https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE"><img src="assets/LICENSE-Apache%20License-blue.svg" alt="License"></a> -->
    <a href="https://arxiv.org/pdf/2401.07324.pdf"><img src="assets/Paper-Arxiv-orange.svg" ></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-PLUG%2FMulti-LLM-Agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
    <!-- <a href="https://twitter.com/xuhaiya2483846/status/1654640739010351106"><img src='assets/-twitter-blue.svg'></a> -->
</div>

<div align="center">
<a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
<hr>
</div>
<!--
English | [简体中文](README_zh.md)
<hr>
-->



<div align="center">


<img src="assets/concept.png"  width="70%">

A conceptual comparison of  traditional single-LLM agent framework (top) and  alpha-UMi (bottom). 

</div>

α-UMi is a Multi-LLM collaborated agent for tool learning. It decomposes the capabilities of a single LLM into three components, namely planner,
caller, and summarizer. For each step of agent execution. The planner generate a rationale for the current step based on the state of the system and selects the caller or summarizer to generate downstream output. The caller is directed by the rationale and responsible for invocating specific tools to interact with. The summarizer is guided by the planner to craft the ultimate user answer based on the execution trajectory.


<div align="center">
<img src="assets/framework.png"  width="95%"> 

An illustration of how α-UMi works to complete a task.
</div>

## Spotlight
* Enabling small LLMs to collaborate and outperform strong close-source large LLMs in tool learning.
* More flexible prompt design than single-LLM agent system.
* Two-stage Global-to-Local Progressive Fine-tuning (GLPFT) for successfully training the multi-LLM agent.



## News
* [01.30] We released code of ✨α-UMi with its pre-trained and instruction tuning checkpoints.

## Checkpoints

| Model | 7b | 13b |
|-------|----|----|
| planner | [huggingface](https://huggingface.co/shenwzh3/alpha-umi-planner-7b)  / [modelscope](https://www.modelscope.cn/models/iic/alpha-umi-planner-7b) | [huggingface](https://huggingface.co/shenwzh3/alpha-umi-planner-13b)  / [modelscope](https://www.modelscope.cn/models/iic/alpha-umi-planner-13b) |
| caller | [huggingface](https://huggingface.co/shenwzh3/alpha-umi-caller-7b)  / [modelscope](https://www.modelscope.cn/models/iic/alpha-umi-caller-7b) | [huggingface](https://huggingface.co/shenwzh3/alpha-umi-caller-13b)  / [modelscope](https://www.modelscope.cn/models/iic/alpha-umi-caller-13b) |
| summarizer | [huggingface](https://huggingface.co/shenwzh3/alpha-umi-summarizer-7b)  / [modelscope](https://www.modelscope.cn/models/iic/alpha-umi-summarizer-7b) | [huggingface](https://huggingface.co/shenwzh3/alpha-umi-summarizer-13b)  / [modelscope](https://www.modelscope.cn/models/iic/alpha-umi-summarizer-13b) |




## Usage
### Install Requirements
1. Create conda environment
```bash
conda create -n multi_llm_agent python=3.10
conda activate multi_llm_agent
```

2. Install PyTorch

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

3. Install other dependencies
```bash
pip install -r requirements.txt
```

### Data Preparation

#### ToolBench
1. First download the oringinal ToolBench dataset from [Google Drive](https://drive.google.com/drive/folders/1yBUQ732mPu-KclJnuQELEhtKakdXFc3J) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/), and put the data to ```./data``` folder.

2. Preprocess data for training
```
cd ./GLPFT

ORI_DATA_DIR="../data/toolbench/data" # your data path to save the toolbench raw data
RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
TRAIN_DATA_OUT_DIR="dataset/toolbench/train"
export PYTHONPATH=./


python process_data/toolbench/prepro_raw_stage_1.py \
 --data_dir $ORI_DATA_DIR \
 --output_path $RAW_DATA_OUT_DIR


python process_data/toolbench/prepro_raw_stage_2.py \
 --input_path $RAW_DATA_OUT_DIR/raw_data_stage_1.json \
 --output_path $RAW_DATA_OUT_DIR



for MODE in 'backbone' 'planner' 'caller' 'summarizer'
do
    python process_data/toolbench/prepro_$MODE.py \
        --input_path $RAW_DATA_OUT_DIR/raw_data_stage_2.json \
        --output_path $TRAIN_DATA_OUT_DIR/train_$MODE.json \
        --prompt_type toolbench_$MODE
done
```

After runnning the above script, you will create the training data of ToolBench for GLPFT, which will be stored in ```./GLPFT/dataset/toolbench/train```.

### GLPFT Training

Our α-UMi adopts a two-stage GLPFT fine-tuning that first warm-up a backbone LLM and then fine-tune the planner, caller, summarizer separately.

1. First, we fine-tune an LLM for the whole tool learning agent task.

```
cd ./GLPFT

LLAMA_PATH="" # your path for initial LLM checkpoint
NNODE=8
PORT=12345
BSZ=6
GA=1

EXP_NAME=/toolbench/backbone  # path to save model
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

```


2. After obtaining the backbone, we begin to fine-tune planner, caller and summarizer:

```
cd ./GLPFT

NNODE=8
PORT=12345
BSZ=6
GA=1


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


```


### Inference and evaluate

We provide the statically test data for the experiments in Section 4.1 of our paper in ```./GLPFT/dataset/toolbench/test```, we can inference and evaluate the α-UMi system as Section 4.1 by running the following script:
```
cd ./GLPFT

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
```

## α-UMi with RapidAPI Simulator

We surpport using α-UMi with the RapidAPI simulator implemented by the ToolBench team ([github](https://github.com/OpenBMB/ToolBench)), the codes are in ```./ToolBench-multiLLM```. To do so, you should first fill out the [form](https://forms.gle/oCHHc8DQzhGfiT9r6) to request a Toolbench Key from Toolbench team. Then you can begin to run the simulator with the trained Planner, Caller and Summarizer:

```
cd ToolBench-multiLLM

DATA_DIR="../data/toolbench/data"
PLAN_PATH="../GLPFT/saved_models/planner"
CAL_PATH="../GLPFT/saved_models/caller"
SUM_PATH="../GLPFT/saved_models/summarizer"
EXP_NAME="multi-llm-agent"
TBKEY="" # your toolbench key



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
```

We also surpport compuing the pass_rate and win_rate metrics as ToolBench.

To compute pass rate:
```
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
```


To compute win_rate,  we choose chatgpt_cot as the reference model, we need to first convert the chatgpt_cot results and compute its pass rate:

```
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
```

Then we bengin to evaluate:
```
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
```

## Experimental Results

Results of the statically evaluation (step-level comparison with annotated reference)

<div align="center">
<img src="assets/result_static.png"  width="95%"> 
</div>

Results of the real-time evaluation (calling real APIs to solve the user task)

<div align="center">
<img src="assets/result_real.png"  width="95%"> 
</div>

## To do

- [ ] Release our model and code for ToolAlpaca.
- [ ] Release our model and code for MATH and GSM8K, and our training data (collected with TORA (Gou et al., 2023))
- [ ] Make α-UMi generalized to more agent tasks!

## Citation

```
@misc{shen2024small,
      title={Small LLMs Are Weak Tool Learners: A Multi-LLM Agent}, 
      author={Weizhou Shen and Chenliang Li and Hongzhan Chen and Ming Yan and Xiaojun Quan and Hehong Chen and Ji Zhang and Fei Huang},
      year={2024},
      eprint={2401.07324},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```