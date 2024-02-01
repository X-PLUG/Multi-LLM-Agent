'''
Close-domain QA Pipeline
'''

import argparse
from toolbench.inference.Downstream_tasks.rapidapi import pipeline_runner


def str2bool(text):
    if text.lower() in ['true', 't', '1']:
        return True
    else:
        return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default="toolllama", required=False, help='chatgpt_function or davinci or toolllama')
    parser.add_argument('--openai_key', type=str, default="", required=False, help='openai key for chatgpt_function or davinci model')
    parser.add_argument('--model_path', type=str, default="your_model_path/", required=False, help='')
    parser.add_argument('--user_agent_collab', type=str, default="true", required=False, help="")
    parser.add_argument('--planner_model_path', type=str, default="", required=False, help='')
    parser.add_argument('--caller_model_path', type=str, default="", required=False, help='')
    parser.add_argument('--summarizer_model_path', type=str, default="", required=False, help='')
    parser.add_argument('--planner_prompt_type', type=str, default="toolbench_planner", required=False, help='')
    parser.add_argument('--caller_prompt_type', type=str, default="toolbench_caller", required=False, help='')
    parser.add_argument('--summarizer_prompt_type', type=str, default="toolbench_summarizer", required=False, help='')
    parser.add_argument('--tool_root_dir', type=str, default="your_tools_path/", required=True, help='')
    parser.add_argument("--lora", action="store_true", help="Load lora model or not.")
    parser.add_argument('--planner_use_lora', type=str, default="false", required=False, help="")
    parser.add_argument('--caller_use_lora', type=str, default="false", required=False, help="")
    parser.add_argument('--summarizer_use_lora', type=str, default="false", required=False, help="")
    parser.add_argument('--lora_path', type=str, default="your_lora_path if lora", required=False, help='')
    parser.add_argument('--use_multi_gpu', type=str, default="false", required=False, help="")
    parser.add_argument('--max_observation_length', type=int, default=1024, required=False, help='maximum observation length')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], required=False, help='observation compress method')
    parser.add_argument('--method', type=str, default="CoT@1", required=False, help='method for answer generation: CoT@n,Reflexion@n,BFS,DFS,UCT_vote')
    parser.add_argument('--input_query_file', type=str, default="", required=False, help='input path')
    parser.add_argument('--output_answer_file', type=str, default="",required=False, help='output path')
    parser.add_argument('--toolbench_key', type=str, default="",required=False, help='your toolbench key to request rapidapi service')
    parser.add_argument('--rapidapi_key', type=str, default="",required=False, help='your rapidapi key to request rapidapi service')
    parser.add_argument('--use_rapidapi_key', action="store_true", help="To use customized rapidapi service or not.")
    
    args = parser.parse_args()
    args.planner_use_lora = str2bool(args.planner_use_lora)
    args.caller_use_lora = str2bool(args.caller_use_lora)
    args.summarizer_use_lora = str2bool(args.summarizer_use_lora)
    args.use_multi_gpu = str2bool(args.use_multi_gpu)

    pipeline_runner = pipeline_runner(args)
    pipeline_runner.run()

