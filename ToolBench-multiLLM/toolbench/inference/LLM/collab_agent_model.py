#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
from peft import PeftModel, PeftConfig
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
)

from toolbench.inference.LLM.modeling_llama import LlamaForCausalLM
import transformers
from toolbench.utils import process_system_message
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser, collab_agent_parser
import json
from toolbench.inference.Downstream_tasks.prompt_lib import prompt_dict


class CollbaAgentV3:
    def __init__(self, model_name_or_path: list, template:str="collab_agent_v3", device: str="cuda", cpu_offloading: bool=False, load_8bit: bool=False, multi_gpu=True) -> None:
        super().__init__()
        assistant_model_name_or_path = model_name_or_path[0][0]
        assistant_use_lora = model_name_or_path[0][1]
        caller_model_name_or_path = model_name_or_path[1][0]
        caller_use_lora = model_name_or_path[1][1]
        summarizer_model_name_or_path = model_name_or_path[2][0]
        summarizer_use_lora = model_name_or_path[2][1]
        self.model_name = model_name_or_path
        self.template = template
        self.use_gpu = (True if device == "cuda" else False)
        if device == "cuda" and not cpu_offloading and multi_gpu:
            device_0 = torch.device("cuda:0")
            device_1 = torch.device("cuda:1")
            device_2 = torch.device("cuda:2")
        else:
            device_0 = device
            device_1 = device
            device_2 = device
        self.assistant_model, self.assistant_tokenizer = self.load_model_and_tokenizer(assistant_model_name_or_path, assistant_use_lora, load_8bit, device_0)
        
        self.caller_model, self.caller_tokenizer = self.load_model_and_tokenizer(caller_model_name_or_path, caller_use_lora, load_8bit, device_1)
        self.summarizer_model, self.summarizer_tokenizer = self.load_model_and_tokenizer(summarizer_model_name_or_path, summarizer_use_lora, load_8bit, device_2)
        
        self.use_gpu = (True if device == "cuda" else False)
        self.chatio = SimpleChatIO()

    def load_model_and_tokenizer(self,model_name_or_path, use_lora, load_8bit,device):
        if use_lora:
            config = PeftConfig.from_pretrained(model_name_or_path)
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False, model_max_length=8192, padding_side="right")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map='balanced'
            )
            # model = model.to(device)
            model = PeftModel.from_pretrained(model, model_name_or_path, torch_dtype=torch.float16)
            # model = model.to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, model_max_length=8192)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, low_cpu_mem_usage=True, device_map='balanced'
            )
            # model = model.to(device)
        if tokenizer.pad_token_id == None:
            tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def prediction(self, history:str, tools:list, prompt_type: dict,  stop: Optional[List[str]] = None) -> str:

        
        tool_docs = ""
        for t in tools:
            tool_docs += json.dumps(t) + '\n'    
        tool_names = ', '.join([t['Name'] for t in tools])
        planner_prompt_temp = prompt_dict[prompt_type['planner']]
        prompt = planner_prompt_temp.replace('{doc}', tool_docs).replace('{tool_names}',tool_names).replace('{history}', history)

        prompt = prompt + " assistant: "
    
        gen_params = {
            "model": "",
            "prompt": prompt,
            "temperature": 0.5,
            "max_new_tokens": 512,
            "stop": "</s>",
            "stop_token_ids": None,
            "echo": False
        }
        generate_stream_func = generate_stream
        # process asssistant model
        cnt = 0
        while cnt < 5:
            output_stream = generate_stream_func(self.planner_model, self.planner_tokenizer, gen_params, "cuda", 4096, force_generate=True)
            outputs,model_input = self.chatio.return_output(output_stream)
            planner_prediction = outputs.strip()
            
            if planner_prediction.startswith(': '):
                planner_prediction = planner_prediction[2:]
            if planner_prediction.strip() in ['','.',',']:
                planner_prediction = 'none'
                cnt += 1
                continue
            else:

                break
        
        if "Next: give up."  in planner_prediction:
            action_end_idx = planner_prediction.index("Next: give up.")
            planner_prediction = planner_prediction[:action_end_idx + len("Next: give up.")]
            return planner_prediction, model_input
        elif "Next: conclusion." in planner_prediction:
            # conclusion
            action_end_idx = planner_prediction.index("Next: conclusion.")
            planner_prediction = planner_prediction[:action_end_idx + len("Next: conclusion.")]

            summarizer_prompt_temp = prompt_dict[prompt_type['conclusion']]
            prompt = summarizer_prompt_temp.replace('{doc}', tool_docs).replace('{tool_names}',tool_names)
            prompt = prompt.replace('{thought}',planner_prediction)
            dispatch = history + ('assistant: ' + planner_prediction + '</s>')
            prompt = prompt.replace('{history}',dispatch)

            summarizer_gen_params =  {
                    "model": "",
                    "prompt": prompt + ' conclusion: ',
                    "temperature": 0.5,
                    "max_new_tokens": 512,
                    "stop": "</s>",
                    "stop_token_ids": None,
                    "echo": False
                }
            output_stream = generate_stream_func(self.summarizer_model, self.summarizer_tokenizer, summarizer_gen_params, "cuda", 4096, force_generate=True)
            outputs,_ = self.chatio.return_output(output_stream)
            summarizer_prediction = outputs.strip()

            final_prediction = planner_prediction + '</s>conclusion: ' + summarizer_prediction

            return final_prediction, model_input
        else:
            if "Next: caller." in planner_prediction:
                # conclusion
                action_end_idx = planner_prediction.index("Next: caller.")
                planner_prediction = planner_prediction[:action_end_idx + len("Next: caller.")]
            else:
                planner_prediction = planner_prediction + ' Next: caller.'

            caller_prompt_temp = prompt_dict[prompt_type['caller']]
            prompt = caller_prompt_temp.replace('{doc}', tool_docs).replace("{tool_names}", tool_names)
            prompt = prompt.replace('{thought}',planner_prediction)
            dispatch = history + ('assistant: ' + planner_prediction + '</s>')
            prompt = prompt.replace('{history}',dispatch)

            # use caller
            caller_gen_params =  {
                "model": "",
                "prompt": prompt + ' caller: ',
                "temperature": 0.5,
                "max_new_tokens": 512,
                "stop": "</s>",
                "stop_token_ids": None,
                "echo": False
            }
            output_stream = generate_stream_func(self.caller_model, self.caller_tokenizer, caller_gen_params, "cuda", 4096, force_generate=True)
            outputs,_ = self.chatio.return_output(output_stream)
            caller_prediction = outputs.strip()

            final_prediction = planner_prediction + '</s>caller: ' + caller_prediction
            # print(final_prediction)
            return final_prediction, model_input

        
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions, functions_for_input, process_id=0, prompt_type=None,**args):
        """
        functions: the registed functions
        
        """

        self.time = time.time()
        conversation_history = self.conversation_history

        conv_history = ""
        for message in conversation_history:
            role = message["role"]
            if role == 'assistant':
                conv_history += ('assistant: '+message['value']+'</s>')
            elif role == 'user':
                conv_history += ('user: '+message['value']+'</s>')
            elif role == 'observation':
                conv_history += ('observation: '+message['value'])
            # print(json.dumps(message, indent=2))
            # assert role == conv.roles[j % 2], f"{i}"


        if functions != []:
            predictions, model_input = self.prediction(conv_history, functions_for_input, prompt_type)
        else:
            predictions, model_input = self.prediction(conv_history, functions_for_input, prompt_type)

        decoded_token_len = len(self.planner_tokenizer(predictions))
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")

        # react format prediction
        thought, action, action_input = collab_agent_parser(predictions)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input,

            },
            'value': predictions,
            'model_input': model_input
        }

        return message, 0, decoded_token_len
    


    def prediction_single(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        gen_params = {
            "model": "",
            "prompt": prompt,
            "temperature": 0.5,
            "max_new_tokens": 512,
            "stop": "</s>",
            "stop_token_ids": None,
            "echo": False
        }
        generate_stream_func = generate_stream
        # process asssistant model
        output_stream = generate_stream_func(self.planner_model, self.planner_tokenizer, gen_params, "cuda", 8192, force_generate=True)
        outputs = self.chatio.return_output(output_stream)
        planner_prediction = outputs.strip()
        
        return planner_prediction
    
    def parse_rank(self,functions,process_id=0,**args):
        """
        functions: the registed functions
        
        """

        self.time = time.time()
        conversation_history = self.conversation_history
        prompt = ""
        for message in conversation_history:
            prompt += message["role"] + ": " + message['value'] + " "
            # assert role == conv.roles[j % 2], f"{i}"
        prompt += " assistant: "

        if functions != []:
            predictions = self.prediction_single(prompt)
        else:
            predictions = self.prediction_single(prompt)

        decoded_token_len = len(self.planner_tokenizer(predictions))
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")
        
        # react format prediction
        thought, action, action_input = collab_agent_parser(predictions)
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input,

            },
            'value': predictions 
        }

        return message, 0, decoded_token_len
