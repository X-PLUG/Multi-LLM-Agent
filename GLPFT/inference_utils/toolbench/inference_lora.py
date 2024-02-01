# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.


from dataclasses import dataclass, field
import json
import pathlib
from typing import Any, Dict, Optional, Union
import os
from tqdm import tqdm
import math

from pathlib import Path
from transformers.generation.configuration_utils import GenerationConfig

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from peft import PeftModel, PeftConfig

from rouge import Rouge
import re

from utils.tool_conversation import SeparatorStyle
from utils.model_adapter import get_conversation_template
from utils.trainer_utils import TrainerForPred
from utils.prompt_lib import prompt_dict

# from bmtools.models.llama_model import LlamaModel

def parse(text: str):
    action_regex = r"Action: (.*)\n"
    input_regex = r"Action Input: (.*)\n"
    text += "\n"
    action = re.findall(action_regex, text)
    action_input = re.findall(input_regex, text)
    return action, action_input

def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
    rougel = rouge_score["rouge-l"]["f"]
    return rougel

def evaluate_em(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    em = 0
    for cand, ref in zip(cand_list, ref_list):
        em += (1 if cand == ref else 0)
    return em/len(cand_list)


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    max_input_length: int = field(
        default=1750
    )
    agent_mode: str = field(
        default='all', metadata={"help": "to train which model, all/assistant(to distribute task to caller and summary)/caller"}
    )
    prompt_type: str = field(
        default='v1', metadata={"help": "the prompt template"}
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


local_rank = None


def rank0_print(training_args, *args):
    if local_rank == 0 or local_rank is None:
        print(*args)





def nested_load_test_data(data_path):
    test_raw_data = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            temp_test = nested_load_test_data(os.path.join(data_path, f))
            test_raw_data += temp_test
        return test_raw_data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        rank0_print("Load data from",data_path)
        temp_data =  json.load(open(data_path, "r"))
        test_raw_data = temp_data
        return test_raw_data
    else:
        return []


def build_infer_samples(data_args):
    print("Loading data...")
    data_paths = data_args.data_path.split(',')
    # we will organize the data by file
    raw_data = []
    for data_path in data_paths:
        raw_data += nested_load_test_data(data_path=data_path)
    conv = get_conversation_template("collab_agent_v3")
    roles = {"user": conv.roles[0], "assistant": conv.roles[1], "caller": conv.roles[2], 'observation':conv.roles[3], 'conclusion': conv.roles[4]}

    assert data_args.agent_mode in ['all', 'assistant', 'caller', 'conclusion']
    if data_args.agent_mode == 'all':
        role_list = ['assistant', 'caller', 'conclusion']
    else:
        role_list = [data_args.agent_mode]

    conversations = []
    prompt_temp = prompt_dict[data_args.prompt_type]
    # Apply prompt templates
    for d in raw_data:
        conv.messages = []
        tools = d['tools']
        instruction = d['instruction']
        c = d['conversations']
        tool_docs = ""
        for t in tools:
            tool_docs += json.dumps(t) + '\n'
        query = prompt_temp.replace('{doc}', tool_docs).replace('{instruction}',instruction)
        
        conv.append_message("user", query)
        for j,u in enumerate(c):
            if u['from'] in role_list:
                prompt = conv.get_prompt()
                conversations.append({
                    'tools':d['tools'],
                    'instruction':d['instruction'],
                    'history':c[:j],
                    'model_input': prompt + '%s: '%(roles[u["from"]]),
                    'reference': u['value']
                })
            role = roles[u["from"]]
            conv.append_message(role, u["value"])
    return conversations


class InferDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, args):
        super(InferDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.args = args

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        return self.raw_data[i]

        
class Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
    
    def __call__(self, features):
        input_ids = [self.tokenizer.encode(x) for x in features] # tokens, not pad
        max_len = max([len(t) for t in input_ids])
        max_len = min(self.args.max_input_length, max_len)
        new_input_ids = []
        for t in input_ids:
            if len(t) > max_len:
                new_t = t[-max_len:]
            else:
                new_t = [self.tokenizer.pad_token_id] * (max_len - len(t)) + t
            new_input_ids.append(new_t)
        input_ids = torch.LongTensor(new_input_ids)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids).masked_fill(attention_mask, 1)
        return dict(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
    
def infer():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    # load data
    infer_samples = build_infer_samples(data_args)
    # print(len(infer_samples))
    # print(json.dumps(infer_samples[0], indent=2))
    # exit()
    # init model
    # customllm = LlamaModel(model_args.model_name_or_path)

    config = PeftConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path
    )
    model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    
    # use_gpu = (True if device == "cuda" else False)

    data_collator = Collator(tokenizer, data_args)

    trainer = TrainerForPred(
        model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator
    )

    if trainer.is_local_process_zero():
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        
    test_dataset = InferDataset([d['model_input'] for d in infer_samples], tokenizer, data_args)
    outputs = trainer.predict(test_dataset)

    preds = []
    refs = []
    for i, o in enumerate(outputs):
        candidate = tokenizer.decode(o, skip_special_tokens=True)
        if candidate.startswith(': '):
            candidate = candidate[2:]
        if candidate.strip() in ['','.',',']:
            candidate = 'none'
        reference = infer_samples[i]['reference']
        infer_samples[i]['predictions'] = candidate
        if reference.strip() in ['','.',',']:
            reference = "none"
        refs.append(reference)
        preds.append(candidate)
            

        rougel = round(evaluate_rougel(preds, refs), 2)

        if trainer.is_world_process_zero():
            with open(os.path.join(training_args.output_dir, 'predictions.json'), 'w') as f:
                json.dump(infer_samples,f, indent=4)
    
    if trainer.is_world_process_zero():
        print('Overall rouge:', rougel)
        metrics = {
            'rouge': rougel,
        }

        with open(os.path.join(training_args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    infer()
