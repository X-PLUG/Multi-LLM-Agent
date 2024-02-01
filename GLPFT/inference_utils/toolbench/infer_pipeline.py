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
from typing import Any, Dict, Optional, Union
import os

import copy
from pathlib import Path
from transformers.generation.configuration_utils import GenerationConfig

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from peft import PeftModel, PeftConfig

from rouge import Rouge
import gc

from utils.trainer_utils import TrainerForPred
from utils.prompt_lib import prompt_dict
from utils.modeling_llama import LlamaForCausalLM_wrapper

# from bmtools.models.llama_model import LlamaModel


def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
    rougel = rouge_score["rouge-l"]["f"]
    return rougel


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    planner_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    caller_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    summarizer_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    planner_use_lora: bool = False
    caller_use_lora: bool = False
    summarizer_use_lora: bool = False
    use_logit_smooth: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    max_input_length: int = field(
        default=1750
    )
    num_infer_samples: int = field(default=-1)
    planner_prompt_type: str = field(
        default='v1', metadata={"help": "the prompt template"}
    )
    caller_prompt_type: str = field(
        default='v1', metadata={"help": "the prompt template"}
    )
    summarizer_prompt_type: str = field(
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


def rank0_print( *args):
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
    conversations = []
    if data_args.num_infer_samples > 0:
        raw_data = raw_data[:data_args.num_infer_samples]
    # Apply prompt templates
    prompt_temp = prompt_dict[data_args.planner_prompt_type]
    for d in raw_data:
        c = d['conversations']
        tool_docs = ""
        for t in d['tools']:
            tool_docs += json.dumps(t) + '\n'    
        tool_names = ', '.join([t['Name'] for t in d['tools']])
        query_temp = prompt_temp.replace('{doc}', tool_docs).replace('{tool_names}',tool_names)
        dispatch = ""
        for j,u in enumerate(c):
            if u['from'] == 'assistant':
                if "Next: caller." in u['value'] or "Next: conclusion." in u['value'] or "Next: give up." in u['value']:
                    prompt = query_temp.replace('{history}',dispatch)
                    if "Next: caller" in u['value'] or "Next: conclusion." in u['value']:
                        # assert c[j+1]['from'] == 'caller'
                        reference = u['from'] + ': ' + u['value'] +"</s>"+ c[j+1]['from'] + ": " + c[j+1]['value'] + '</s>'
                    else:
                        reference = u['value']
                    conversations.append({
                        'tools':d['tools'],
                        'instruction':d['instruction'],
                        'history':c[:j],
                        "dispath": copy.deepcopy(dispatch),
                        'model_input': prompt + ' assistant: ',
                        'reference': reference
                    })
                dispatch += ('assistant: '+u['value']+'</s>')
            elif u['from'] == 'user':
                dispatch += ('user: '+u['value']+'</s>')
            elif u['from'] == 'observation':
                dispatch += ('observation: '+u['value'])
            elif u['from'] == 'caller':
                dispatch += ('caller: '+u['value']+'</s>')
            elif u['from'] == 'conclusion':
                dispatch += ('conclusion: '+u['value']+'</s>')
    
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


def load_model_and_tokenizer(model_name_or_path, use_lora, use_logit_smooth):
    if use_lora:
        config = PeftConfig.from_pretrained(model_name_or_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
        if use_logit_smooth:
            model = LlamaForCausalLM_wrapper.from_pretrained(config.base_model_name_or_path)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path
            )
        model = PeftModel.from_pretrained(model, model_name_or_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        if use_logit_smooth:
            model = LlamaForCausalLM_wrapper.from_pretrained(model_name_or_path)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def infer():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    # load data
    infer_samples = build_infer_samples(data_args)

    print('load model begin')

    planner_model, planner_tokenizer = load_model_and_tokenizer(model_args.planner_model_name_or_path, model_args.planner_use_lora, model_args.use_logit_smooth)
    
    data_collator = Collator(planner_tokenizer, data_args)

    planner_trainer = TrainerForPred(
        model=planner_model, tokenizer=planner_tokenizer, args=training_args, data_collator=data_collator
    )

    process_zero = planner_trainer.is_local_process_zero()

    if process_zero:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        
    test_dataset = InferDataset([d['model_input'] for d in infer_samples], data_collator, data_args)
    outputs,_ = planner_trainer.predict(test_dataset)


    for i, o in enumerate(outputs):
        candidate = planner_tokenizer.decode(o, skip_special_tokens=True)
        if candidate.startswith(': '):
            candidate = candidate[2:]
        if candidate.strip() in ['','.',',']:
            candidate = 'none'
        infer_samples[i]['predictions'] = candidate

        # empty cache
    planner_model.to('cpu')
    planner_trainer.model.to('cpu')
    planner_trainer.model_wrapped.to('cpu')
    del planner_model
    del planner_tokenizer
    del planner_trainer.model
    del planner_trainer.model_wrapped
    del planner_trainer
    gc.collect()

    torch.cuda.empty_cache()


    infer_samples_planner = []
    infer_samples_caller = []
    infer_samples_summarizer = []
    for sample in infer_samples:
        if "Next: give up." in sample['predictions']:
            action_end_idx = sample['predictions'].index("Next: give up.")
            planner_prediction = sample['predictions'][:action_end_idx + len("Next: give up.")]
            sample['predictions'] = planner_prediction
            infer_samples_planner.append(sample)
        elif "Next: conclusion." in sample['predictions']:
            # conclusion
            action_end_idx = sample['predictions'].index("Next: conclusion.")
            planner_prediction = sample['predictions'][:action_end_idx + len("Next: conclusion.")]
            sample['planner_prediction'] = planner_prediction
            prompt_temp = prompt_dict[data_args.summarizer_prompt_type]
            tools = sample['tools']
            tool_docs = ""
            for t in tools:
                tool_docs += json.dumps(t) + '\n'
            query = prompt_temp.replace('{doc}', tool_docs)
            tool_names = ', '.join([t['Name'] for t in tools])
            query = query.replace("{tool_names}", tool_names)
            query = query.replace('{thought}',sample['planner_prediction'])
            dispatch = sample['dispath'] + ('planner: ' + sample['planner_prediction'] + '</s>')
            query = query.replace('{history}',dispatch)

            sample['model_input_for_summarizer'] = query +" conclusion: "
            infer_samples_summarizer.append(sample)
        else:
            if "Next: caller." in sample['predictions']:
                action_end_idx = sample['predictions'].index("Next: caller.")
                planner_prediction = sample['predictions'][:action_end_idx + len("Next: caller.")]
                sample['planner_prediction'] = planner_prediction
            else:
                planner_prediction = sample['predictions'] + "Next: caller."
                sample['planner_prediction'] = planner_prediction
            prompt_temp = prompt_dict[data_args.caller_prompt_type]
            tools = sample['tools']
            tool_docs = ""
            for t in tools:
                tool_docs += json.dumps(t) + '\n'
            query = prompt_temp.replace('{doc}', tool_docs)
            tool_names = ', '.join([t['Name'] for t in tools])
            query = query.replace("{tool_names}", tool_names)
            query = query.replace('{thought}',sample['planner_prediction'])
            dispatch = sample['dispath'] + ('planner: ' + sample['planner_prediction'] + '</s>')
            query = query.replace('{history}',dispatch)

            sample['model_input_for_caller'] = query +" caller: "
            infer_samples_caller.append(sample)
    

    # caller inference
    if len(infer_samples_caller) != 0:
        caller_model, caller_tokenier = load_model_and_tokenizer(model_args.caller_model_name_or_path, model_args.caller_use_lora, model_args.use_logit_smooth)
        caller_trainer = TrainerForPred(
            model=caller_model, tokenizer=caller_tokenier, args=training_args, data_collator=data_collator
        )
        caller_test_dataset = InferDataset([d['model_input_for_caller'] for d in infer_samples_caller], data_collator, data_args)
        outputs,_ = caller_trainer.predict(caller_test_dataset)
        for i, o in enumerate(outputs):
            candidate = caller_tokenier.decode(o, skip_special_tokens=True)
            if candidate.startswith(': '):
                candidate = candidate[2:]
            if candidate.strip() in ['','.',',']:
                candidate = 'none'
            infer_samples_caller[i]['predictions'] = "asssitant: " + infer_samples_caller[i]['planner_prediction'] + "</s>caller: " + candidate
        caller_model.to('cpu')
        caller_trainer.model.to('cpu')
        caller_trainer.model_wrapped.to('cpu')
        del caller_model
        del caller_tokenier
        del caller_trainer.model
        del caller_trainer.model_wrapped
        del caller_trainer
        gc.collect()

        torch.cuda.empty_cache()

    # summarizer inference
    if len(infer_samples_summarizer) != 0:
        summarizer_model, summarizer_tokenier = load_model_and_tokenizer(model_args.summarizer_model_name_or_path, model_args.summarizer_use_lora, model_args.use_logit_smooth)
        summarizer_trainer = TrainerForPred(
            model=summarizer_model, tokenizer=summarizer_tokenier, args=training_args, data_collator=data_collator
        )


        summarizer_test_dataset = InferDataset([d['model_input_for_summarizer'] for d in infer_samples_summarizer], data_collator, data_args)
        outputs,_ = summarizer_trainer.predict(summarizer_test_dataset)
        for i, o in enumerate(outputs):
            candidate = summarizer_tokenier.decode(o, skip_special_tokens=True)
            if candidate.startswith(': '):
                candidate = candidate[2:]
            if candidate.strip() in ['','.',',']:
                candidate = 'none'
            infer_samples_summarizer[i]['predictions'] = "asssitant: " + infer_samples_summarizer[i]['planner_prediction'] + "</s>conclusion: " + candidate
        summarizer_model.to('cpu')
        summarizer_trainer.model.to('cpu')
        summarizer_trainer.model_wrapped.to('cpu')
        del summarizer_model
        del summarizer_tokenier
        del summarizer_trainer.model
        del summarizer_trainer.model_wrapped
        del summarizer_trainer
        gc.collect()

        torch.cuda.empty_cache()



    final_infer_sample = infer_samples_caller + infer_samples_planner + infer_samples_summarizer
    if process_zero:
        with open(os.path.join(training_args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(final_infer_sample,f, indent=4)



if __name__ == "__main__":
    infer()
