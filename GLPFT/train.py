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

from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional,List
import os

import numpy as np
import random
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from utils.tool_conversation import SeparatorStyle
from utils.model_adapter import get_conversation_template
from transformers.utils import is_torch_available, is_tf_available
from utils.prompt_lib import prompt_dict
from utils.llama_condense_monkey_patch import replace_llama_with_condense

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
    max_num_sample_per_data: int = field(
        default=0, metadata={"help": "default number of samples for each dataset"}
    )
    max_num_sample_ratio: float = field(
        default=0, metadata={"help": "default number of samples for each dataset"}
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
    source_model_max_length: int = field(
        default=0,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # conv = get_conversation_template("collab_agent_v3")
    # roles = {"user": conv.roles[0], "assistant": conv.roles[1], "caller": conv.roles[2], 'observation':conv.roles[3], 'conclusion': conv.roles[4]}

    # Apply prompt templates
    datas = []
    for i, source in enumerate(sources):
        input = 'system: ' + source['input']
        input_ids = tokenizer.encode(input) #list
        target = source['target'] + "</s>"
        target_ids = tokenizer.encode(target)[1:]
        instruction_len = len(input_ids)
        answer_len = len(target_ids)
        input_ids = input_ids + target_ids
        if len(input_ids) < tokenizer.model_max_length:
            input_ids += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(input_ids))
            input_ids = torch.LongTensor(input_ids)
            labels = input_ids.clone()
            labels[:instruction_len] = IGNORE_TOKEN_ID
            labels[instruction_len + answer_len:] = IGNORE_TOKEN_ID
            # print(input_ids.detach().cpu().numpy().tolist())
            # print('---------------------')
            # print(targets.detach().cpu().numpy().tolist())

        else:
            input_ids = torch.LongTensor(input_ids)
            labels = input_ids.clone()
            labels[:instruction_len] = IGNORE_TOKEN_ID
            input_ids = input_ids[-tokenizer.model_max_length:]
            labels = labels[-tokenizer.model_max_length:]
        input_ids.requires_grad=False
        labels.requires_grad=False
        datas.append(dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        ))

    return datas

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data,tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.data_dict = data_dict
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.data_dict[i]['input_ids'],
            labels=self.data_dict[i]['labels'],
            attention_mask=self.data_dict[i]['attention_mask'],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if 'conversations' in self.raw_data[i].keys():
            ret = preprocess([self.raw_data[i]], self.tokenizer)
        else:
            ret = preprocess([self.raw_data[i]], self.tokenizer)

        ret = dict(
            input_ids=ret[0]["input_ids"],
            labels=ret[0]["labels"],
            attention_mask=ret[0]["attention_mask"],
        )
        self.cached_data_dict[i] = ret

        return ret

def nested_load_data(data_path, max_num_sample, max_num_sample_ratio):
    train_raw_data = []
    dev_raw_data = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            temp_train, temp_dev = nested_load_data(os.path.join(data_path, f), max_num_sample)
            train_raw_data += temp_train
            dev_raw_data += temp_dev
        return train_raw_data, dev_raw_data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        rank0_print("Load data from",data_path)
        temp_data =  json.load(open(data_path, "r"))
        random.shuffle(temp_data)
        if max_num_sample != 0:
            temp_data = temp_data[:max_num_sample]
        elif max_num_sample_ratio != 0:
            temp_data = temp_data[:int(len(temp_data) * max_num_sample_ratio)]
        split = int(len(temp_data) * 0.98)
        train_raw_data = temp_data[:split]
        dev_raw_data = temp_data[split:]
        return train_raw_data, dev_raw_data
    else:
        return [],[]

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    data_paths = data_args.data_path.split(',')
    train_raw_data = []
    eval_raw_data = []
    for p in data_paths:
        train_temp, dev_temp = nested_load_data(p, data_args.max_num_sample_per_data, data_args.max_num_sample_ratio)
        train_raw_data += train_temp
        eval_raw_data += dev_temp


    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    # prompt_temp = prompt_dict["v7_" + data_args.prompt_type]
    # gorilla_prompt_temp = prompt_dict["v7_gorilla_" + data_args.prompt_type]
    

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    if is_torch_available():
        torch.manual_seed(training_args.seed)
        torch.cuda.manual_seed_all(training_args.seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        tf.random.set_seed(training_args.seed)

    if training_args.source_model_max_length != 0 and training_args.source_model_max_length < training_args.model_max_length:
        condense_ratio = int(training_args.model_max_length/training_args.source_model_max_length)
        # ratio = N means the sequence length is expanded by N, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
        replace_llama_with_condense(ratio=condense_ratio)

    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if trainer.is_local_process_zero():
        model.save_pretrained(training_args.output_dir)
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)




if __name__ == "__main__":
    train()
