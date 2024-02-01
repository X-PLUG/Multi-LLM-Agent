import json
import argparse
import os
from utils.prompt_lib import prompt_dict

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--prompt_type', type=str)


args = parser.parse_args()

prompt_temp = prompt_dict[args.prompt_type]

print("####################### PREPRO SUMMARIZER DATA #####################")


def nested_load_data(data_path):
    if os.path.isdir(data_path):
        data =[]
        for f in os.listdir(data_path):
            temp_train = nested_load_data(os.path.join(data_path, f))
            data += temp_train
        return data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        temp_data =  json.load(open(data_path, "r"))
        return temp_data
    else:
        return []


data_paths = args.input_path.split(',')
data= []
for p in data_paths:
    train_temp = nested_load_data(p)
    data += train_temp

if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))



new_data = []
for d in data:
    tool_docs = ""
    for t in d['tools']:
        tool_docs += json.dumps(t) + '\n'    
    tool_names = ', '.join([t['Name'] for t in d['tools']])
    query_temp = prompt_temp.replace('{doc}', tool_docs).replace('{tool_names}',tool_names)
    
    history = ""
    for i in range(len(d['conversations'])):
        utter = d['conversations'][i]
        if utter['from'] == 'assistant':
            history += ('assistant: '+utter['value']+'</s>')
        elif utter['from'] == 'user':
            history += ('user: '+utter['value']+'</s>')
        elif utter['from'] == 'observation':
            history += ('observation: '+utter['value'])
        elif utter['from'] == 'caller':
            history += ('caller: '+utter['value']+'</s>')
        elif utter['from'] == 'conclusion' and utter['value'] != 'none':
            input = query_temp.replace('{history}',history)

            new_data.append({
                'tools':d['tools'],
                'history':d['conversations'][:i],
                'input':input+" conclusion: ",
                'target': utter['value']
            })
            history += ('conclusion: '+utter['value']+'</s>')

print(len(new_data))

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(new_data,f, indent=2)
