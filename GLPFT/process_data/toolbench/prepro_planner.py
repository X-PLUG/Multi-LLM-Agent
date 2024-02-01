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

print("####################### PREPRO PLANNER DATA #####################")

def str2bool(text):
    if text.lower() in ['true', 't', '1']:
        return True
    else:
        return False


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


finish_num = 0
giveup_num = 0
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
            input = query_temp.replace('{history}',history)
            if  "Next: give up." in utter['value'] or d['conversations'][i+1]['from'] in ['conclusion', 'caller'] :
                answer = utter['value']
                if i < len(d['conversations']) - 1 and d['conversations'][i+1]['from'] == 'caller':
                    if 'invalid_hallucination_function_name' in d['conversations'][i+1]['value']:
                        pass
                    else:
                        new_data.append({
                            'tools':d['tools'],
                            'history':d['conversations'][:i],
                            'input':input + ' assistant: ',
                            'target': utter['value']
                        })
                else:
                    if "Next: give up." in utter['value']:
                        giveup_num += 1
                    elif d['conversations'][i+1]['from'] == 'conclusion':
                        finish_num += 1
                    new_data.append({
                        'tools':d['tools'],
                        'history':d['conversations'][:i],
                        'input':input + ' assistant: ',
                        'target': utter['value']
                    })
            history += ('assistant: '+utter['value']+'</s>')
        elif utter['from'] == 'user':
            history += ('user: '+utter['value']+'</s>')
        elif utter['from'] == 'observation':
            history += ('observation: '+utter['value'])
        elif utter['from'] == 'caller':
            history += ('caller: '+utter['value']+'</s>')
        elif utter['from'] == 'conclusion':
            history += ('conclusion: '+utter['value']+'</s>')

print(args.output_path)
print(len(new_data))
print('finish case:',finish_num)
print('give up case:',giveup_num)
print('-------------------')

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(new_data,f, indent=2)
