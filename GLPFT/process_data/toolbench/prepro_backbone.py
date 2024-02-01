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

print("####################### PREPRO BACKBONE DATA #####################")

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

            if "Next: give up." in utter['value']:
                giveup_num += 1
                answer_span = utter['value'].replace('Next: give up.',"").strip()
                answer_span = answer_span + '\nconclusion: give up.'
                new_data.append({
                    'tools':d['tools'],
                    'history':d['conversations'][:i],
                    'input':input + ' assistant: ',
                    'target': answer_span
                })
            else:
                if d['conversations'][i+1]['from'] == 'caller':
                    answer_span = utter['value'].replace('Next: caller.',"").strip()
                    answer_span = answer_span + '\n' + d['conversations'][i+1]['value']
                    input = query_temp.replace('{history}',history)
                    new_data.append({
                        'tools':d['tools'],
                        'history':d['conversations'][:i],
                        'input':input+ ' assistant: ',
                        'target': answer_span
                    })
                elif d['conversations'][i+1]['from'] == 'conclusion':
                    finish_num += 1
                    answer_span = utter['value'].replace('Next: conclusion.',"").strip()
                    answer_span = answer_span + '\nconclusion: ' + d['conversations'][i+1]['value']
                    input = query_temp.replace('{history}',history)
                    new_data.append({
                        'tools':d['tools'],
                        'history':d['conversations'][:i],
                        'input':input+ ' assistant: ',
                        'target': answer_span
                    })
                else:
                    answer_span = utter['value']
            history += ('assistant: '+answer_span+'</s>')
        elif utter['from'] == 'user':
            history += ('user: '+utter['value']+'</s>')
        elif utter['from'] == 'observation':
            history += ('observation: '+utter['value'])
            
print(args.output_path)
print(len(new_data))
print('finish case:',finish_num)
print('give up case:',giveup_num)
print('-------------------')

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(new_data,f, indent=2)
