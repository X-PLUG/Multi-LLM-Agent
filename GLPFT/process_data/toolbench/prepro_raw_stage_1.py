import json
import argparse
import os
import re
import tqdm
# this code will reorganize the training data of toolbench into the conversation form, and change to tool document schema

print("####################### PREPRO RAW DATA STAGE 1 #####################")

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="")
parser.add_argument('--output_path', type=str, default='')

args = parser.parse_args()


test_ids = []
for test_set in ['G1_category', 'G1_instruction', 'G1_tool', 'G2_category', 'G2_instruction', 'G3_instruction']:
    test_ids += list(json.load(open(f"{args.data_dir}/test_query_ids/{test_set}.json")))

print('test_ids', test_ids)

data = []
for train_set in ["G1_answer", "G2_answer", "G3_answer"]:
    for file in tqdm.tqdm(os.listdir(f"{args.data_dir}/answer/{train_set}")):
        id = file.split('_')[0]
        if id not in test_ids:
            with open(f"{args.data_dir}/answer/{train_set}/{file}") as f:
                d = json.load(f)
            instruction = d['answer_generation']['query']
            new_tools = []
            for t in d['answer_generation']['function']:
                # tool_name = change_name(standardize(t['api_name'])) + '_for_' + standardize(t['tool_name'])
                tool_name = t['name']
                if tool_name != "Finish":
                    tool_name = tool_name[-64:]
                    tool_function = t['description'][:256]
                    tool_input = {}
                    for arg_name,arg_value in t['parameters']['properties'].items():
                        arg_type = arg_value['type']
                        if 'description' in arg_value.keys():
                            tool_input[arg_name] = arg_type + ', ' + arg_value['description'][-256:]
                        else:
                            tool_input[arg_name] = arg_type
                    # print(json.dumps(t, indent=2))
                    new_tools.append({
                        'Name': tool_name,
                        'function':tool_function,
                        'input':tool_input
                    })
            data.append({
                'tools':new_tools,
                'instruction':instruction,
                'chains':d['answer_generation']['train_messages'][-1]
            })

    print(train_set,'data processed')
            
os.makedirs(args.output_path, exist_ok=True)
with open(f"{args.output_path}/raw_data_stage_1.json", 'w') as f:
    json.dump(data[:20],f, indent=2)
