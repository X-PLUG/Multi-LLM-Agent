import json
import argparse
import os

print("####################### PREPRO RAW DATA STAGE 2 #####################")

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--output_path', type=str, default='')


args = parser.parse_args()


with open(args.input_path, encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for d in data:
    conversations = [{
        'from':'user',
        'value':d['instruction']
    }]
    for c in d['chains'][2:]:
        if c['role'] == 'assistant':
            answer_span = ""
            if c['content'] is not None:
                answer_span  += c['content'] + '\n'
            if 'function_call' in c.keys():
                if c['function_call']['name'] == "Finish":
                    give_up = False
                    try:
                        if not isinstance(c['function_call']['arguments'],dict):
                            final_answer_json = json.loads(c['function_call']['arguments'])
                        else:
                            final_answer_json = c['function_call']['arguments']
                        # print(c['function_call']['arguments'])
                        if final_answer_json['return_type'] == 'give_up_and_restart':
                            give_up = True
                        else:
                            if "final_answer" in final_answer_json.keys():
                                final_answer = final_answer_json['final_answer']
                            else:
                                final_answer = "none"
                    # print()
                    except:
                        if 'give_up_and_restart' in c['function_call']['arguments']:
                            give_up = True
                        else:
                            if c['function_call']['arguments'].find('\"final_answer\": ') != -1:
                                begin_index = c['function_call']['arguments'].index('\"final_answer\": ') + len('\"final_answer\": ')
                                # print(c['function_call']['arguments'])
                                # print('--------------------')
                                final_answer = c['function_call']['arguments'][begin_index:]
                                final_answer = c['function_call']['arguments'][begin_index:].strip().strip('}').strip('\"')
                                # print(final_answer)
                                # print('=================================')
                            else:
                                if c['content'] is not None:
                                    final_answer = c['content']
                                else:
                                    final_answer = 'none'
                    if give_up:
                        conversations.append({
                            'from':'assistant',
                            'value':answer_span +  ' Next: give up.'
                        })
                    else:
                        conversations.append({
                            'from': 'assistant',
                            'value': answer_span + " Next: conclusion."
                        })
                        conversations.append({
                            'from': 'conclusion',
                            'value': final_answer
                        })
                else:
                    conversations.append({
                        'from': 'assistant',
                        'value': answer_span + " Next: caller."
                    })
                    action = c['function_call']['name']
                    action_input = c['function_call']['arguments']
                    conversations.append({
                        'from':'caller',
                        'value': "Action: " + action + "\nAction Input: " +action_input
                    })
                    # answer_span += '<|start_of_api|> ' + c['function_call']['name'] + ' <|end_of_api|>'
        elif c['role'] == 'user':
            conversations.append({
                'from':'user',
                'value':c['content']
            })
        elif c['role'] == 'function':
            ori_obs = c['content']
            try:
                ori_obs = json.loads(ori_obs)
                obs_res = ori_obs['response']
            except:
                if ori_obs.find('\"response\":') != -1:
                    begin_index = ori_obs.index('\"response\":') + len('\"response\":')
                    obs_res = ori_obs[begin_index:]
                else:
                    obs_res = ""
            conversations.append({
                'from': 'observation',
                'value': obs_res
            })
    new_data.append({
        'tools':d['tools'],
        'instruction':d['instruction'],
        'conversations':conversations
    })


print(len(new_data))
os.makedirs(args.output_path, exist_ok=True)

with open(f"{args.output_path}/raw_data_stage_2.json",'w',encoding='utf-8') as f:
    json.dump(new_data,f, indent=2)
