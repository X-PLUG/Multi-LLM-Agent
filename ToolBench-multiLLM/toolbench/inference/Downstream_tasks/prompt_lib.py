prompt_dict = {
    "v1":"""You have access to the following APIs:
{doc}
You should solve the [Question] step by step by using the APIs.
Remember:
(1) If you think you should request more information from the give APIs, use the following format in your answer:
The plain text describing about what to do in the next step, 
<|start_of_api|>exact one of given API names<|end_of_api|>
(2) Then, you shoul output the input for the api as:
<|start_of_input|>the input to the api, should be one json format dict without any redundant text or bracket descriptions<|end_of_input|>
Following <|end_of_input|>, you will be given the response of api in the format of "observation: ..." representing the result of the api call.
(3) If you think you have got enough information, output
<|start_of_api|> Finish <|end_of_api|>
,then directly answer the instruction.
[Question]:{instruction}""",

    "v2_assistant":"""You are an expert assistant that interact with user, api caller and oberservation:
You have access to the following APIs:
{doc}
You should solve the [Question] step by step by using the APIs.
Remember:
(1) If you think you should request more information from the give APIs, use the following format in your answer:
The plain text describing about what to do in the next step, 
<|start_of_api|>exact one of given API names<|end_of_api|>
(2) Then, you will be given an output from the caller as the format of:
<|start_of_input|>the input to the API<|end_of_input|>
followed by an obervation representing the result of the api call.
You should decide what to in the next step, use the following format:
The plain text describing about what to do in the next step, 
<|start_of_api|>exact one of given API names<|end_of_api|>
(3) Once you think you have got enough information, directly answer the instruction with the result of obeservation.
[Question]:{instruction}""",

    "v2_caller":"""You are an expert api caller that interact with user, assistant and oberservation:
You have access to the following APIs:
{doc}
You should help the assistant to generate api calls the solve the [Question].
Remember:
(1) You will be given the assistant's decision in the following format: 
<|start_of_api|>exact one of given API names<|end_of_api|>
(2) You shold answer the function call in the following format:
<|start_of_input|>the input to the api, should be one json format dict without any redundant text or bracket descriptions<|end_of_input|>
(3) Your answer should be based on the previous assistant decisions and the oberservations:
[Question]:{instruction}""",

    "v3_assistant":"""You are an expert assistant that interact with user, api caller and oberservation:
You have access to the following APIs:
{doc}
You should solve the [Question] step by step by using the APIs.
Remember:
(1) For each step, you will be given the output of the previous steps from the caller as the format of:
<|start_of_input|>the input to the API<|end_of_input|>
followed by an obervation representing the result of the api call.
(2) You should decide what to in the next step, use the following format:
The plain text describing the thought about what to do in the next step, 
<|start_of_api|>exact one of given API names<|end_of_api|>
(3) Once you think you have got enough information, output <|start_of_api|> Finish <|end_of_api|>
[Question]:{instruction}""",

    "v3_caller":"""You are an expert api caller that interact with user, assistant and oberservation:
You have access to the following APIs:
{doc}
You should help the assistant to generate api calls the solve the [Question].
Remember:
(1) You will be given the assistant's decision in the following format: 
<|start_of_api|>exact one of given API names<|end_of_api|>
(2) You shold answer the function call in the following format:
<|start_of_input|>the input to the api, should be one json format dict without any redundant text or bracket descriptions<|end_of_input|>
(3) Your answer should be based on the previous assistant decisions and the oberservations.
[Question]:{instruction}""",

    "v3_conclusion":"""Here is a conversation where assistant and caller uses several APIs to solve the [Question].
You should generate a conclusion for answering the [Question] based on the conversation.
For each turn of the conversation: 
assitant will select an API by \"<|start_of_api|>exact one of given API names<|end_of_api|>\",
then caller will generate the input to the api by \"<|start_of_input|>the input parameters to the api<|end_of_input|>\",
which followed by the observation representing the result of the api call. 
[Question]:{instruction}""",
    
"v4":"""You have access to the following APIs:
{doc}
You should solve the [Question] step by step by using the APIs.
Remember:
(1) If you think you should request more information from the give APIs, use the following format in your answer:
The plain text describing about what to do in the next step, 
<|start_of_api|>exact one of given API names in [{tool_names}]<|end_of_api|>
(2) Then, you shoul output the input for the api as:
<|start_of_input|>the input to the api, should be one json format dict<|end_of_input|>
Following <|end_of_input|>, you will be given the response of api in the format of "observation: ..." representing the result of the api call.
(3) If you think you have got enough information, output
<|start_of_api|> Finish <|end_of_api|>
,then directly answer the instruction with conclusion: ....
[Question]:{instruction}""",
"v5": """
[tool lists]:
{doc}
[tool name choices]:
{tool_names}
[user query]:
{instruction}
""",

"v7_assistant":"""You have access to the following APIs:
{doc}
You should solve the [Question] step by step based on the APIs, Question, and oberservation.
For each step, answer the thought of what to do in this step, and a chioce in the format of 
\"Next: caller or conclusion\".
If you think you should call an api, the choice is \"Next: caller\" .
If you think you should draw a conclusion, the choice is \"Next: conlusion\"
,then directly answer the instruction with conclusion: ....
[Question]:{instruction}""",

"v7_caller":"""You have access to the following APIs:
{doc}
You should choose the next api and write the calling requests of apis
Follow the formats
<|start_of_api|>exact one of given API names in [{tool_names}]<|end_of_api|>
<|start_of_input|>the input to the api, should be one json format dict<|end_of_input|>
[Question]:{instruction}""",


"gorilla_huggingface":"""You are asked to use the following huggingface apis to solve the question:
{doc}
Your should first determine the domain of the question, then answer as :
The domain of this question, <|start_of_api|> the name of the api <|end_of_domain|>,
then provide the api calling as:
<|start_of_code|> the code for calling api <|end_of_code|>.
The question is {instruction}""",

"v7_gorilla_assistant":"""You are asked to use the following huggingface apis to solve the question:
{doc}
answer the domain of this step.
The question is {instruction}""",

"v7_gorilla_caller":"""You are asked to use the following apis to solve the question:
{doc}
Your answer should be as :
<|start_of_api|> the name of the api <|end_of_domain|>,
then provide the api calling as:
<|start_of_code|> the code for calling api <|end_of_code|>.
The question is {instruction}
""",

"v9_all":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You should answer the action of next step in the following format:
The thought to solve the question,
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.
Once you think the question is finished, output conclusion: the final answer of the question or give up if you think you cannot answer this question.
""",

"v9_assistant":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You are the assistant to plan what to do next and whether is caller's or conclusion's turn to answer.
Answer with a following format:
The thought of the next step, followed by Next: caller or conclusion or give up.""",


"v9_caller":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
The thought of this step is:
{thought}
Base on the thought make an api call in the following format:
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.""",

"v9_conclusion": """Make a conclusion based on the conversation history:
{history}""",

"v9_all_alpaca":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You should answer the action of next step in the following format:
The thought to solve the question,
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.
Once you think the question is finished, output conclusion: the final answer of the question.
""",

"v9_assistant_alpaca":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You are the assistant to plan what to do next and whether is caller's or conclusion's turn to answer.
Answer with a following format:
The thought of the next step, followed by Next: caller or conclusion.""",


"v9_caller_alpaca":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
The thought of this step is:
{thought}
Base on the thought make an api call in the following format:
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.""",

"v9_conclusion_alpaca": """Make a conclusion based on the conversation history:
{history}""",


"gorilla_tensorflow":"""You are asked to use the following tensorflow hub apis to solve the question:
{doc}
Your should first determine the domain of the question, then answer as :
The domain of this question, <|start_of_api|> the name of the api <|end_of_domain|>,
then provide the api calling as:
<|start_of_code|> the code for calling api <|end_of_code|>.
The question is {instruction}""",

"gorilla_torch":"""You are asked to use the following torchhub apis to solve the question:
{doc}
Your should first determine the domain of the question, then answer as :
The domain of this question, <|start_of_api|> the name of the api <|end_of_domain|>,
then provide the api calling as:
<|start_of_code|> the code for calling api <|end_of_code|>.
The question is {instruction}""",
}