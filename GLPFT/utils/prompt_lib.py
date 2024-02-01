prompt_dict = {
"toolbench_backbone":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You should answer the action of next step in the following format:
The thought to solve the question,
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.
Once you think the question is finished, output conclusion: the final answer of the question or give up if you think you cannot answer this question.
""",

"toolbench_planner":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You are the assistant to plan what to do next and whether is caller's or conclusion's turn to answer.
Answer with a following format:
The thought of the next step, followed by Next: caller or conclusion or give up.""",


"toolbench_caller":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
The thought of this step is:
{thought}
Base on the thought make an api call in the following format:
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.""",

"toolbench_summarizer": """Make a conclusion based on the conversation history:
{history}""",

"toolalpaca_backbone":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You should answer the action of next step in the following format:
The thought to solve the question,
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.
Once you think the question is finished, output conclusion: the final answer of the question.
""",

"toolalpaca_planner":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You are the assistant to plan what to do next and whether is caller's or conclusion's turn to answer.
Answer with a following format:
The thought of the next step, followed by Next: caller or conclusion.""",


"toolalpaca_caller":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
The thought of this step is:
{thought}
Base on the thought make an api call in the following format:
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.""",

"toolalpaca_summarizer": """Make a conclusion based on the conversation history:
{history}""",

"toolbench_toolllama":"""You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:
Thought:
Action
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say \"I give up and restart\".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:
{tool_doc}
Specifically, you have access to the following APIs: {api_list}""",
}