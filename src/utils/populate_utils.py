from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tool_select import *
from utils.node_utils import *
import time

load_dotenv()

def createGraph(llm, tools):
    agent = llm.bind_tools(tools)
    def chatbot(state: State):
        return {"messages": [agent.invoke(state["messages"])]}
    tool_node = BasicToolNode(tools = tools)
    graph_builder = StateGraph(State)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()
    return graph


def stream_gpt_graph_updates(graph, user_input: str, sys_prompt:str=""):
    calls = []
    messages = []
    responses = []
    inputs = []
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_input})

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            if value['messages'][-1].additional_kwargs and 'tool_calls' in value['messages'][-1].additional_kwargs:
                calls.append(value['messages'][-1].additional_kwargs['tool_calls'][0]['function']['name'])
                inputs.append(value['messages'][-1].additional_kwargs['tool_calls'][0]['function']['arguments'])
            elif str(type(value['messages'][-1])) == "<class 'langchain_core.messages.tool.ToolMessage'>":
                responses.append(value['messages'][-1].content)
            else:
                messages.append(value['messages'][-1].content)

    return {"messages": messages, "calls": calls, "responses": responses, "inputs": inputs}

def stream_claude_graph_updates(graph, user_input: str, sys_prompt:str = ""):
    calls = []
    messages = []
    responses = []
    inputs = []
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_input})

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            if str(type(value['messages'][-1])) == "<class 'langchain_core.messages.tool.ToolMessage'>":
                responses.append(value['messages'][-1].content)
                calls.append(value['messages'][-1].name)
            elif value['messages'][-1].tool_calls and value['messages'][-1].tool_calls[0]['args']:
                inputs.append(value['messages'][-1].tool_calls[0]['args'])
            else:
                messages.append(value['messages'][-1].content)

    return {"messages": messages, "calls": calls, "responses": responses, "inputs": inputs}

def runPrompt(test_dict, llm_graph, llm, sys_prompt:str="", prompt_num:int=0, use_case: str=""):
    test_dict['outputs'] = {}
    test_dict['tools'] = {}
    test_dict['tool_outputs'] = {}
    test_dict['tool_inputs'] = {}
    test_dict['error'] = {}

    results_dict = {}
    try:
        with open(f"./data/{llm}-{use_case}-{prompt_num}-cache.json", 'r') as file:
            results_dict = json.load(file)
    except Exception as e:
        print(e)

    i = 0
    failures = {}
    while i < len(test_dict['prompt'].keys()):
        if 'error' in results_dict and str(i) in results_dict['error'] and results_dict['error'][str(i)] == '':
            for col in test_dict:
                test_dict[col][i] = results_dict[col][str(i)]
            print(f"using cached result for {i}")
            i += 1
        else:
            user_input = test_dict['prompt'][i]
            try:
                print(f"System Prompt {prompt_num}")
                print(f"{i} User: " + user_input)
                if 'claude' in llm:
                    events = stream_claude_graph_updates(llm_graph, user_input, sys_prompt)
                else:
                    events = stream_gpt_graph_updates(llm_graph, user_input, sys_prompt)
                test_dict['outputs'][i] = events['messages']
                test_dict['tools'][i] = events['calls']
                test_dict['tool_outputs'][i] = events['responses']
                test_dict['tool_inputs'][i] = events['inputs']
                test_dict['error'][i] = '' 
                i += 1
            except Exception as e:
                if i not in failures:
                    print(f"retrying {i}")
                    time.sleep(90)
                    failures[i] = True 
                else:
                    print("error: " + str(e))
                    test_dict['outputs'][i] = ["ERROR"]
                    test_dict['tools'][i] = ["ERROR"]
                    test_dict['tool_outputs'][i] = ["ERROR"]
                    test_dict['tool_inputs'][i] = ["ERROR"]
                    test_dict['error'][i] = str(e)
                    i += 1
        with open(f"./data/{llm}-{use_case}-{prompt_num}-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)

    try: 
        df_dict = {}
        for i in test_dict:
            df_dict[i] = list(test_dict[i].values())
        populated_df = pd.DataFrame.from_dict(df_dict)
        populated_df.to_csv(f'./data/{llm}-{use_case}-{prompt_num}-axis-dataset.csv')
        print(f'Sucessfully created {llm} with prompt {prompt_num} for {use_case} dataset')
    except:
        print(f'Error creating {llm} dataset')
        with open(f"./data/{llm}-{use_case}-{prompt_num}-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)


def runPromptWithRouting(test_dict, llm, sys_prompt:str="", prompt_num:int=0, tool_dict:dict = {}, use_case: str=""):
    test_dict['outputs'] = {}
    test_dict['tools'] = {}
    test_dict['tool_outputs'] = {}
    test_dict['tool_inputs'] = {}
    test_dict['error'] = {}

    results_dict = {}
    try:
        with open(f"./data/{llm}-{use_case}-cache.json", 'r') as file:
            results_dict = json.load(file)
    except Exception as e:
        print(e)

    i = 0
    failures = {}
    while i < len(test_dict['prompt'].keys()):
        if 'error' in results_dict and str(i) in results_dict['error'] and results_dict['error'][str(i)] == '':
            for col in test_dict:
                test_dict[col][i] = results_dict[col][str(i)]
            print(f"using cached result for {i}")
            i += 1
        else:
            user_input = test_dict['prompt'][i]
            try:
                print(f"System Prompt {prompt_num}")
                print(f"{i} User: " + user_input)
                tools = []
                sourceSet = set()
                for name in set(test_dict['tool_name'][i].split(",")):
                    if name.split("_")[0] not in sourceSet and name.split("_")[0] in tool_dict:
                        tools += tool_dict[name.split("_")[0]]
                        sourceSet.add(name.split("_")[0])
                if 'claude' in llm:
                    claude_llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", timeout=None, stop=None)
                    claude_graph = createGraph(claude_llm, tools)
                    events = stream_claude_graph_updates(claude_graph, user_input, sys_prompt)
                elif llm == 'o3-gpt':
                    o3_llm = ChatOpenAI(model="o3-mini")
                    o3_graph = createGraph(o3_llm, tools)
                    events = stream_gpt_graph_updates(o3_graph, user_input, sys_prompt)
                elif llm == 'gpt-4o':
                    gpt_llm = ChatOpenAI(model="gpt-4o")
                    gpt_graph = createGraph(gpt_llm, tools)
                    events = stream_gpt_graph_updates(gpt_graph, user_input, sys_prompt)
                else:
                    print("model not supported yet")
                    events = stream_gpt_graph_updates("placeholder", user_input, sys_prompt)
                test_dict['outputs'][i] = events['messages']
                test_dict['tools'][i] = events['calls']
                test_dict['tool_outputs'][i] = events['responses']
                test_dict['tool_inputs'][i] = events['inputs']
                test_dict['error'][i] = '' 
                i += 1
            except Exception as e:
                if i not in failures:
                    print(f"retrying {i}")
                    time.sleep(90)
                    failures[i] = True 
                else:
                    print("error: " + str(e))
                    test_dict['outputs'][i] = ["ERROR"]
                    test_dict['tools'][i] = ["ERROR"]
                    test_dict['tool_outputs'][i] = ["ERROR"]
                    test_dict['tool_inputs'][i] = ["ERROR"]
                    test_dict['error'][i] = str(e)
                    i += 1
        with open(f"./data/{llm}-{use_case}-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)

    try: 
        df_dict = {}
        for i in test_dict:
            df_dict[i] = list(test_dict[i].values())
        populated_df = pd.DataFrame.from_dict(df_dict)
        populated_df.to_csv(f'./data/{llm}-{use_case}-axis-dataset.csv')
        print(f'Sucessfully created {llm} with prompt {prompt_num} and enhanced description dataset')
    except:
        print(f'Error creating {llm} dataset')
        with open(f"./data/{llm}-{use_case}-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)

def getToolsDict(tools: list) -> dict:
    tool_dict = {}
    for func in tools:
        if func['function']['name'].split("_")[0] == 'SALESFORCE':
            tool_dict['SALESFORCE'] = tool_dict.get('SALESFORCE', [])
            tool_dict['SALESFORCE'].append(func)
        elif func['function']['name'].split("_")[0] == 'HUBSPOT':
            tool_dict['HUBSPOT'] = tool_dict.get('HUBSPOT', [])
            tool_dict['HUBSPOT'].append(func)
        elif func['function']['name'].split("_")[0] == 'NOTION':
            tool_dict['NOTION'] = tool_dict.get('NOTION', [])
            tool_dict['NOTION'].append(func)
        elif func['function']['name'].split("_")[0] == 'GOOGLE':
            tool_dict['GOOGLE'] = tool_dict.get('GOOGLE', [])
            tool_dict['GOOGLE'].append(func)
        elif func['function']['name'].split("_")[0] == 'SLACK':
            tool_dict['SLACK'] = tool_dict.get('SLACK', [])
            tool_dict['SLACK'].append(func)
        elif func['function']['name'].split("_")[0] == 'GMAIL':
            tool_dict['GMAIL'] = tool_dict.get('GMAIL', [])
            tool_dict['GMAIL'].append(func)
    return tool_dict


