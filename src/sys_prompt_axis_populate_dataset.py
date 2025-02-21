from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tool_select import *
from node_utils import *
import time

load_dotenv()

test_df = pd.read_csv("./data/tool-based-test-cases.csv")
test_dict = test_df.to_dict()
tools = getTools()


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

def runPrompt(test_dict, llm_graph, llm, sys_prompt:str="", prompt_num:int=0):
    test_dict['outputs'] = {}
    test_dict['tools'] = {}
    test_dict['tool_outputs'] = {}
    test_dict['tool_inputs'] = {}
    test_dict['error'] = {}

    results_dict = {}
    try:
        with open(f"./data/{llm}-sys-prompt-{prompt_num}-cache.json", 'r') as file:
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
                if llm == 'claude':
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
        with open(f"./data/{llm}-sys-prompt-{prompt_num}-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)

    try: 
        df_dict = {}
        for i in test_dict:
            df_dict[i] = list(test_dict[i].values())
        populated_df = pd.DataFrame.from_dict(df_dict)
        populated_df.to_csv(f'./data/{llm}-sys-prompt-{prompt_num}-axis-dataset.csv')
        print(f'Sucessfully created {llm} with prompt {prompt_num} dataset')
    except:
        print(f'Error creating {llm} dataset')
        with open(f"./data/{llm}-sys-prompt-{prompt_num}-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)

# claude_llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", timeout=None, stop=None)
# claude_graph = createGraph(claude_llm, tools)

sys_prompt_1 = '''
    You are a helpful assistant that helps users perform actions in 3rd-party applications. 
    Users will ask to create records and search for records in CRMs like Salesforce and Hubspot. 
    Users will also ask you to send messages/emails and search for messages/emails in messaging 
    platforms like Slack and Gmail. Lastly, users will ask to search through pages and documents like 
    in Google Drive and Notion. Use tools provided to help you with these tasks.
    '''
sys_prompt_2 = '''
    You are a helpful assistant that helps users perform actions in 3rd-party applications. 
    Users will ask to create records and search for records in CRMs like Salesforce and Hubspot, 
    send and search messages in Gmail and Slack, and search for documents and pages in Notion and 
    Google Drive. Use tools provided to help you with these tasks.
    
    Some Rules:
    * If a user mentions multiple data sources, use tools to search and/or take action across each data source.
    * If you use a tool that searches for information, but fails, try using a slightly different filter with your next best guess. For example, if a user asks to find Mistral.ai in Salesforce, but you can’t find it by using Name=Mistral.ai, then try searching with name=Mistral
    * If a tool continues to fail even after trying different inputs, try a different tool. For example when searching Salesforce contacts, 
    if SALESFORCE_WRITE_SOQL_QUERY fails, try using SALESFORCE_SEARCH_RECORDS_CONTACT
    '''

gpt_llm = ChatOpenAI(model="o3-mini")
o3_graph = createGraph(gpt_llm, tools)

runPrompt(test_dict, o3_graph, "o3-gpt", sys_prompt_1, 1)
runPrompt(test_dict, o3_graph, "o3-gpt", sys_prompt_2, 2)
