from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tool_select import *
from node_utils import *

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


def stream_gpt_graph_updates(graph, user_input: str):
    calls = []
    messages = []
    responses = []
    inputs = []
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if value['messages'][-1].additional_kwargs and 'tool_calls' in value['messages'][-1].additional_kwargs:
                calls.append(value['messages'][-1].additional_kwargs['tool_calls'][0]['function']['name'])
                inputs.append(value['messages'][-1].additional_kwargs['tool_calls'][0]['function']['arguments'])
            elif str(type(value['messages'][-1])) == "<class 'langchain_core.messages.tool.ToolMessage'>":
                responses.append(value['messages'][-1].content)
            else:
                messages.append(value['messages'][-1].content)

    return {"messages": messages, "calls": calls, "responses": responses, "inputs": inputs}

def stream_claude_graph_updates(graph, user_input: str):
    calls = []
    messages = []
    responses = []
    inputs = []
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if str(type(value['messages'][-1])) == "<class 'langchain_core.messages.tool.ToolMessage'>":
                responses.append(value['messages'][-1].content)
                calls.append(value['messages'][-1].name)
            elif value['messages'][-1].tool_calls and value['messages'][-1].tool_calls[0]['args']:
                inputs.append(value['messages'][-1].tool_calls[0]['args'])
            else:
                messages.append(value['messages'][-1].content)

    return {"messages": messages, "calls": calls, "responses": responses, "inputs": inputs}

def runPrompt(test_dict, llm_graph, prefix):
    test_dict['outputs'] = {}
    test_dict['tools'] = {}
    test_dict['tool_outputs'] = {}

    for i in range(0, len(test_dict['prompt'].keys())):
        user_input = test_dict['prompt'][i]
        try:
            print("User: " + user_input)
            if prefix == 'claude':
                events = stream_claude_graph_updates(llm_graph, user_input)
            else:
                events = stream_gpt_graph_updates(llm_graph, user_input)
            print(events)
            test_dict['outputs'][i] = events['messages']
            test_dict['tools'][i] = events['calls']
            test_dict['tool_outputs'][i] = events['responses']
        except Exception as e:
            test_dict['outputs'][i] = ["ERROR"]
            test_dict['tools'][i] = ["ERROR"]
            test_dict['tool_outputs'][i] = ["ERROR"]
            print("error: " + str(e))

    try: 
        df_dict = {}
        for i in test_dict:
            df_dict[i] = list(test_dict[i].values())
        populated_df = pd.DataFrame.from_dict(df_dict)
        populated_df.to_csv(f'./data/{prefix}-llm-axis-dataset.csv')
        print(f'Sucessfully created {prefix} dataset')
    except:
        print(f'Error creating {prefix} dataset')
        with open(f"./data/{prefix}-llm-cache.json", "w") as outfile: 
            json.dump(test_dict, outfile)

gpt_llm = ChatOpenAI(model="gpt-4o")
gpt_graph = createGraph(gpt_llm, tools)
runPrompt(test_dict, gpt_graph, "gpt")

claude_llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", timeout=None, stop=None)
claude_graph = createGraph(claude_llm, tools)
runPrompt(test_dict, claude_graph, "claude")
