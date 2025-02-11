from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from tool_select import *
from node_utils import *

load_dotenv()

test_df = pd.read_csv("./data/tool-based-test-cases.csv")
test_dict = test_df.to_dict()

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o")
tools = getTools()
agent = llm.bind_tools(tools)
def chatbot(state: State):
    return {"messages": [agent.invoke(state["messages"])]}
tool_node = BasicToolNode(tools = tools)

graph_builder.add_node("tools", tool_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    calls = []
    messages = []
    responses = []
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if value['messages'][-1].additional_kwargs and 'tool_calls' in value['messages'][-1].additional_kwargs:
                calls.append(value['messages'][-1].additional_kwargs['tool_calls'])
            elif str(type(value['messages'][-1])) == "<class 'langchain_core.messages.tool.ToolMessage'>":
                responses.append(value['messages'][-1].content)
            else:
                messages.append(value['messages'][-1].content)

    return {"messages": messages, "calls": calls, "responses": responses}

test_dict['outputs'] = {}
test_dict['tools'] = {}
test_dict['tool_outputs'] = {}

for i in range(0, len(test_dict['prompt'].keys())):
    user_input = test_dict['prompt'][i]
    try:
        print("User: " + user_input)
        events = stream_graph_updates(user_input)
        test_dict['outputs'][i] = events['messages']
        test_dict['tools'][i] = events['calls']
        test_dict['tool_outputs'][i] = events['responses']
    except:
        print("error: " + user_input)

df_dict = {}
for i in test_dict:
    df_dict[i] = list(test_dict[i].values())
populated_df = pd.DataFrame.from_dict(df_dict)
populated_df.to_csv('./data/llm-axis-dataset.csv')
