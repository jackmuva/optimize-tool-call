from dotenv import load_dotenv
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
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
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

user_input = "what custom tools do you have?"
print("User: " + user_input)
stream_graph_updates(user_input)

for i in range(0, len(test_dict['prompt'].keys())):
    user_input = test_dict['prompt'][i]
    print("User: " + user_input)
    stream_graph_updates(user_input)
    break
