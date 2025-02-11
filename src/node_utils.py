from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
import json
import requests
from dotenv import load_dotenv
import jwt
import os
import time
from langgraph.graph import END

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool['function']['name']: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            currentTime = time.time()
            encoded_jwt = jwt.encode({
                "sub": "jack.mu@useparagon.com",
                "iat": currentTime,
                "exp": currentTime + (60 * 60 * 24 * 7)
            }, os.environ['PARAGON_SIGNING_KEY'].replace("\\n", "\n"), algorithm="RS256")

            run_actions_body = {
                "action": tool_call["name"],
                "parameters": tool_call["args"]
            }
            print(run_actions_body)
            response = requests.post("https://actionkit.useparagon.com/projects/" + os.environ['PARAGON_PROJECT_ID'] + "/actions",
                             headers={"Authorization": "Bearer " + encoded_jwt}, json=run_actions_body)
            tool_result = response.json()

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

