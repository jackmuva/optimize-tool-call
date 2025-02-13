from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

def completeJsonFormat(parseList:list) -> list:
    temp = []
    for s in parseList:
        if len(s) > 0 and s[-1] != "}":
            temp.append(s + "}")
        else:
            temp.append(s)
    return temp


def formatToolCalls(index: int, output_dict: dict) -> list:
    res = []
    toolMetadata = {}
    try:
        with open(f"./data/tool-metadata.json", 'r') as file:
            toolMetadata = json.load(file)
    except Exception as e:
        print(e)

    parsedToolNames = output_dict['tools'][index][2:len(output_dict['tools'][index]) - 2].split(",")
    print(parsedToolNames)
    parsedToolOutputs = output_dict['tool_outputs'][index][2:len(output_dict['tool_outputs'][index]) - 2].split("','")
    parsedToolInputs =  output_dict['tool_inputs'][index][2:len(output_dict['tool_inputs'][index]) - 2].split("','")

    parsedToolOutputs = completeJsonFormat(parsedToolOutputs) 
    parsedToolInputs = completeJsonFormat(parsedToolInputs) 

    for i, toolName in enumerate(parsedToolNames):
        if toolName == '':
            continue
        description = ""
        for source in toolMetadata['actions']:
            actions = toolMetadata['actions'][source]
            for action in actions:
                if action['function']['name'] == toolName:
                    description = action['function']['description']
        toolCall = ToolCall(name=toolName, description=description, input_parameters=json.loads(parsedToolInputs[i]), output=str(json.loads(parsedToolOutputs[i])))
        res.append(toolCall)
    print(res)
    return res

def formatExpectedToolCalls(index: int, output_dict: dict) -> list:
    toolCalls = []
    expectedTools = output_dict['tool_name'][index].split(",")
    for name in expectedTools:
        toolCall = ToolCall(name = name, input_parameters={})
        toolCalls.append(toolCall)
    return toolCalls



def evaluateTestCases(prefix):
    output_dict = pd.read_csv(f'./data/{prefix}-llm-axis-dataset.csv').to_dict()

    test_cases = []

    for index in output_dict['prompt'].keys():
        test_case = LLMTestCase(
            input=output_dict['prompt'][index],
            actual_output=output_dict['outputs'][index][2:len(output_dict['outputs'][index]) - 3],
            tools_called=formatToolCalls(index, output_dict),
            expected_tools=formatExpectedToolCalls(index, output_dict)
        ) 
        test_cases.append(test_case)
        break
    print(test_cases)

    tool_correctness= ToolCorrectnessMetric(evaluation_params = [ToolCallParams.TOOL])
    task_completion = TaskCompletionMetric(model="gpt-4o-mini")

    evaluation = evaluate(test_cases, [tool_correctness, task_completion],max_concurrent=1, ignore_errors=True, run_async=False, throttle_value=10, use_cache=True) 
    with open(f"./results/{prefix}-llm-results.json", "w") as f:
         json.dump(evaluation.model_dump(), f)

evaluateTestCases("gpt")
# evaluateTestCases("claude")
