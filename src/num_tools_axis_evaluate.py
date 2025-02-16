from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from dotenv import load_dotenv
import json
import pandas as pd
import re

load_dotenv()

def completeJsonFormat(parseList:list, prefix: str) -> list:
    res = []
    if prefix == 'gpt':
        for s in parseList:
            new = s.replace("\\'", "'").replace('\\\\"', "'")
            if s == 'ERROR':
                res.append('{"error": "request too large"}')
            else:
                res.append(new)
    elif prefix == 'claude':
        for s in parseList:
            new = re.sub(r' = \"(.*?)\"', r" = '\1'", s).replace('"{', '{').replace('}"', '}').replace('\\"', "'")\
                    .replace('True', '"True"').replace('False', '"False"').replace('="text/csv"', "='text/csv'").replace('\\', '')\
                    .replace('I"m', "I'm").replace('I"d', "I'd").replace('you"d', "you'd").replace("{{settings.WorkspaceMember}}", '"{{settings.WorkspaceMember}}"')\
                    .replace('you"re', "you're").replace('{{Your Gmail address}}', '"{{Your Gmail address}}"').replace('"11c3VDnHiyDDvOqBGJZvvp02EdourQjTB"', "'11c3VDnHiyDDvOqBGJZvvp02EdourQjTB'")\
                    .replace('"csv"', "'csv'").replace('other"s', "other's").replace('Here"s', "Here's").replace('here"s', "here's")\
                    .replace("'11c3VDnHiyDDvOqBGJZvvp02EdourQjTB'", '"11c3VDnHiyDDvOqBGJZvvp02EdourQjTB"')\
                    .replace("[n","[").replace("]n", "]").replace("{n", "{").replace("}n", "}").replace(",n", ",").replace('"[', '[').replace(']"', ']')\
                    .replace('"["j', '"[\'j').replace('com"]"', 'com\']"')
            new = re.sub(r' LIKE \"(.*?)\"', r" = '\1'", new)
            new = re.sub(r' CONTAINS \"(.*?)\"', r" = '\1'", new)
            new = re.sub(r'=\"(.*?)\"', r" = '\1'", new)
            if s == 'ERROR' or s == '"ERROR"':
                res.append('{"error": "request too large"}')
            elif len(new) > 0 and new[0] == '[' and new[-1] == ']':
                new = '{"results":' + new + "}"
                res.append(new)
            elif s == '' or not s:
                res.append('{}')
            else:
                res.append(new)

    return res


def formatToolCalls(index: int, output_dict: dict, prefix: str) -> list:
    res = []
    toolMetadata = {}
    try:
        with open(f"./data/tool-metadata.json", 'r') as file:
            toolMetadata = json.load(file)
    except Exception as e:
        print(e)

    parsedToolNames = output_dict['tools'][index][1:len(output_dict['tools'][index]) - 1].replace("'","").split(",")
    parsedToolInputs = None
    parsedToolOutputs = None
    if prefix == 'gpt':
        parsedToolOutputs = output_dict['tool_outputs'][index][2:len(output_dict['tool_outputs'][index]) - 2].replace("', '", "','").split("','")
        parsedToolInputs =  output_dict['tool_inputs'][index][2:len(output_dict['tool_inputs'][index]) - 2].replace("', '", "','").split("','")
    elif prefix == 'claude':
        parsedToolOutputs = output_dict['tool_outputs'][index][2:len(output_dict['tool_outputs'][index]) - 2].replace("'", '"').replace("},{", "}||{").replace('}", "{', "}||{").replace('}", "[', "}||[").split("||")
        parsedToolInputs =  output_dict['tool_inputs'][index][1:len(output_dict['tool_inputs'][index]) - 1].replace("'", '"').replace("},{", "}||{").replace("}, {", "}||{").split("||")

    if parsedToolInputs and parsedToolOutputs:
        parsedToolOutputs = completeJsonFormat(parsedToolOutputs, prefix) 
        parsedToolInputs = completeJsonFormat(parsedToolInputs, prefix) 
    else:
        return res

    for i, toolName in enumerate(parsedToolNames):
        if toolName == '':
            continue
        description = ""
        for source in toolMetadata['actions']:
            actions = toolMetadata['actions'][source]
            for action in actions:
                if action['function']['name'] == toolName:
                    description = action['function']['description']
        #quick and dirty fix; not parsing inputs and outputs 100% for Claude :(
        if i >= len(parsedToolInputs):
            parsedToolInputs.append("{}")
        if i >= len(parsedToolOutputs):
            parsedToolOutputs.append("{}")

        try:
            json.loads(parsedToolInputs[i])
        except:
            with open('./error.txt', 'w') as file:
                file.write(parsedToolInputs[i])
            print("ERROR")

        toolCall = ToolCall(name=toolName, description=description, input_parameters=json.loads(parsedToolInputs[i]), output=str(parsedToolOutputs[i]))
        res.append(toolCall)
    return res

def formatExpectedToolCalls(index: int, output_dict: dict) -> list:
    toolCalls = []
    expectedTools = output_dict['tool_name'][index].split(",")
    for name in expectedTools:
        toolCall = ToolCall(name = name, input_parameters={})
        toolCalls.append(toolCall)
    return toolCalls



def evaluateTestCases(prefix):
    output_dict = pd.read_csv(f'./data/{prefix}-num-tools-axis-dataset.csv').to_dict()

    test_cases = []

    for index in output_dict['prompt'].keys():
        test_case = LLMTestCase(
            input=output_dict['prompt'][index],
            actual_output=output_dict['outputs'][index][2:len(output_dict['outputs'][index]) - 3],
            tools_called=formatToolCalls(index, output_dict, prefix),
            expected_tools=formatExpectedToolCalls(index, output_dict)
        ) 
        test_cases.append(test_case)

    tool_correctness= ToolCorrectnessMetric(evaluation_params = [ToolCallParams.TOOL])
    task_completion = TaskCompletionMetric(model="gpt-4o-mini")

    evaluation = evaluate(test_cases, [tool_correctness, task_completion],max_concurrent=1, ignore_errors=True, run_async=False, throttle_value=10, use_cache=True) 
    with open(f"./results/{prefix}-num-tools-results.json", "w") as f:
         json.dump(evaluation.model_dump(), f)

evaluateTestCases("claude")
