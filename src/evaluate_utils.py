from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from dotenv import load_dotenv
import json
import pandas as pd
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def completeJsonFormat(parseList:list, output_dict, index) -> list:
    res = []
    for s in parseList:
        new = s
        if s == 'ERROR':
            new = '{"error": "request too large"}'
        else:
            try:
                json.loads(new)
            except:
                new = correctFormatWithLlm(new)

            try:
                json.loads(new if new != None else "NONE")
            except:
                print("ERROR")
                print(output_dict['tool_inputs'][index][1:len(output_dict['tool_inputs'][index]) - 1])
                with open('./error.txt', 'w') as file:
                    file.write(new if new != None else "NONE")
                new = '{"error": "unable to convert to json"}'
        res.append(new)
    return res

def correctFormatWithLlm(jsonString: str):
    print('using LLM to format')
    with open('./jsonString.txt', 'w') as file:
        file.write(jsonString)

    messages=[]
    messages.append({
        "role": "user",
        "content": "Correct this JSON to be correctly formatted. Only return the formatted JSON as a raw string with no new lines. Do not return the result in a code block, just plain text. The incorrect JSON is:" +
            jsonString
    })

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )
    res = chat_completion.choices[0].message.content
    return res.replace('\\"', "'").replace('\\n', "") if res != None else res

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
    formattedToolInputs = None
    formattedToolOutputs = None
    if prefix == 'gpt' or prefix == 'o3-gpt':
        parsedToolOutputs = output_dict['tool_outputs'][index][2:len(output_dict['tool_outputs'][index]) - 2].replace("', '", "','").split("','")
        parsedToolInputs =  output_dict['tool_inputs'][index][2:len(output_dict['tool_inputs'][index]) - 2].replace("', '", "','").split("','")
    elif prefix == 'claude':
        parsedToolOutputs = output_dict['tool_outputs'][index][2:len(output_dict['tool_outputs'][index]) - 2].replace("'", '"').replace("},{", "}||{").replace('}", "{', "}||{").replace('}", "[', "}||[").split("||")
        parsedToolInputs =  output_dict['tool_inputs'][index][1:len(output_dict['tool_inputs'][index]) - 1].replace("'", '"').replace("},{", "}||{").replace('}", "{', "}||{").replace('}", "[', "}||[").split("||")
    if parsedToolInputs and parsedToolOutputs:
        formattedToolOutputs = completeJsonFormat(parsedToolOutputs, output_dict, index) 
        formattedToolInputs = completeJsonFormat(parsedToolInputs, output_dict, index) 
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
            formattedToolInputs.append("{}")
        if i >= len(parsedToolOutputs):
            formattedToolOutputs.append("{}")
        try:
            toolCall = ToolCall(name=toolName, description=description, input_parameters=json.loads(formattedToolInputs[i]), output=str(formattedToolOutputs[i]))
        except:
            toolCall = ToolCall(name=toolName, description=description, input_parameters=json.loads("{}"), output=str("{}"))
        res.append(toolCall)
    return res

def formatExpectedToolCalls(index: int, output_dict: dict) -> list:
    toolCalls = []
    expectedTools = output_dict['tool_name'][index].split(",")
    for name in expectedTools:
        toolCall = ToolCall(name = name, input_parameters={})
        toolCalls.append(toolCall)
    return toolCalls



def evaluateTestCases(prefix, use_case):
    output_dict = pd.read_csv(f'./data/{prefix}-{use_case}-axis-dataset.csv').to_dict()

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
    with open(f"./results/{prefix}-{use_case}-results.json", "w") as f:
         json.dump(evaluation.model_dump(), f)


