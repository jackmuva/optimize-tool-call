from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

def evaluateTestCases(prefix):
    output_dict = pd.read_csv(f'./data/{prefix}-llm-axis-dataset.csv').to_dict()

    test_cases = []

    for index in output_dict['input'].keys():
        test_case = LLMTestCase(
            input=output_dict['input'][index],
            actual_output=output_dict['actual_output'][index],
            tools_called=None,
            expected_tools=None
        ) 
        test_cases.append(test_case)

    tool_correctness= ToolCorrectnessMetric()
    task_completion = TaskCompletionMetric(model="gpt-4")

    evaluation = evaluate(test_cases, [tool_correctness, task_completion],max_concurrent=1, ignore_errors=True, run_async=False, throttle_value=10, use_cache=True) 
    with open(f"./results/{prefix}-llm-results.json", "w") as f:
         json.dump(evaluation.model_dump(), f)


