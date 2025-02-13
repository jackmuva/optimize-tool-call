import json
import pandas as pd

with open('./results/gpt-llm-results.json') as f:
    gpt_dict = json.load(f)

print(gpt_dict.keys())

for i in range(0, len(gpt_dict['test_results'])):
    print(gpt_dict['test_results'][i].keys())
    break

