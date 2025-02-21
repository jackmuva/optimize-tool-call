from analyze_utils import *
import json
import pandas as pd

with open('./results/o3-gpt-sys-prompt-1-results.json') as f:
    prompt_1_dict = json.load(f)
with open('./results/o3-gpt-sys-prompt-2-results.json') as f:
    prompt_2_dict = json.load(f)
with open('./results/o3-gpt-llm-results.json') as f:
    prompt_0_dict = json.load(f)

prompt_1_data = pd.read_csv('./data/o3-gpt-sys-prompt-1-axis-dataset.csv').to_dict()
prompt_2_data = pd.read_csv('./data/o3-gpt-sys-prompt-2-axis-dataset.csv').to_dict()
prompt_0_data = pd.read_csv('./data/o3-gpt-llm-axis-dataset.csv').to_dict()
prompt_1_res = clean_results(prompt_1_dict, prompt_1_data)
prompt_2_res = clean_results(prompt_2_dict, prompt_2_data)
prompt_0_res = clean_results(prompt_0_dict, prompt_0_data)
results_table = create_results_table([prompt_0_res, prompt_1_res, prompt_2_res], ['prompt-0', 'prompt-1', 'prompt-2'])
results_table.to_csv('./results/o3-gpt-sys-prompt-comparison-table.csv')
