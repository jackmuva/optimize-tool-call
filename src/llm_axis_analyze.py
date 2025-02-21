from analyze_utils import *
import json
import pandas as pd


with open('./results/gpt-llm-results.json') as f:
    gpt_dict = json.load(f)
with open('./results/claude-llm-results.json') as f:
    claude_dict = json.load(f)
with open('./results/o3-gpt-llm-results.json') as f:
    o3_dict = json.load(f)

gpt_data = pd.read_csv('./data/gpt-llm-axis-dataset.csv').to_dict()
claude_data = pd.read_csv('./data/claude-llm-axis-dataset.csv').to_dict()
o3_data = pd.read_csv('./data/o3-gpt-llm-axis-dataset.csv').to_dict()
gpt_res = clean_results(gpt_dict, gpt_data)
claude_res = clean_results(claude_dict, claude_data)
o3_res = clean_results(o3_dict, o3_data)
results_table = create_results_table([gpt_res, claude_res, o3_res], ['gpt', 'claude', 'o3-gpt'])
results_table.to_csv('./results/llm-comparison-table.csv')
