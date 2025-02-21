from analyze_utils import *
import json
import pandas as pd

with open('./results/claude-num-tools-results.json') as f:
    num_tools_dict = json.load(f)
with open('./results/claude-llm-results.json') as f:
    all_tools_dict = json.load(f)

num_tools_data = pd.read_csv('./data/claude-num-tool-axis-dataset.csv').to_dict()
num_tools_res = clean_results(num_tools_dict, num_tools_data)
all_tools_data = pd.read_csv('./data/claude-llm-axis-dataset.csv').to_dict()
all_tools_res = clean_results(all_tools_dict, all_tools_data)
results_table = create_results_table([num_tools_res, all_tools_res], ['num_tools', 'all_tools'])
results_table.to_csv('./results/claude-num-tools-comparison-table.csv')
