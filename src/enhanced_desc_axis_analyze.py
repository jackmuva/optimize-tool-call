from analyze_utils import *
import json
import pandas as pd

with open('./results/o3-gpt-llm-results.json') as f:
    num_tools_dict = json.load(f)
with open('./results/o3-gpt-enh-desc-results.json') as f:
    enh_desc_dict = json.load(f)

num_tools_data = pd.read_csv('./data/claude-llm-axis-dataset.csv').to_dict()
num_tools_res = clean_results(num_tools_dict, num_tools_data)
enh_desc_data = pd.read_csv('./data/claude-enh-desc-axis-dataset.csv').to_dict()
enh_desc_res = clean_results(enh_desc_dict, enh_desc_data)
results_table = create_results_table([num_tools_res, enh_desc_res], ['num_tools', 'enhanced_descriptions'])
results_table.to_csv('./results/claude-enhanced-desc-comparison-table.csv')
