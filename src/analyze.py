import json
import pandas as pd
from utils.analyze_utils import *

#LLM
# with open('./results/gpt-4o-upd-llm-0-results.json') as f:
#     gpt_dict = json.load(f)
# with open('./results/claude-upd-llm-0-results.json') as f:
#     claude_dict = json.load(f)
# with open('./results/o3-gpt-upd-llm-0-results.json') as f:
#     o3_dict = json.load(f)
#
# gpt_data = pd.read_csv('./data/gpt-4o-upd-llm-0-axis-dataset.csv').to_dict()
# claude_data = pd.read_csv('./data/claude-upd-llm-0-axis-dataset.csv').to_dict()
# o3_data = pd.read_csv('./data/o3-gpt-upd-llm-0-axis-dataset.csv').to_dict()
# gpt_res = clean_results(gpt_dict, gpt_data)
# claude_res = clean_results(claude_dict, claude_data)
# o3_res = clean_results(o3_dict, o3_data)
# results_table = create_results_table([gpt_res, claude_res, o3_res], ['gpt', 'claude', 'o3-gpt'])
# results_table.to_csv('./results/llm-comparison-table.csv')


#system prompt
# with open('./results/gpt-4o-sys_prompt-1-results.json') as f:
#     prompt_1_dict = json.load(f)
# with open('./results/gpt-4o-sys_prompt-2-results.json') as f:
#     prompt_2_dict = json.load(f)
# with open('./results/gpt-4o-upd-llm-0-results.json') as f:
#     prompt_0_dict = json.load(f)
#
# prompt_1_data = pd.read_csv('./data/gpt-4o-sys_prompt-1-axis-dataset.csv').to_dict()
# prompt_2_data = pd.read_csv('./data/gpt-4o-sys_prompt-2-axis-dataset.csv').to_dict()
# prompt_0_data = pd.read_csv('./data/gpt-4o-upd-llm-0-axis-dataset.csv').to_dict()
# prompt_1_res = clean_results(prompt_1_dict, prompt_1_data)
# prompt_2_res = clean_results(prompt_2_dict, prompt_2_data)
# prompt_0_res = clean_results(prompt_0_dict, prompt_0_data)
# results_table = create_results_table([prompt_0_res, prompt_1_res, prompt_2_res], ['prompt-0', 'prompt-1', 'prompt-2'])
# results_table.to_csv('./results/gpt-4o-sys-prompt-comparison-table.csv')
#
# #routing
with open('./results/claude-num_tools-results.json') as f:
    num_tools_dict = json.load(f)
with open('./results/claude-upd-llm-0-results.json') as f:
    all_tools_dict = json.load(f)

num_tools_data = pd.read_csv('./data/claude-num_tools-axis-dataset.csv').to_dict()
num_tools_res = clean_results(num_tools_dict, num_tools_data)
all_tools_data = pd.read_csv('./data/claude-upd-llm-0xis-dataset.csv').to_dict()
all_tools_res = clean_results(all_tools_dict, all_tools_data)
results_table = create_results_table([num_tools_res, all_tools_res], ['num_tools', 'all_tools'])
results_table.to_csv('./results/claude-num-tools-comparison-table.csv')

# #descriptions
# with open('./results/gpt-4o-upd-llm-0-results.json') as f:
#     norm_dict = json.load(f)
# with open('./results/gpt-4o-enh_desc-0-results.json') as f:
#     enh_desc_dict = json.load(f)
#
# num_tools_data = pd.read_csv('./data/gpt-4o-upd-llm-0-axis-dataset.csv').to_dict()
# norm_res = clean_results(norm_dict, num_tools_data)
# enh_desc_data = pd.read_csv('./data/gpt-4o-enh_desc-0-axis-dataset.csv').to_dict()
# enh_desc_res = clean_results(enh_desc_dict, enh_desc_data)
# results_table = create_results_table([norm_res, enh_desc_res], ['normal_descriptions', 'enhanced_descriptions'])
# results_table.to_csv('./results/gpt-4o-enhanced-desc-comparison-table.csv')
