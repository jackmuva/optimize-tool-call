import json
import pandas as pd


def clean_results(input_dict: dict, dataset: dict) -> dict:
    clean_query_results = {}

    for i in range(0, len(input_dict['test_results'])):
        clean_record = {}

        clean_record['success'] = input_dict['test_results'][i]['success']
        clean_record['input'] = input_dict['test_results'][i]['input']
        clean_record['actual_output'] = input_dict['test_results'][i]['actual_output']

        clean_record['source'] = dataset['source'][i]
        clean_record['tool_name'] = dataset['tool_name'][i]
        clean_record['intent'] = dataset['intent'][i]
        clean_record['tools_used'] = dataset['tools'][i]
        clean_record['tool_inputs'] = dataset['tool_inputs'][i]
        clean_record['tool_outputs'] = dataset['tool_outputs'][i]
        clean_record['error'] = dataset['error'][i]

        metrics = input_dict['test_results'][i]['metrics_data']
        metric_record = {}
        for j in range(0, len(metrics)):
            metric_details = {}

            metric_details['threshold'] = metrics[j]['threshold']
            metric_details['success'] = metrics[j]['success']
            metric_details['score'] = metrics[j]['score']
            metric_details['reason'] = metrics[j]['reason']

            metric_record[metrics[j]['name']] = metric_details

        clean_record['metrics'] = metric_record
        clean_query_results[input_dict['test_results'][i]['name']] = clean_record
    return clean_query_results



def create_results_table(results: list, labels: list) -> pd.DataFrame:
    df_dict = {}
    df_dict['method'] = []
    df_dict['test_case'] = []
    df_dict['input'] = []
    df_dict['actual_output'] = []
    df_dict['source'] = []
    df_dict['tool_name'] = []
    df_dict['intent'] = []
    df_dict['tools_used'] = []
    df_dict['tool_outputs'] = []
    df_dict['tool_inputs'] = []
    df_dict['error'] = []
    df_dict['tool_correctness_score'] = []
    df_dict['tool_correctness_reason'] = []
    df_dict['task_completion_score'] = []
    df_dict['task_completion_reason'] = []

    for index, res in enumerate(results):
        for i in res.keys():
            df_dict['test_case'].append(i)
            df_dict['method'].append(labels[index])
            df_dict['input'].append(res[i]['input'])
            df_dict['actual_output'].append(res[i]['actual_output'])
            df_dict['source'].append(res[i]['source'])
            df_dict['tool_name'].append(res[i]['tool_name'])
            df_dict['intent'].append(res[i]['intent'])
            df_dict['tools_used'].append(res[i]['tools_used'])
            df_dict['tool_outputs'].append(res[i]['tool_outputs'])
            df_dict['tool_inputs'].append(res[i]['tool_inputs'])
            df_dict['error'].append(res[i]['error'])

            for j in res[i]['metrics'].keys():
                if j == 'Tool Correctness':
                    df_dict['tool_correctness_score'].append(res[i]['metrics'][j]['score'])
                    df_dict['tool_correctness_reason'].append(res[i]['metrics'][j]['reason'])
                elif j == 'Task Completion':
                    df_dict['task_completion_score'].append(res[i]['metrics'][j]['score'])
                    df_dict['task_completion_reason'].append(res[i]['metrics'][j]['reason'])
    return pd.DataFrame.from_dict(df_dict)


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
