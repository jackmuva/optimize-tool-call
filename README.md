# optimize-tool-call

## Script Rundown
In this exercise we tested 4 axes to observe their differences in tool correctness and task completion. These axes are:
- LLM choice
- System prompt
- Number of tools w/ routing
- Tool Descriptions

For each axis there is a script for:

- Populating the dataset with responses from the LLM (`..._populate_dataset.py`)
- Evaluating metrics - tool correctness and task completion - with DeepEval (`..._evaluate.py`)
- Analyzing the results in a table by combining the test bank with the evaluation metrics, and then exporting the data to csv to analyze in Sheets, Excel, pandas, etc. (`..._analyze.py`)

The `tool_select.py` script calls the ActionKit API that will retrieve the tool metadata

The `node_utils.py` file has some helper functions for our LangGraph architecture

## Running the scripts
1) Start by creating a `.env` file with the following variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PARAGON_PROJECT_ID`, `PARAGON_SIGNING_KEY`
2) Create the following directories in the root of your project `data`, `results`
3) In the `data` directory, put the `tool-based-test-cases.csv` file into this directory
    1) The CSV should have the following schema: `source`, `tool_name`, `prompt` and `intent`
4) For each axis, run the scripts from the root of your project in the following order:
    1) Populate
    2) Evaluate
    3) Analyze
    - Example: `python src/llm_axis_populate_dataset.py`
