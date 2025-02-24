from utils.evaluate_utils import *

#LLM
# evaluateTestCases("o3-gpt-upd", "llm-0")
# evaluateTestCases("gpt-4o-upd", "llm-0")
# evaluateTestCases("claude-upd", "llm-0")

#SYS_PROMPTS
# evaluateTestCases("gpt-4o", "sys_prompt-1")
# evaluateTestCases("gpt-4o", "sys_prompt-2")

#ROUTING
# evaluateTestCases("gpt-4o", "num_tools")
evaluateTestCases("claude", "num_tools")

#DESCRIPTIONS
# evaluateTestCases("gpt-4o", "enh_desc-0")
