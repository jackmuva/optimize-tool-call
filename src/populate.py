from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from utils.populate_utils import *

load_dotenv()
test_df = pd.read_csv("./data/tool-based-test-cases.csv")
test_dict = test_df.to_dict()
tools = getTools()

#LLM
sys_prompt_0 = '''
    You are a task assistant. Always attempt to use a tool. No need to ask clarifications and followups.
'''
# o3_llm = ChatOpenAI(model="o3-mini")
# o3_graph = createGraph(o3_llm, tools)
# runPrompt(test_dict, o3_graph, "o3-gpt", sys_prompt_0, 0, "upd-llm")

# gpt_llm = ChatOpenAI(model="gpt-4o")
# gpt_graph = createGraph(gpt_llm, tools)
# runPrompt(test_dict, gpt_graph, "gpt-4o", sys_prompt_0, 0, "upd-llm")
#
# claude_llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", timeout=None, stop=None)
# claude_graph = createGraph(claude_llm, tools)
# runPrompt(test_dict, claude_graph, "claude", sys_prompt_0, 0, "upd-llm")

#System Prompt
# sys_prompt_1 = '''
#     You are a task assistant. Always attempt to use a tool. No need to ask clarifications and followups.
#
#     Help users perform actions in 3rd-party applications. 
#     Users will ask to create records and search for records in CRMs like Salesforce and Hubspot. 
#     Users will also ask you to send messages/emails and search for messages/emails in messaging 
#     platforms like Slack and Gmail. Lastly, users will ask to search through pages and documents like 
#     in Google Drive and Notion. Use tools provided to help you with these tasks.
#     '''
# sys_prompt_2 = '''
#     You are a task assistant. Always attempt to use a tool. No need to ask clarifications and followups.
#
#     Help users perform actions in 3rd-party applications. 
#     Users will ask to create records and search for records in CRMs like Salesforce and Hubspot, 
#     send and search messages in Gmail and Slack, and search for documents and pages in Notion and 
#     Google Drive. Use tools provided to help you with these tasks.
#
#     Some Rules:
#     * If a user mentions multiple data sources, use tools to search and/or take action across each data source.
#     * If you use a tool that searches for information, but fails, try using a slightly different filter with your next best guess. For example, if a user asks to find Mistral.ai in Salesforce, but you can’t find it by using Name=Mistral.ai, then try searching with name=Mistral
#     * If a tool continues to fail even after trying different inputs, try a different tool. For example when searching Salesforce contacts, 
#     if SALESFORCE_WRITE_SOQL_QUERY fails, try using SALESFORCE_SEARCH_RECORDS_CONTACT
#     '''
#
# gpt_llm = ChatOpenAI(model="gpt-4o")
# gpt_graph = createGraph(gpt_llm, tools)
#
# runPrompt(test_dict, gpt_graph, "gpt-4o", sys_prompt_1, 1, "sys_prompt")
# runPrompt(test_dict, gpt_graph, "gpt-4o", sys_prompt_2, 2, "sys_prompt")
#
# #Routing
runPromptWithRouting(test_dict, "claude", sys_prompt_0, 0, getToolsDict(getTools()), "num_tools")
#
# #Enhanced Descriptions
# gpt_llm = ChatOpenAI(model="gpt-4o")
# enh_desc_tools = getEnhancedDescTools()
# gpt_graph = createGraph(gpt_llm, enh_desc_tools)
#
# runPrompt(test_dict, gpt_graph, "gpt-4o", sys_prompt_0, 0, "enh_desc")
