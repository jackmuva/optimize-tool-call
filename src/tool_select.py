import json
import requests
from dotenv import load_dotenv
import jwt
import os
import time

load_dotenv()

def getTools():
    currentTime = time.time()
    encoded_jwt = jwt.encode({
        "sub": "jack.mu@useparagon.com",
        "iat": currentTime,
        "exp": currentTime + (60 * 60 * 24 * 7)
    }, os.environ['PARAGON_SIGNING_KEY'].replace("\\n", "\n"), algorithm="RS256")

    response = requests.get("https://actionkit.useparagon.com/projects/" + os.environ['PARAGON_PROJECT_ID'] + "/actions/",
                             headers={"Authorization": "Bearer " + encoded_jwt})
    print("Actions retrieved with: " + str(response.status_code))

    allActions = json.loads(response.text)
    with open("./data/tool-metadata.json", "w") as json_file:
        json.dump(allActions, json_file, indent=4)

    print("Integrations enabled: " + str(allActions['actions'].keys()))

    selectedActionNames = [
        'SALESFORCE_CREATE_RECORD_CONTACT',
        'SALESFORCE_SEARCH_RECORDS_CONTACTS',
        'SALESFORCE_WRITE_SOQL_QUERY',

        'HUBSPOT_CREATE_RECORD_CONTACTS',
        'HUBSPOT_GET_RECORDS_ANY',
        'HUBSPOT_SEARCH_RECORDS_CONTACTS',

        'SLACK_SEARCH_MESSAGES',
        'SLACK_SEND_DIRECT_MESSAGE',
        'SLACK_LIST_MEMBERS',
        'SLACK_GET_USER_BY_EMAIL',

        'GMAIL_SEND_EMAIL',
        'GMAIL_SEARCH_FOR_EMAIL',
        'GMAIL_GET_EMAIL_BY_ID'

        'GOOGLE_DRIVE_GET_FILE_BY_ID',
        'GOOGLE_DRIVE_LIST_FILES',

        'NOTION_CREATE_PAGE',
        'NOTION_SEARCH_PAGES',
        'NOTION_GET_PAGE_CONTENT',
    ]

    toolbox = []

    for integration, functions in allActions['actions'].items():
        for function in functions:
            if function['function']['name'] in selectedActionNames:
                toolbox.append(function)

    return toolbox

def getEnhancedDescTools():
    with open("./meta/upd_tool_desc.json", "r") as json_file:
        allActions = json.load(json_file)
    print("Integrations enabled: " + str(allActions['actions'].keys()))
    toolbox = []

    for integration, functions in allActions['actions'].items():
        for function in functions:
            toolbox.append(function)

    return toolbox

