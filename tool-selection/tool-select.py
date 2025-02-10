import json
import requests
from dotenv import load_dotenv
import jwt
import os
import time

load_dotenv()

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
    'SALESFORCE_SEARCH_RECORDS_CONTACT',
    'SALESFORCE_SEARCH_RECORDS_ANY',
    'SALESFORCE_GET_RECORD_BY_ID_CONTACT',
    'SALESFORCE_WRITE_SOQL_QUERY',

    'HUBSPOT_CREATE_RECORD_CONTACTS',
    'HUBSPOT_GET_RECORDS_ANY',
    'HUBSPOT_SEARCH_RECORDS_ANY',
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
    'GOOGLE_DRIVE_SEARCH_FOLDERS',
    'GOOGLE_DRIVE_GET_FOLDER_BY_ID',
    'GOOGLE_DRIVE_EXPORT_DOC',

    'NOTION_CREATE_PAGE',
    'NOTION_GET_PAGE_BY_ID',
    'NOTION_SEARCH_PAGES',
    'NOTION_GET_PAGE_CONTENT',
    'NOTION_GET_BLOCK_BY_ID',
]

toolbox = []

for integration, functions in allActions['actions'].items():
    for function in functions:
        toolbox.append(function)


