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
print(response.status_code)
print(response.json())
print(os.environ['PARAGON_SIGNING_KEY'])
print(os.environ['PARAGON_PROJECT_ID'])
