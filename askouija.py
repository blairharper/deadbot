import requests
import requests.auth
import json

print("IamAdeadbot 0.1 - Welcome to the spirit world. e: b@blairdev.com\n")

secrets = open(".secrets", 'r').readlines()

client_id = secrets[0].rstrip()
client_secret = secrets[1].rstrip()
client_username = secrets[2].rstrip()
client_password = secrets[3].rstrip()

client_auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
post_data = {"grant_type": "password", 
    "username": client_username, 
    "password": client_password}
headers = {"User-Agent": "deadbot/0.1 by blairdev.com"}
print("Authenticating...")
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
json_data = json.loads(response.text)

if 'error' in json_data:
    print("Authentication failed. Please check credentials and internet connection.")
else:
    print("Authenticated!\n Access token: {0}".format(json_data['access_token']))
