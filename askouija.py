import requests
import requests.auth

secrets = open(".secrets", 'r').readlines()

client_id = secrets[0]
client_secret = secrets[1]

print("Client ID: {0}Client Secret: {1}".format(client_id, client_secret))
