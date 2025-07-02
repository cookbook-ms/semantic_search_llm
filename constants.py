import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
GPT_URL = "https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com/v1/chat/completions"
API_URL = "https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com"

API_ENDPOINT = "/embeddings"

