import requests
from constants import API_URL, API_KEY, API_ENDPOINT

def embed(text, embedding_model_choice):
    """Send text to the embedding model."""
    payload = {
        "model": f"text-embedding-3-{embedding_model_choice}",
        "input": text
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(API_URL + API_ENDPOINT, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]