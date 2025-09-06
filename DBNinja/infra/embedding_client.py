# infra/embedding_client.py
import requests
from typing import List
from config.settings import settings

class EmbeddingClient:
    def __init__(self):
        self.url = f"{settings.EMBEDDINGS_PATH}"

    def embed(self, inputs: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": settings.API_KEY
        }
        payload = {"input": inputs}
        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        print(resp.json())
        data = resp.json()["data"]
        return [d["embedding"] for d in data]
