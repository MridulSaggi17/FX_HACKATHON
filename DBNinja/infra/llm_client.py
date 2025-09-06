# infra/llm_client.py
import requests
from typing import List, Dict
from config.settings import settings

class LLMClient:
    def __init__(self):
        self.url = f"{settings.API_BASE.rstrip('/')}{settings.PREDICT_PATH}"

    def chat(self, messages: List[Dict]) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": settings.API_KEY
        }
        payload = {"messages": messages}
        resp = requests.post(self.url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["output"]
