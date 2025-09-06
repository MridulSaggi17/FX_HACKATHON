# retrieval/reranker.py
import requests
from typing import List, Tuple
from config.settings import settings

class Reranker:
    def __init__(self):
        self.url = f"{settings.API_BASE.rstrip('/')}{settings.RERANK_PATH}"

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": settings.API_KEY
        }
        payload = {"query": query, "candidates": candidates}
        resp = requests.post(self.url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        # Expecting list of {index, score}
        return [(r["index"], r["score"]) for r in resp.json()["results"]]
