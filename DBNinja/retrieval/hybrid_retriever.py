# retrieval/hybrid_retriever.py
from typing import List, Dict
from infra.embedding_client import EmbeddingClient
from infra.qdrant_client import Qdrant
from domain.ranking import min_max_normalize, fuse_scores

class HybridRetriever:
    def __init__(self, topk_sem: int, topk_kw: int, alpha: float):
        self.embedder = EmbeddingClient()
        self.qdrant = Qdrant()
        self.topk_sem = topk_sem
        self.topk_kw = topk_kw
        self.alpha = alpha

    def retrieve(self, query: str):
        vec = self.embedder.embed([query])[0]

        sem_hits = self.qdrant.semantic_search(vec, self.topk_sem)
        kw_hits  = self.qdrant.keyword_search(query, self.topk_kw)

        # Extract scores and payloads
        sem_items = [{"id": h.id, "score": h.score, "payload": h.payload} for h in sem_hits]
        # Build keyword score: use score if present else heuristic
        kw_items = []
        q_terms = set(t.lower() for t in query.split())
        for h in kw_hits:
            payload = h.payload
            text = (payload.get("doc_text") or "").lower()
            matched = sum(1 for t in q_terms if t in text)
            coverage = matched / max(1, len(q_terms))
            phrase = 0.1 if query.lower() in text else 0.0
            pop = float(payload.get("popularity", 0.5))
            score = min(1.0, coverage + phrase + 0.1 * pop)
            kw_items.append({"id": h.id, "score": score, "payload": payload})

        # Normalize within each list
        sem_scores = [x["score"] for x in sem_items]
        kw_scores  = [x["score"] for x in kw_items]
        sem_norm = min_max_normalize(sem_scores)
        kw_norm  = min_max_normalize(kw_scores)

        for i, x in enumerate(sem_items):
            x["norm"] = sem_norm[i]
        for i, x in enumerate(kw_items):
            x["norm"] = kw_norm[i]

        # Merge by id with fusion
        by_id: Dict[str, Dict] = {}
        for x in sem_items:
            by_id[str(x["id"])] = {"payload": x["payload"], "sem": x["norm"], "kw": 0.0}
        for x in kw_items:
            e = by_id.get(str(x["id"]))
            if e:
                e["kw"] = max(e["kw"], x["norm"])
            else:
                by_id[str(x["id"])] = {"payload": x["payload"], "sem": 0.0, "kw": x["norm"]}

        results = []
        for k, v in by_id.items():
            fused = fuse_scores(v["sem"], v["kw"], self.alpha)
            results.append({"id": k, "payload": v["payload"], "sem": v["sem"], "kw": v["kw"], "score": fused})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results
