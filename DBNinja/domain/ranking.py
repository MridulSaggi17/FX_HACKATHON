# domain/ranking.py
from typing import List

def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    smin, smax = min(scores), max(scores)
    if smax == smin:
        return [0.0 for _ in scores]
    return [(s - smin) / (smax - smin) for s in scores]

def fuse_scores(sem: float, kw: float, alpha: float) -> float:
    return alpha * sem + (1 - alpha) * kw
