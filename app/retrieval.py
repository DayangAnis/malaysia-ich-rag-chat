"""Pure retrieval utilities shared by the app and unit tests."""
from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Basic lexical tokenisation for BM25."""
    return re.findall(r"\\b[\\w'-]+\\b", str(text).lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked lists of document indices using reciprocal rank fusion."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
