"""Module C vector store: TF-IDF vectors over the local IFAB excerpt corpus,
indexed with FAISS for nearest-neighbor retrieval (CLAUDE.md section 4 says
"vector store with Chroma or FAISS"). TF-IDF is used instead of a
sentence-embedding model to keep the skeleton pass free of large model
downloads -- swapping in real embeddings is a deepening-phase item
(TODO.md).
"""
from __future__ import annotations

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.assistant.ifab_corpus import IFAB_EXCERPTS


class IFABRetriever:
    def __init__(self):
        texts = [f"{e['title']} {e['text']}" for e in IFAB_EXCERPTS]
        self.vectorizer = TfidfVectorizer().fit(texts)
        vectors = self.vectorizer.transform(texts).toarray().astype("float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, query: str, k: int = 1) -> list[dict]:
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**IFAB_EXCERPTS[idx], "score": float(score)})
        return results


_retriever = None


def get_retriever() -> IFABRetriever:
    global _retriever
    if _retriever is None:
        _retriever = IFABRetriever()
    return _retriever
