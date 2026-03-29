"""
Lightweight memory store with TF-IDF retrieval.

Loads memories from the JSON produced by ingest_locomo.py and supports
cosine-similarity search over observation and event_summary texts.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MEMORIES_PATH = Path(__file__).parent / "memories.json"


class MemoryStore:
    def __init__(self, path: Path = MEMORIES_PATH):
        self.memories: list[dict] = []
        self.conversations: list[dict] = []
        self.sample_questions: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

        if path.exists():
            self.load(path)
        else:
            logger.warning(f"Memory store not found at {path}. Run ingest_locomo.py first.")

    def load(self, path: Path):
        with open(path) as f:
            data = json.load(f)
        self.memories = data.get("memories", [])
        self.conversations = data.get("conversations", [])
        self.sample_questions = data.get("sample_questions", [])
        self._build_index()
        logger.info(f"Loaded {len(self.memories)} memories from {len(self.conversations)} conversations")

    def _build_index(self):
        if not self.memories:
            return
        texts = [m["text"] for m in self.memories]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.memories or self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                break
            mem = dict(self.memories[idx])
            mem["score"] = round(float(scores[idx]), 4)
            results.append(mem)
        return results

    @property
    def ready(self) -> bool:
        return len(self.memories) > 0
