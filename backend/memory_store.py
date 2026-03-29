"""
Semantic memory store with sentence-transformer embeddings.

Uses all-MiniLM-L6-v2 for 384-dim dense embeddings — understands meaning,
not just keywords. Falls back to TF-IDF if sentence-transformers unavailable.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MEMORIES_PATH = Path(__file__).parent / "memories.json"

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed, falling back to TF-IDF")


class MemoryStore:
    def __init__(self, path: Path = MEMORIES_PATH):
        self.memories: list[dict] = []
        self.conversations: list[dict] = []
        self.sample_questions: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._model = None
        self._use_st = _ST_AVAILABLE

        # TF-IDF fallback
        self._vectorizer = None
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

        if self._use_st:
            logger.info("Building semantic index with all-MiniLM-L6-v2...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            logger.info(f"Encoded {len(texts)} memories → {self._embeddings.shape}")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def _detect_speaker(self, query: str) -> str | None:
        """Return the speaker name if the query targets a specific person."""
        all_speakers = set()
        for c in self.conversations:
            all_speakers.update(c.get("speakers", []))
        query_lower = query.lower()
        matched = [s for s in all_speakers if s.lower() in query_lower]
        return matched[0] if len(matched) == 1 else None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.memories:
            return []

        if self._use_st and self._model is not None and self._embeddings is not None:
            query_emb = self._model.encode([query], normalize_embeddings=True)
            scores = (self._embeddings @ query_emb.T).flatten()
        elif self._vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        else:
            return []

        target_speaker = self._detect_speaker(query)
        if target_speaker:
            for i, mem in enumerate(self.memories):
                if mem.get("speaker", "").lower() == target_speaker.lower():
                    scores[i] *= 2.5
                else:
                    scores[i] *= 0.2

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
