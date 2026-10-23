"""Embedding and reranking models behind small interfaces, so tests can swap in fakes.

Both default models run on CPU through ONNX (FastEmbed, FlashRank): no PyTorch, a much smaller
install and a faster cold start on free hosting. They are downloaded on first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in [0, 1]."""
        ...


class FastEmbedEmbedder:
    """Sentence embeddings via FastEmbed. E5 models get their required prefixes."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if e5 else ""
        self._passage_prefix = "passage: " if e5 else ""

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [v.tolist() for v in model.embed([self._passage_prefix + t for t in texts])]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().embed([self._query_prefix + text]))).tolist()


class FlashRankReranker:
    """Cross-encoder reranking via FlashRank (ONNX)."""

    def __init__(self, model_name: str, cache_dir: Path, max_length: int = 512) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._ranker = None

    def _load(self):
        if self._ranker is None:
            from flashrank import Ranker

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                max_length=self.max_length,
            )
        return self._ranker

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        from flashrank import RerankRequest

        passages = [{"id": i, "text": t} for i, t in enumerate(texts)]
        results = self._load().rerank(RerankRequest(query=query, passages=passages))
        scores = [0.0] * len(texts)
        for item in results:
            scores[int(item["id"])] = min(1.0, max(0.0, float(item["score"])))
        return scores
