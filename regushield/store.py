"""Persistent chunk store on ChromaDB (vectors + text + metadata)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from regushield.chunking import Chunk
from regushield.models import Embedder

_BATCH = 64


@dataclass(frozen=True)
class DocumentInfo:
    source: str
    chunks: int
    pages: int
    tables: int
    sha256: str
    ingested_at: str


class VectorStore:
    def __init__(self, path: Path | None, embedder: Embedder, collection_name: str) -> None:
        """``path=None`` keeps everything in memory (tests)."""
        settings = ChromaSettings(anonymized_telemetry=False, allow_reset=False)
        if path is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path), settings=settings)
        self._collection = self._client.get_or_create_collection(
            collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = embedder

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], sha256: str, ingested_at: str) -> None:
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embedder.embed_documents([c.text for c in batch]),
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "kind": c.kind,
                        "sha256": sha256,
                        "ingested_at": ingested_at,
                    }
                    for c in batch
                ],
            )

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def all_chunks(self) -> list[Chunk]:
        data = self._collection.get(include=["documents", "metadatas"])
        return [
            Chunk(id_, text, meta["source"], int(meta["page"]), meta.get("kind", "text"))
            for id_, text, meta in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        ]

    def documents(self) -> list[DocumentInfo]:
        data = self._collection.get(include=["metadatas"])
        grouped: dict[str, list[dict]] = {}
        for meta in data["metadatas"]:
            grouped.setdefault(meta["source"], []).append(meta)
        return sorted(
            (
                DocumentInfo(
                    source=source,
                    chunks=len(metas),
                    pages=max(int(m["page"]) for m in metas),
                    tables=sum(1 for m in metas if m.get("kind") == "table"),
                    sha256=metas[0].get("sha256", ""),
                    ingested_at=metas[0].get("ingested_at", ""),
                )
                for source, metas in grouped.items()
            ),
            key=lambda d: d.source.lower(),
        )

    def search(self, query: str, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        """Nearest chunks by cosine similarity (1 = identical direction)."""
        if k <= 0 or self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, self.count()),
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for id_, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            chunk = Chunk(id_, text, meta["source"], int(meta["page"]), meta.get("kind", "text"))
            hits.append((chunk, 1.0 - float(distance)))
        return hits
