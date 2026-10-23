"""Deterministic stand-ins for the ML models and the LLM, so tests run offline in seconds."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator

import pymupdf

from regushield.llm import Completion, LLMError, Usage

DIM = 128


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector: shared words -> similar vectors."""

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in _words(text):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SynonymEmbedder(HashEmbedder):
    """Like HashEmbedder, but maps synonyms to one word, mimicking a semantic model."""

    SYNONYMS = {"hadiah": "gratifikasi", "bingkisan": "gratifikasi", "cendera": "gratifikasi"}

    def _vector(self, text: str) -> list[float]:
        return super()._vector(" ".join(self.SYNONYMS.get(w, w) for w in _words(text)))


class OverlapReranker:
    """Relevance = share of query words found in the passage."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        q = set(_words(query))
        return [len(q & set(_words(t))) / len(q) if q else 0.0 for t in texts]


class FakeChat:
    """Streams a scripted answer (in pieces) or raises an LLMError part-way."""

    def __init__(
        self,
        answer: str = "Jawaban [1].",
        usage: Usage | None = None,
        fail: LLMError | None = None,
        fail_after: int = 0,
    ) -> None:
        self.model = "fake/model"
        self.answer = answer
        self.usage = usage if usage is not None else Usage(120, 30)
        self.fail = fail
        self.fail_after = fail_after
        self.messages: list[list[dict[str, str]]] = []

    def stream(self, messages: list[dict[str, str]]) -> Completion:
        self.messages.append(messages)
        return Completion(self._deltas(), sum(len(m["content"]) for m in messages))

    def _deltas(self) -> Iterator[tuple[str, Usage | None]]:
        pieces = re.findall(r"\S+\s*", self.answer) or [""]
        for i, piece in enumerate(pieces):
            if self.fail is not None and i == self.fail_after:
                raise self.fail
            yield piece, None
        yield "", self.usage


def make_pdf(
    pages: list[str],
    header: str | None = None,
    table_on: int | None = None,
    table: list[list[str]] | None = None,
    page_numbers: bool = False,
) -> bytes:
    """Build a PDF: one text block per page, optional running header, optional grid table."""
    doc = pymupdf.open()
    for n, text in enumerate(pages, start=1):
        page = doc.new_page()
        if header:
            page.insert_text((72, 40), header, fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 80, 520, 400), text, fontsize=11)
        if page_numbers:
            page.insert_text((300, 800), str(n), fontsize=9)
        if table_on == n and table:
            x0, y0, cw, rh = 72, 450, 200, 24
            for r, row in enumerate(table):
                for c, cell in enumerate(row):
                    rect = pymupdf.Rect(
                        x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh
                    )
                    page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                    page.insert_text((rect.x0 + 4, rect.y0 + 16), cell, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data
