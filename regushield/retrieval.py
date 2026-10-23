"""Hybrid retrieval: BM25 + vector search, fused with Reciprocal Rank Fusion, then reranked.

Why both: BM25 finds exact legal terms and article numbers ("Pasal 12", "POJK 51"), vector
search finds paraphrases ("hadiah" vs "gratifikasi"). RRF combines the two rankings without
having to calibrate their very different score scales. A cross-encoder then rereads the best
candidates together with the question and gives each a comparable relevance score in [0, 1].

Context expansion: a hit that stops mid-sentence or mid-list ("...tiga kategori: Ringan,") is
followed by the next chunk of the same document, because the rest of the list usually shares
no words with the question and would never be retrieved on its own. A hit that starts in the
middle of a numbered list ("2. E-mail ...") is preceded by the chunk holding the list's start.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from regushield.chunking import Chunk
from regushield.models import Reranker
from regushield.store import VectorStore

# Very common Indonesian function words and question words. Removing them stops BM25 from
# ranking chunks highly just because they share "yang", "dan" or "apa" with the question.
STOPWORDS = frozenset(
    """
    yang dan di ke dari untuk dengan pada adalah ini itu atau dalam oleh sebagai juga akan
    tidak dapat bagi para serta karena maka jika bila agar telah sudah harus kepada tersebut
    apa apakah bagaimana berapa kapan siapa mengapa dimana manakah saja ada sebuah suatu
    """.split()
)
RRF_K = 60
_SENTENCE_END = (".", "!", "?")
_CONTINUES = (",", ":", ";", "-", "–")
_CLOSERS = "\"'”’)]*_ "
HEADING_MAX_CHARS = 60
# A chunk that opens with item 2 or later ("2. E-mail", "c) ...") starts inside a list.
_MID_LIST_START = re.compile(r"\s*(?:[2-9]|[1-9]\d|[b-z])[.)]\s")


def is_open_ended(text: str) -> bool:
    """True when a chunk was cut before its sentence or list ended.

    A short last line in Title Case ("4. Program Awareness Kode Etik") is a section heading,
    not a cut sentence: the section that follows is a different topic, so it is not added.
    """
    stripped = text.rstrip(_CLOSERS)
    if not stripped or stripped.endswith(_SENTENCE_END):
        return False
    if stripped.endswith(_CONTINUES):
        return True
    last_line = stripped.splitlines()[-1].strip()
    last_word = last_line.split()[-1] if last_line else ""
    is_heading = len(last_line) <= HEADING_MAX_CHARS and last_word[:1].isupper()
    return not is_heading


def starts_mid_list(text: str) -> bool:
    """True when a chunk opens with list item 2 or later, so the list began in the chunk before."""
    return _MID_LIST_START.match(text) is not None


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokens = [tokenize(c.text) or ["_"] for c in chunks]
        self._terms = [set(t) for t in tokens]
        self._bm25 = BM25Okapi(tokens) if chunks else None

    def search(self, query: str, k: int, source: str | None = None) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        wanted = set(terms)
        ranked = sorted(
            (
                (chunk, float(score))
                for chunk, score, chunk_terms in zip(self.chunks, scores, self._terms, strict=True)
                # keep any chunk sharing a term: in tiny corpora BM25's IDF can be 0 for a match
                if chunk_terms & wanted and (source is None or chunk.source == source)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def _order(chunk: Chunk) -> tuple[str, int]:
    return chunk.source, chunk.seq


@dataclass
class Retrieved:
    chunk: Chunk
    fusion_score: float
    bm25_rank: int | None
    vector_rank: int | None
    rerank_score: float | None = None
    continues: str | None = None  # id of the hit this chunk was added after (context expansion)
    precedes: str | None = None  # id of the hit this chunk was added before (context expansion)

    @property
    def relevance(self) -> float | None:
        return self.rerank_score


@dataclass
class RetrievalResult:
    hits: list[Retrieved]
    candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        reranker: Reranker | None,
        candidates: int = 20,
        top_k: int = 4,
        rerank_candidates: int = 10,
        context_window: int = 1,
    ) -> None:
        self.store = store
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates
        self.context_window = context_window
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the BM25 index from the store (after documents are added or removed)."""
        chunks = self.store.all_chunks()
        self.bm25 = BM25Index(chunks)
        # Text chunks of each document in reading order, for context expansion.
        self._reading_order: dict[str, list[Chunk]] = {}
        for c in sorted((c for c in chunks if c.kind == "text" and c.seq >= 0), key=_order):
            self._reading_order.setdefault(c.source, []).append(c)
        self._position = {
            c.id: i for ordered in self._reading_order.values() for i, c in enumerate(ordered)
        }

    def preceding(self, chunk: Chunk) -> Chunk | None:
        """The text chunk right before ``chunk`` on the same page.

        Only the same page: a page that opens with "3. ..." is usually the next numbered
        section, not a list cut by the chunker.
        """
        position = self._position.get(chunk.id)
        if not position:  # unknown, or the first chunk of the document
            return None
        before = self._reading_order[chunk.source][position - 1]
        return before if before.page == chunk.page else None

    def following(self, chunk: Chunk, n: int) -> list[Chunk]:
        """Up to ``n`` text chunks that come right after ``chunk`` in its document."""
        position = self._position.get(chunk.id)
        if position is None:
            return []
        return self._reading_order[chunk.source][position + 1 : position + 1 + n]

    def expand(self, hits: list[Retrieved]) -> list[Retrieved]:
        """Add the chunk before a hit that starts mid-list and the chunk after a hit that stops
        mid-sentence, next to that hit (each chunk at most once)."""
        if self.context_window <= 0:
            return hits
        taken = {hit.chunk.id for hit in hits}
        out: list[Retrieved] = []
        for hit in hits:
            if hit.chunk.kind != "text":
                out.append(hit)
                continue
            before = self.preceding(hit.chunk) if starts_mid_list(hit.chunk.text) else None
            if before is not None and before.id not in taken:
                taken.add(before.id)
                out.append(Retrieved(before, 0.0, None, None, precedes=hit.chunk.id))
            out.append(hit)
            if not is_open_ended(hit.chunk.text):
                continue
            for chunk in self.following(hit.chunk, self.context_window):
                if chunk.id in taken:
                    break
                taken.add(chunk.id)
                out.append(Retrieved(chunk, 0.0, None, None, continues=hit.chunk.id))
                if not is_open_ended(chunk.text):
                    break
        return out

    def retrieve(self, query: str, source: str | None = None) -> RetrievalResult:
        if self.store.count() != len(self.bm25.chunks):
            self.refresh()  # another process (UI or API) changed the shared index
        timings: dict[str, float] = {}
        started = time.perf_counter()
        sparse = self.bm25.search(query, self.candidates, source)
        timings["bm25"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dense = self.store.search(query, self.candidates, source)
        timings["vector"] = (time.perf_counter() - started) * 1000

        by_id = {c.id: c for c, _ in sparse + dense}
        bm25_rank = {c.id: r for r, (c, _) in enumerate(sparse, start=1)}
        vector_rank = {c.id: r for r, (c, _) in enumerate(dense, start=1)}
        fused = reciprocal_rank_fusion([list(bm25_rank), list(vector_rank)])
        pool = [
            Retrieved(by_id[i], score, bm25_rank.get(i), vector_rank.get(i))
            for i, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ][: self.candidates]

        if self.reranker is not None and pool:
            # The cross-encoder is the slowest stage on CPU, so it only rereads the best fused
            # candidates; the rest keep their fusion order behind them.
            pool = pool[: self.rerank_candidates]
            started = time.perf_counter()
            scores = self.reranker.score(query, [hit.chunk.text for hit in pool])
            timings["rerank"] = (time.perf_counter() - started) * 1000
            for hit, score in zip(pool, scores, strict=True):
                hit.rerank_score = score
            pool.sort(key=lambda hit: hit.rerank_score or 0.0, reverse=True)

        hits = self.expand(pool[: self.top_k])
        return RetrievalResult(hits=hits, candidates=len(fused), timings_ms=timings)