import uuid

import pytest

from regushield.chunking import Chunk
from regushield.retrieval import (
    BM25Index,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from regushield.store import VectorStore
from tests.fakes import HashEmbedder, OverlapReranker, SynonymEmbedder


def chunk(i: int, text: str, source: str = "a.pdf", page: int = 1) -> Chunk:
    return Chunk(f"c{i}", text, source, page)


CHUNKS = [
    chunk(1, "Pasal 12 POJK 51 mengatur laporan keberlanjutan."),
    chunk(2, "Pegawai dilarang menerima gratifikasi dari nasabah."),
    chunk(3, "Sanksi berat berupa pemutusan hubungan kerja.", source="b.pdf", page=4),
    chunk(4, "Rapat direksi diadakan setiap bulan."),
]


def store_with(chunks, embedder=None) -> VectorStore:
    store = VectorStore(None, embedder or HashEmbedder(), f"test-{uuid.uuid4().hex[:8]}")
    store.add(chunks, sha256="x", ingested_at="2026-01-01")
    return store


def test_tokenize_drops_punctuation_case_and_stopwords():
    assert tokenize("Apa sanksi yang BERAT, menurut Pasal 12?") == [
        "sanksi",
        "berat",
        "menurut",
        "pasal",
        "12",
    ]


def test_bm25_finds_exact_terms_and_ignores_zero_scores():
    hits = BM25Index(CHUNKS).search("POJK 51", 10)
    assert [c.id for c, _ in hits] == ["c1"]


def test_bm25_source_filter():
    index = BM25Index(CHUNKS)
    assert [c.id for c, _ in index.search("sanksi berat", 10, source="b.pdf")] == ["c3"]
    assert index.search("sanksi berat", 10, source="a.pdf") == []


def test_bm25_with_only_stopwords_returns_nothing():
    assert BM25Index(CHUNKS).search("apa yang dan", 10) == []


def test_rrf_rewards_items_ranked_by_both_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert max(scores, key=scores.get) == "a"
    assert scores["b"] == pytest.approx(1 / 62)


def test_hybrid_finds_a_paraphrase_that_bm25_alone_misses():
    store = store_with(CHUNKS, SynonymEmbedder())
    retriever = HybridRetriever(store, reranker=None, candidates=10, top_k=4)
    assert retriever.bm25.search("hadiah", 10) == []  # no shared word
    hits = retriever.retrieve("hadiah").hits
    assert hits[0].chunk.id == "c2"
    assert hits[0].bm25_rank is None and hits[0].vector_rank == 1


def test_reranker_scores_are_real_and_reorder_candidates():
    reranker = OverlapReranker()
    retriever = HybridRetriever(store_with(CHUNKS), reranker, candidates=10, top_k=2)
    result = retriever.retrieve("sanksi berat pemutusan hubungan kerja")
    assert reranker.calls == 1
    assert result.hits[0].chunk.id == "c3"
    assert result.hits[0].rerank_score == pytest.approx(1.0)
    assert len(result.hits) == 2
    assert result.hits[1].rerank_score < 1.0
    assert set(result.timings_ms) == {"bm25", "vector", "rerank"}


def test_without_reranker_scores_are_absent_not_faked():
    result = HybridRetriever(store_with(CHUNKS), None, 10, 4).retrieve("gratifikasi nasabah")
    assert all(hit.rerank_score is None for hit in result.hits)
    assert "rerank" not in result.timings_ms


def test_source_filter_applies_to_both_retrievers():
    retriever = HybridRetriever(store_with(CHUNKS), OverlapReranker(), 10, 4)
    hits = retriever.retrieve("sanksi pegawai", source="b.pdf").hits
    assert {h.chunk.source for h in hits} == {"b.pdf"}


def test_empty_store_returns_nothing():
    store = VectorStore(None, HashEmbedder(), f"test-{uuid.uuid4().hex[:8]}")
    assert HybridRetriever(store, OverlapReranker()).retrieve("apa saja").hits == []


def test_bm25_is_rebuilt_when_another_process_changed_the_store():
    store = store_with(CHUNKS[:1])
    retriever = HybridRetriever(store, None, 10, 4)
    store.add([CHUNKS[1]], sha256="y", ingested_at="2026-01-02")  # e.g. written by the API
    assert retriever.bm25.search("gratifikasi", 10) == []
    hits = retriever.retrieve("gratifikasi").hits
    assert any(h.chunk.id == "c2" and h.bm25_rank == 1 for h in hits)


def test_only_the_best_fused_candidates_are_reranked():
    seen = []

    class CountingReranker(OverlapReranker):
        def score(self, query, texts):
            seen.append(len(texts))
            return super().score(query, texts)

    many = [chunk(i, f"Pegawai wajib melapor pasal {i}.") for i in range(30)]
    retriever = HybridRetriever(store_with(many), CountingReranker(), 20, 4, rerank_candidates=5)
    result = retriever.retrieve("pegawai wajib melapor")
    assert seen == [5] and len(result.hits) == 4


def test_warm_up_loads_both_models(engine):
    calls = engine.retriever.reranker.calls
    engine.warm_up()
    assert engine.retriever.reranker.calls == calls + 1