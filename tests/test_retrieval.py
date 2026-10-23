import uuid

import pytest

from regushield.chunking import Chunk
from regushield.retrieval import (
    BM25Index,
    HybridRetriever,
    Retrieved,
    is_open_ended,
    reciprocal_rank_fusion,
    starts_mid_list,
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


@pytest.mark.parametrize(
    ("text", "open_ended"),
    [
        ("Sanksi diklasifikasikan ke dalam tiga kategori: Ringan,", True),
        ("Pegawai wajib melapor sebagai berikut:", True),
        ("1. Website : https://contoh.id", True),
        ("Pelanggaran yang terbukti dikenakan sanksi.", False),
        ("Apakah pegawai boleh menerima hadiah?", False),
        ("Tata kelola perusahaan (Good Corporate Governance/GCG).", False),
        ('Disebut "Letter to CEO."', False),
    ],
)
def test_open_ended_detection(text, open_ended):
    assert is_open_ended(text) is open_ended


def ordered(texts: list[str], source: str = "a.pdf", start: int = 0) -> list[Chunk]:
    return [
        Chunk(f"{source}-{i}", t, source, 1 + i // 2, "text", seq=i)
        for i, t in enumerate(texts, start=start)
    ]


SANKSI = ordered(
    [
        "Pegawai dilarang menerima gratifikasi dari nasabah.",
        "Jenis sanksi pelanggaran kode etik diklasifikasikan ke dalam tiga kategori: Ringan,",
        "Sedang, dan Berat. Contoh pelanggaran berat adalah menerima suap.",
        "Rapat direksi diadakan setiap bulan.",
    ]
)


def expanded_ids(retriever, hits):
    return [(h.chunk.id, h.continues) for h in retriever.expand(hits)]


def as_hits(chunks):
    return [Retrieved(c, 0.0, None, None) for c in chunks]


def test_open_ended_hit_is_followed_by_the_next_chunk_of_its_document():
    retriever = HybridRetriever(store_with(SANKSI), OverlapReranker(), 10, 1)
    result = retriever.retrieve("jenis sanksi pelanggaran kode etik kategori")
    assert [(h.chunk.id, h.continues) for h in result.hits] == [
        ("a.pdf-1", None),
        ("a.pdf-2", "a.pdf-1"),
    ]
    assert result.hits[1].rerank_score is None  # added for context, not scored


def test_complete_hits_are_not_expanded():
    retriever = HybridRetriever(store_with(SANKSI), None, 10, 4)
    assert expanded_ids(retriever, as_hits([SANKSI[0], SANKSI[3]])) == [
        ("a.pdf-0", None),
        ("a.pdf-3", None),
    ]


def test_a_following_chunk_that_is_already_a_hit_is_not_repeated():
    retriever = HybridRetriever(store_with(SANKSI), None, 10, 4)
    assert expanded_ids(retriever, as_hits([SANKSI[1], SANKSI[2]])) == [
        ("a.pdf-1", None),
        ("a.pdf-2", None),
    ]


def test_expansion_stays_inside_the_document_and_can_be_turned_off():
    other = ordered(["Lampiran lain."], source="b.pdf")
    store = store_with(SANKSI[:2] + other)
    retriever = HybridRetriever(store, None, 10, 4)
    assert expanded_ids(retriever, as_hits([SANKSI[1]])) == [("a.pdf-1", None)]  # last chunk
    retriever = HybridRetriever(store_with(SANKSI), None, 10, 4, context_window=0)
    assert expanded_ids(retriever, as_hits([SANKSI[1]])) == [("a.pdf-1", None)]


def test_window_follows_an_unfinished_list_until_it_closes():
    chunks = ordered(["Sanksi ringan:", "1. teguran lisan,", "2. teguran tertulis.", "Penutup."])
    retriever = HybridRetriever(store_with(chunks), None, 10, 4, context_window=3)
    assert [h.chunk.id for h in retriever.expand(as_hits(chunks[:1]))] == [
        "a.pdf-0",
        "a.pdf-1",
        "a.pdf-2",
    ]


def test_tables_and_chunks_without_order_are_not_expanded():
    rows = "| Jenis | Contoh |\n| --- | --- |\n| Ringan | Teguran"
    table = Chunk("t", rows, "a.pdf", 1, "table", 0)
    legacy = [Chunk(f"old-{i}", t, "a.pdf", 1) for i, t in enumerate(["Kategori:", "Ringan."])]
    retriever = HybridRetriever(store_with([table, *legacy]), None, 10, 4)
    assert len(retriever.expand(as_hits([table, legacy[0]]))) == 2


@pytest.mark.parametrize(
    ("text", "open_ended"),
    [
        ("Program awareness dilakukan rutin.\n\n4. Program Awareness Kode Etik", False),
        ("Bagian ini.\n\nPersaingan Usaha yang Adil", False),
        ("Upaya ini dilakukan dengan penuh", True),
        (
            "Kebijakan Berperilaku Saling Menghargai di Tempat Kerja/Respectful Workplace (RWP)",
            True,
        ),
    ],
)
def test_headings_are_not_treated_as_cut_sentences(text, open_ended):
    assert is_open_ended(text) is open_ended


@pytest.mark.parametrize(
    ("text", "mid_list"),
    [
        ("2. E-mail : lapor@contoh.id 3. SMS / WA : 0811", True),
        ("c) dilarang menerima bingkisan", True),
        ("1. Website : https://contoh.id", False),
        ("Pelaporan pelanggaran kode etik", False),
        ("2024 adalah tahun pelaporan", False),
    ],
)
def test_mid_list_start_detection(text, mid_list):
    assert starts_mid_list(text) is mid_list


WBS = [
    Chunk(
        "w0", "Pelanggaran dilaporkan melalui WBS-LTC dengan media berikut:", "a.pdf", 12, "text", 0
    ),
    Chunk(
        "w1", "1. Website : https://contoh.id 2. E-mail : lapor@contoh.id", "a.pdf", 12, "text", 1
    ),
    Chunk("w2", "2. E-mail : lapor@contoh.id 3. SMS / WA : 0811-000.", "a.pdf", 12, "text", 2),
    Chunk("w3", "3. Pembuatan Pernyataan Tahunan wajib setiap tahun.", "a.pdf", 13, "text", 3),
]


def test_a_hit_that_starts_mid_list_gets_the_chunk_before_it():
    retriever = HybridRetriever(store_with(WBS), None, 10, 4)
    assert [(h.chunk.id, h.precedes, h.continues) for h in retriever.expand(as_hits(WBS[2:3]))] == [
        ("w1", "w2", None),
        ("w2", None, None),
    ]


def test_a_numbered_section_at_the_top_of_a_page_is_not_expanded_backwards():
    retriever = HybridRetriever(store_with(WBS), None, 10, 4)
    assert [h.chunk.id for h in retriever.expand(as_hits(WBS[3:]))] == ["w3"]


def test_rank_fusion_keeps_a_chunk_that_bm25_and_vector_put_first_when_the_reranker_does_not():
    from tests.fakes import ReversedReranker

    query = "jumlah pelanggaran kode etik tahun 2024"
    texts = [
        "Jumlah pelanggaran kode etik tahun 2024 adalah 1.043.",
        "Kode etik berlaku untuk seluruh pegawai.",
        "Pelanggaran dilaporkan melalui WBS.",
        "Rapat direksi diadakan setiap bulan.",
    ]
    chunks = [chunk(i, t) for i, t in enumerate(texts)]
    fused = HybridRetriever(store_with(chunks), ReversedReranker(), 10, 2, fuse_rerank=True)
    assert fused.retrieve(query).hits[0].chunk.id == "c0"
    rerank_only = HybridRetriever(store_with(chunks), ReversedReranker(), 10, 2, fuse_rerank=False)
    assert "c0" not in [h.chunk.id for h in rerank_only.retrieve(query).hits]


def test_rerank_score_is_kept_for_display_when_fused():
    from tests.fakes import ReversedReranker

    result = HybridRetriever(store_with(CHUNKS), ReversedReranker(), 10, 4).retrieve("sanksi berat")
    assert all(h.rerank_score is not None for h in result.hits)
    assert all(h.vector_score is not None for h in result.hits if h.vector_rank)


def test_only_the_top_hits_get_context_expansion():
    cut = ordered(["Kategori: Ringan,", "Sedang, Berat.", "Daftar: satu,", "dua, tiga."])
    retriever = HybridRetriever(store_with(cut), None, 10, 4, expand_top=1)
    hits = as_hits([cut[0], cut[2]])
    assert [h.chunk.id for h in retriever.expand(hits)] == ["a.pdf-0", "a.pdf-1", "a.pdf-2"]


def test_term_coverage_counts_rare_words_more_and_matches_inside_longer_words():
    docs = [chunk(i, f"Bank Mandiri menetapkan aturan nomor {i}.") for i in range(6)]
    docs.append(chunk(9, "Pegawai wajib melaporkan bingkisan hari raya kepada atasan."))
    retriever = HybridRetriever(store_with(docs), None, 10, 7)
    hits = as_hits(docs)
    coverage, matched, missing = retriever.term_coverage("Bolehkah lapor bingkisan?", hits)
    assert matched == ["lapor", "bingkisan"] and missing == ["bolehkah"]
    off_topic, _, missing = retriever.term_coverage("Berapa suku bunga KPR Bank Mandiri?", hits)
    assert missing == ["suku", "bunga", "kpr"]
    assert off_topic < 0.2 < coverage  # "bank", "mandiri" are everywhere: they barely count


def test_term_coverage_is_none_for_a_question_of_stopwords():
    retriever = HybridRetriever(store_with(CHUNKS), None, 10, 4)
    assert retriever.term_coverage("apa yang dan", as_hits(CHUNKS))[0] is None


@pytest.mark.parametrize(
    ("word", "term", "match"),
    [
        ("melaporkan", "lapor", True),
        ("pelaporan", "lapor", True),
        ("bingkisan", "bingkisan", True),
        ("dikenakan", "enak", False),  # di-k-enak-an: not an affix of "enak"
        ("informasi", "masi", False),
        ("kepadanasabah", "nasabah", False),  # OCR merged two words: no false match either
    ],
)
def test_words_match_with_indonesian_affixes_only(word, term, match):
    from regushield.retrieval import _contains

    assert _contains({word}, term) is match


def test_an_off_topic_question_sharing_only_a_hidden_substring_is_not_covered():
    docs = [chunk(i, f"Pelanggaran yang terbukti dikenakan sanksi nomor {i}.") for i in range(5)]
    retriever = HybridRetriever(store_with(docs), None, 10, 5)
    coverage, matched, _ = retriever.term_coverage("Resep nasi goreng yang enak?", as_hits(docs))
    assert coverage == 0.0 and matched == []


def test_the_top_hits_of_each_retriever_always_reach_the_reranker():
    from regushield.retrieval import _rerank_pool

    hits = {i: Retrieved(chunk(i, f"teks {i}"), 0.0, None, None) for i in range(6)}
    fused_order = [hits[i] for i in (1, 2, 3, 4, 5, 0)]  # c0: BM25 #1 only, fused last
    pool = _rerank_pool(fused_order, ["c0", "c1"], ["c2", "c1"], size=3, each=1)
    assert [h.chunk.id for h in pool] == ["c0", "c2", "c1"]


def test_trace_shows_each_candidates_rank_per_stage():
    result = HybridRetriever(store_with(CHUNKS), OverlapReranker(), 10, 2).retrieve(
        "sanksi berat pemutusan hubungan kerja"
    )
    first = result.trace[0]
    assert first["final_rank"] == 1 and first["used"] and first["page"] == 4
    assert first["bm25_rank"] == 1 and first["rerank_rank"] == 1
    assert first["preview"].startswith("Sanksi berat")
    assert [row["used"] for row in result.trace].count(True) == 2