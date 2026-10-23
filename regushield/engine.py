"""The RAG engine used by the Streamlit app, the REST API and the CLI.

ingest: PDF -> pages -> chunks -> vectors + BM25
ask:    question -> hybrid retrieval -> rerank -> relevance gate -> LLM (streamed) -> citation
        check -> query log
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from regushield.chunking import chunk_pages
from regushield.config import Settings
from regushield.llm import ChatModel, LLMError
from regushield.models import Embedder, Reranker
from regushield.monitoring import QueryLog, estimate_cost
from regushield.ocr import OcrEngine
from regushield.pdf import PDFError, read_pdf
from regushield.prompts import NOT_FOUND, build_messages, cited_numbers, is_not_found
from regushield.retrieval import HybridRetriever, RetrievalResult
from regushield.store import DocumentInfo, VectorStore

NO_LLM = "LLM belum dikonfigurasi (GROQ_API_KEY kosong). Kutipan yang relevan tetap ditampilkan."


class IngestError(Exception):
    """The upload was rejected; the message is safe to show to the user."""


@dataclass
class IngestReport:
    source: str
    status: str  # "indexed", "replaced" or "unchanged"
    pages: int = 0
    chunks: int = 0
    tables: int = 0
    ocr_images: int = 0


@dataclass
class Source:
    number: int
    source: str
    page: int
    kind: str
    text: str
    relevance: float | None
    cited: bool = False
    continues: int | None = None  # number of the source this one continues (context expansion)
    precedes: int | None = None  # number of the source this one comes right before


@dataclass
class Answer:
    id: str
    question: str
    scope: str | None
    text: str = ""
    status: str = "answered"  # "answered", "not_found" or "error"
    sources: list[Source] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    ttft_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    cost_usd: float | None = None
    model: str | None = None
    error: str | None = None
    gate: dict[str, Any] = field(default_factory=dict)  # why the LLM was or was not called

    @property
    def cited(self) -> bool:
        return any(s.cited for s in self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cited": self.cited}


def safe_pdf_name(name: str) -> str:
    base = re.sub(r"\.pdf$", "", name.replace("\\", "/").rsplit("/", 1)[-1], flags=re.I)
    stem = re.sub(r"[^\w .()\-]+", "_", base).strip(" ._") or "dokumen"
    return f"{stem}.pdf"


class AnswerStream:
    """Iterate for the answer text as it is generated; ``result`` holds the full ``Answer``."""

    def __init__(self, engine: RAGEngine, question: str, scope: str | None) -> None:
        self._engine = engine
        self.result = Answer(id=uuid.uuid4().hex[:12], question=question, scope=scope)

    def __iter__(self) -> Iterator[str]:
        engine, answer = self._engine, self.result
        started = time.perf_counter()
        try:
            yield from self._run(engine, answer)
        finally:
            answer.latency_ms["total"] = (time.perf_counter() - started) * 1000
            engine._log(answer)

    def _run(self, engine: RAGEngine, answer: Answer) -> Iterator[str]:
        if not answer.question.strip():
            answer.status, answer.error = "error", "Pertanyaan kosong."
            answer.text = answer.error
            yield answer.text
            return

        retrieval = engine.retriever.retrieve(answer.question, answer.scope)
        answer.latency_ms.update(retrieval.timings_ms)
        answer.sources = _sources(retrieval)
        passed, reason = engine.relevance_gate(retrieval)
        answer.gate = {
            "passed": passed,
            "reason": reason,
            "term_coverage": _round(retrieval.term_coverage),
            "matched_terms": retrieval.matched_terms,
            "missing_terms": retrieval.missing_terms,
            "top_similarity": _round(retrieval.top_similarity),
        }

        if not passed:
            answer.status, answer.text = "not_found", NOT_FOUND
            yield NOT_FOUND
            return
        if engine.chat is None:
            answer.status, answer.error, answer.text = "error", NO_LLM, NO_LLM
            yield NO_LLM
            return

        answer.model = engine.chat.model
        llm_started = time.perf_counter()
        completion = engine.chat.stream(build_messages(answer.question, retrieval.hits))
        try:
            for delta in completion:
                answer.text += delta
                yield delta
        except LLMError as exc:
            answer.status, answer.error = "error", str(exc)
            notice = ("\n\n" if answer.text else "") + f"⚠️ {exc}"
            answer.text += notice
            yield notice
            return
        finally:
            answer.latency_ms["llm"] = (time.perf_counter() - llm_started) * 1000
            answer.ttft_ms = completion.ttft_ms
            if completion.usage:
                answer.prompt_tokens = completion.usage.prompt_tokens
                answer.completion_tokens = completion.usage.completion_tokens
                answer.tokens_estimated = completion.usage.estimated
                answer.cost_usd = estimate_cost(
                    answer.prompt_tokens,
                    answer.completion_tokens,
                    engine.settings.prices_for(answer.model),
                )

        if is_not_found(answer.text):
            answer.status = "not_found"
        for number in cited_numbers(answer.text, len(answer.sources)):
            answer.sources[number - 1].cited = True


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _sources(retrieval: RetrievalResult) -> list[Source]:
    numbers = {h.chunk.id: n for n, h in enumerate(retrieval.hits, start=1)}
    return [
        Source(
            n,
            h.chunk.source,
            h.chunk.page,
            h.chunk.kind,
            h.chunk.text,
            h.rerank_score,
            continues=numbers.get(h.continues) if h.continues else None,
            precedes=numbers.get(h.precedes) if h.precedes else None,
        )
        for n, h in enumerate(retrieval.hits, start=1)
    ]


class RAGEngine:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        reranker: Reranker | None,
        chat: ChatModel | None,
        log: QueryLog,
        ocr: OcrEngine | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.chat = chat
        self.log = log
        self.ocr = ocr
        self.retriever = HybridRetriever(
            store,
            reranker,
            settings.candidates,
            settings.top_k,
            settings.rerank_candidates,
            settings.context_window,
            settings.expand_top,
            settings.fuse_rerank,
        )
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        chat: ChatModel | None = None,
        ocr: OcrEngine | None = None,
    ) -> RAGEngine:
        from regushield import ocr as ocr_module
        from regushield.llm import GroqChat
        from regushield.models import FastEmbedEmbedder, FlashRankReranker

        cache = settings.model_cache_dir
        embedder = embedder or FastEmbedEmbedder(settings.embedding_model, cache)
        if reranker is None and settings.rerank_enabled:
            reranker = FlashRankReranker(settings.reranker_model, cache, settings.rerank_max_tokens)
        if chat is None and settings.groq_api_key:
            chat = GroqChat(settings.groq_api_key, settings.llm_model, settings.reasoning_effort)
        if ocr is None and settings.ocr_enabled and ocr_module.available():
            ocr = ocr_module.RapidOcrEngine()
        store = VectorStore(settings.index_dir, embedder, settings.collection_name)
        return cls(settings, store, reranker, chat, QueryLog(settings.log_path), ocr)

    @property
    def index_version(self) -> str:
        """How documents are parsed and chunked now; a document indexed differently is
        re-indexed when uploaded again."""
        s = self.settings
        return f"3|{s.chunk_size}|{s.chunk_overlap}|ocr={'on' if self.ocr else 'off'}"

    def warm_up(self) -> None:
        """Load the embedding and reranker models now, so the first question is not slow."""
        self.store.embedder.embed_query("pemanasan")
        if self.retriever.reranker is not None:
            self.retriever.reranker.score("pemanasan", ["pemanasan model"])

    # --- documents --------------------------------------------------------------------
    def documents(self) -> list[DocumentInfo]:
        return self.store.documents()

    def ingest(self, filename: str, data: bytes) -> IngestReport:
        source = safe_pdf_name(filename)
        if len(data) > self.settings.max_file_mb * 1024 * 1024:
            raise IngestError(f"Ukuran file melebihi {self.settings.max_file_mb} MB.")
        sha256 = hashlib.sha256(data).hexdigest()
        existing = {d.source: d for d in self.documents()}
        # Same file again: nothing to do, unless it was indexed with other settings.
        doc = existing.get(source)
        current = doc and doc.ordered and doc.index_version == self.index_version
        if current and doc.sha256 == sha256:
            return IngestReport(
                source, "unchanged", doc.pages, doc.chunks, doc.tables, doc.ocr_chunks
            )
        duplicate = next(
            (d for d in existing.values() if d.sha256 == sha256 and d.source != source), None
        )
        if duplicate is not None:
            raise IngestError(f"Isi file ini sudah terindeks sebagai '{duplicate.source}'.")
        try:
            pages = read_pdf(data, self.ocr, self.settings.ocr_max_images)
        except PDFError as exc:
            raise IngestError(str(exc)) from exc

        chunks = chunk_pages(pages, source, self.settings.chunk_size, self.settings.chunk_overlap)
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock:
            if source in existing:
                self.store.delete_source(source)
            self.store.add(chunks, sha256, now, self.index_version)
            self.retriever.refresh()
            self.settings.raw_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.raw_dir / source).write_bytes(data)
        return IngestReport(
            source,
            "replaced" if source in existing else "indexed",
            pages=len(pages),
            chunks=len(chunks),
            tables=sum(1 for c in chunks if c.kind == "table"),
            ocr_images=sum(p.ocr_images for p in pages),
        )

    def delete(self, source: str) -> bool:
        if source not in {d.source for d in self.documents()}:
            return False
        with self._lock:
            self.store.delete_source(source)
            self.retriever.refresh()
            (self.settings.raw_dir / safe_pdf_name(source)).unlink(missing_ok=True)
        return True

    # --- questions --------------------------------------------------------------------
    def relevance_gate(self, retrieval: RetrievalResult) -> tuple[bool, str]:
        """Decide whether the retrieved chunks are worth sending to the LLM, and why.

        The reranker score alone could not do this: on Indonesian text it gave off-topic
        questions ("suku bunga KPR") ~0.999 too. So the best chunk must also contain enough of
        the question's words, or be clearly similar in meaning. The gate is deliberately loose:
        it saves the LLM call for clearly unrelated questions, and the LLM, told to answer only
        from the passages, stays the judge for the rest.
        """
        if not retrieval.hits:
            return False, "no_hits"
        scores = [h.rerank_score for h in retrieval.hits if h.rerank_score is not None]
        if scores and max(scores) < self.settings.min_relevance:
            return False, "low_rerank"
        coverage = retrieval.term_coverage
        if coverage is None or coverage >= self.settings.min_term_coverage:
            return True, "terms_matched"
        similarity = retrieval.top_similarity
        if similarity is not None and similarity >= self.settings.min_similarity:
            return True, "similar"
        return False, "off_topic"

    def passes_relevance_gate(self, retrieval: RetrievalResult) -> bool:
        """Skip the LLM when nothing relevant was found: cheaper, and no chance to invent."""
        return self.relevance_gate(retrieval)[0]

    def ask(self, question: str, scope: str | None = None) -> AnswerStream:
        return AnswerStream(self, question, scope)

    def ask_sync(self, question: str, scope: str | None = None) -> Answer:
        stream = self.ask(question, scope)
        for _ in stream:
            pass
        return stream.result

    def _log(self, answer: Answer) -> None:
        top = [s.relevance for s in answer.sources if s.relevance is not None]
        self.log.log_query(
            {
                "id": answer.id,
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "question": answer.question if self.settings.log_questions else None,
                "question_chars": len(answer.question),
                "scope": answer.scope,
                "status": answer.status,
                "model": answer.model,
                "latency_ms": {k: round(v, 1) for k, v in answer.latency_ms.items()},
                "ttft_ms": round(answer.ttft_ms, 1) if answer.ttft_ms is not None else None,
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "tokens_estimated": answer.tokens_estimated,
                "cost_usd": answer.cost_usd,
                "top_relevance": round(max(top), 4) if top else None,
                "n_sources": len(answer.sources),
                "cited": answer.cited,
                "gate": answer.gate,
                "error": answer.error,
            }
        )

    def feedback(self, answer_id: str, positive: bool) -> None:
        self.log.log_feedback(answer_id, positive)