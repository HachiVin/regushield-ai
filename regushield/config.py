"""Runtime settings, read from environment variables (or a local .env file).

Every value has a default so the app starts without configuration; only ``GROQ_API_KEY`` is
needed to generate answers. On Streamlit Community Cloud, top-level secrets are exposed as
environment variables, so the same names work there.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Groq list prices in USD per 1M tokens (input, output), used only to *estimate* cost in the
# monitoring view. Verify at https://console.groq.com/docs/models before relying on them, and
# override with REGUSHIELD_PRICE_INPUT_PER_M / REGUSHIELD_PRICE_OUTPUT_PER_M.
# Source: cloudprice.net, "Groq openai/gpt-oss-20b", updated 2026-10-22.
LIST_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = BASE_DIR / "data"
    groq_api_key: str | None = None
    llm_model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "low"  # gpt-oss only: low reasoning keeps latency and cost down
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "ms-marco-MultiBERT-L-12"  # multilingual cross-encoder (FlashRank)
    rerank_enabled: bool = True
    rerank_candidates: int = 10  # fused candidates the cross-encoder rescores (latency!)
    rerank_max_tokens: int = 256  # tokens per question+passage pair for the reranker
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120
    candidates: int = 20  # taken from each retriever (BM25 and vector) before fusion
    top_k: int = 4  # chunks sent to the LLM
    min_relevance: float = 0.10  # rerank score below this for every chunk -> "not found"
    max_file_mb: int = 25
    api_key: str | None = None  # protects the REST API when set
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None
    log_questions: bool = True

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "queries.jsonl"

    @property
    def model_cache_dir(self) -> Path:
        return Path(os.environ.get("REGUSHIELD_MODEL_CACHE", self.data_dir / "models"))

    @property
    def collection_name(self) -> str:
        """One collection per embedding model, so switching models never mixes vectors."""
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.embedding_model).strip("-").lower()
        return f"chunks-{slug}"[:60].rstrip("-")

    def prices_for(self, model: str | None) -> tuple[float, float] | None:
        """(input, output) USD per 1M tokens: the configured override, else the list price."""
        if self.price_input_per_m is not None and self.price_output_per_m is not None:
            return self.price_input_per_m, self.price_output_per_m
        return LIST_PRICES_PER_M.get(model or "")


def _get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    return value.strip() if value and value.strip() else None


def _float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    value = _get(env, name)
    return float(value) if value is not None else default


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _get(env, name)
    return int(value) if value is not None else default


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = _get(env, name)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    d = Settings()
    return Settings(
        data_dir=Path(_get(env, "REGUSHIELD_DATA_DIR") or d.data_dir),
        groq_api_key=_get(env, "GROQ_API_KEY"),
        llm_model=_get(env, "REGUSHIELD_LLM_MODEL") or d.llm_model,
        reasoning_effort=_get(env, "REGUSHIELD_REASONING_EFFORT") or d.reasoning_effort,
        embedding_model=_get(env, "REGUSHIELD_EMBEDDING_MODEL") or d.embedding_model,
        reranker_model=_get(env, "REGUSHIELD_RERANKER_MODEL") or d.reranker_model,
        rerank_enabled=_bool(env, "REGUSHIELD_RERANK", d.rerank_enabled),
        rerank_candidates=_int(env, "REGUSHIELD_RERANK_CANDIDATES", d.rerank_candidates),
        rerank_max_tokens=_int(env, "REGUSHIELD_RERANK_MAX_TOKENS", d.rerank_max_tokens),
        chunk_size=_int(env, "REGUSHIELD_CHUNK_SIZE", d.chunk_size),
        chunk_overlap=_int(env, "REGUSHIELD_CHUNK_OVERLAP", d.chunk_overlap),
        candidates=_int(env, "REGUSHIELD_CANDIDATES", d.candidates),
        top_k=_int(env, "REGUSHIELD_TOP_K", d.top_k),
        min_relevance=_float(env, "REGUSHIELD_MIN_RELEVANCE", d.min_relevance) or 0.0,
        max_file_mb=_int(env, "REGUSHIELD_MAX_FILE_MB", d.max_file_mb),
        api_key=_get(env, "REGUSHIELD_API_KEY"),
        price_input_per_m=_float(env, "REGUSHIELD_PRICE_INPUT_PER_M", None),
        price_output_per_m=_float(env, "REGUSHIELD_PRICE_OUTPUT_PER_M", None),
        log_questions=_bool(env, "REGUSHIELD_LOG_QUESTIONS", d.log_questions),
    )