"""Streamlit helpers shared by the pages: cached engine, theme, formatting."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from regushield.config import load_settings
from regushield.engine import Answer, RAGEngine

PAGE_ASSISTANT = "views/assistant.py"
PAGE_DOCUMENTS = "views/documents.py"
PAGE_MONITORING = "views/monitoring.py"
ALL_DOCUMENTS = "Semua dokumen"

_CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1150px; }
.rs-title { font-size: 2rem !important; font-weight: 700; letter-spacing: -0.01em;
    line-height: 1.25; margin: 0; }
.rs-subtitle { color: #8b949e; margin: 0.35rem 0 1.2rem 0; max-width: 60rem; }
div[class*="st-key-source_"] { border-left: 3px solid #1f6feb !important; }
[data-testid="stHeaderActionElements"] { display: none; }
</style>
"""


@st.cache_resource(show_spinner="Memuat model pencarian (hanya sekali saat app mulai)...")
def get_engine() -> RAGEngine:
    load_dotenv()
    engine = RAGEngine.from_settings(load_settings())
    engine.warm_up()
    return engine


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rs-title">{title}</div><div class="rs-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def seconds(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.2f} dtk"


def percent(rate: float | None) -> str:
    return "-" if rate is None else f"{rate * 100:.0f}%"


def usd(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.4f}" if value >= 0.0001 else f"${value:.6f}"


def answer_caption(answer: Answer) -> str:
    lat = answer.latency_ms
    parts = [f"Total {seconds(lat.get('total'))}"]
    retrieval = sum(lat.get(k, 0.0) for k in ("bm25", "vector", "rerank"))
    parts.append(f"retrieval {seconds(retrieval)}")
    if "llm" in lat:
        parts.append(f"LLM {seconds(lat['llm'])}")
    if answer.ttft_ms is not None:
        parts.append(f"token pertama {seconds(answer.ttft_ms)}")
    tokens = answer.prompt_tokens + answer.completion_tokens
    if tokens:
        parts.append(f"{tokens:,} token" + (" (estimasi)" if answer.tokens_estimated else ""))
    if answer.cost_usd is not None:
        estimate = f"≈ {usd(answer.cost_usd)} jika paket berbayar"
        parts.append(estimate.replace("$", "\\$"))  # "$" starts math in Markdown
    return " · ".join(parts)