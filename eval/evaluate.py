"""Offline evaluation of the whole pipeline with an LLM judge.

    python -m eval.evaluate                      # uses eval/dataset.json
    python -m eval.evaluate --judge-model openai/gpt-oss-120b --sleep 3

For every question it runs the real engine (same settings as the app) and scores:

* faithfulness      - is every claim in the answer supported by the retrieved context?
* context_relevance - does the retrieved context contain what the question needs?
* correctness       - does the answer agree with the hand-written ground truth?
* page_hit          - optional: did retrieval return one of ``expected_pages``?

Scores the judge fails to return as JSON are reported as missing, not as 0, so a parsing
problem can never look like a bad answer. Results go to ``eval/results/``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from regushield.config import load_settings
from regushield.engine import RAGEngine
from regushield.llm import GroqChat, LLMError
from regushield.monitoring import percentile

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_DOCUMENTS = {"", "semua dokumen", "all"}

JUDGE_PROMPTS = {
    "faithfulness": (
        "Nilai apakah SETIAP klaim dalam Jawaban didukung oleh Konteks. 1.0 = semua didukung, "
        "0.0 = tidak ada yang didukung. Jika Jawaban menyatakan informasi tidak ditemukan, beri "
        "1.0 bila Konteks memang tidak memuatnya dan 0.0 bila memuatnya.\n\n"
        "Konteks:\n{context}\n\nJawaban:\n{answer}"
    ),
    "context_relevance": (
        "Nilai apakah Konteks memuat informasi yang dibutuhkan untuk menjawab Pertanyaan. "
        "1.0 = lengkap, 0.0 = tidak relevan.\n\nPertanyaan:\n{question}\n\nKonteks:\n{context}"
    ),
    "correctness": (
        "Bandingkan Jawaban dengan Jawaban Acuan. 1.0 = maknanya sama dan lengkap, 0.5 = sebagian "
        "benar, 0.0 = salah atau tidak menjawab.\n\nPertanyaan:\n{question}\n\n"
        "Jawaban Acuan:\n{ground_truth}\n\nJawaban:\n{answer}"
    ),
}
JUDGE_SYSTEM = (
    "Anda adalah auditor kualitas yang ketat dan independen. Balas HANYA dengan JSON valid: "
    '{"score": <angka 0.0 sampai 1.0>, "reason": "<alasan singkat>"}'
)


def parse_score(text: str) -> tuple[float | None, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "judge did not return JSON"
    try:
        data = json.loads(match.group(0))
        score = float(data["score"])
    except (ValueError, KeyError, TypeError):
        return None, "judge returned malformed JSON"
    return min(1.0, max(0.0, score)), str(data.get("reason", ""))


def judge(chat: GroqChat, metric: str, **fields: str) -> tuple[float | None, str]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_PROMPTS[metric].format(**fields)},
    ]
    try:
        return parse_score("".join(chat.stream(messages)))
    except LLMError as exc:
        return None, f"judge error: {exc}"


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 3) if present else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset.json")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--sleep", type=float, default=2.0, help="pause between cases (rate limits)"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    settings = load_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is required for evaluation.")
        return 2
    engine = RAGEngine.from_settings(settings)
    if not engine.documents():
        print("No documents indexed. Run: python -m regushield ingest path/to/file.pdf")
        return 2
    judge_chat = GroqChat(settings.groq_api_key, args.judge_model, "low")
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))

    rows = []
    for i, case in enumerate(cases, start=1):
        scope = case.get("document") or ""
        scope = None if scope.strip().lower() in ALL_DOCUMENTS else scope
        print(f"[{i}/{len(cases)}] {case['question'][:80]}")
        answer = engine.ask_sync(case["question"], scope)
        context = "\n\n".join(f"[{s.number}] (hal. {s.page}) {s.text}" for s in answer.sources)
        fields = {
            "question": case["question"],
            "answer": answer.text,
            "context": context or "(kosong)",
            "ground_truth": case.get("ground_truth", ""),
        }
        scores = {m: judge(judge_chat, m, **fields) for m in JUDGE_PROMPTS}
        expected = set(case.get("expected_pages") or [])
        rows.append(
            {
                "question": case["question"],
                "status": answer.status,
                "answer": answer.text,
                "pages": [s.page for s in answer.sources],
                "page_hit": bool(expected & {s.page for s in answer.sources}) if expected else None,
                "cited": answer.cited,
                "latency_ms": round(answer.latency_ms.get("total", 0.0), 1),
                "tokens": answer.prompt_tokens + answer.completion_tokens,
                "cost_usd": answer.cost_usd,
                **{m: s for m, (s, _) in scores.items()},
                "judge_notes": {m: note for m, (_, note) in scores.items()},
            }
        )
        time.sleep(args.sleep)

    latencies = [r["latency_ms"] for r in rows]
    hits = [r["page_hit"] for r in rows if r["page_hit"] is not None]
    summary = {
        "cases": len(rows),
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "context_relevance": mean([r["context_relevance"] for r in rows]),
        "correctness": mean([r["correctness"] for r in rows]),
        "judge_failures": sum(r[m] is None for r in rows for m in JUDGE_PROMPTS),
        "page_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "cited_rate": round(sum(r["cited"] for r in rows) / len(rows), 3) if rows else None,
        "not_found": sum(r["status"] == "not_found" for r in rows),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 6),
    }
    config = {
        "llm_model": settings.llm_model,
        "judge_model": args.judge_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model if settings.rerank_enabled else None,
        "chunk_size": settings.chunk_size,
        "top_k": settings.top_k,
        "context_window": settings.context_window,
        "min_relevance": settings.min_relevance,
        "documents": [d.source for d in engine.documents()],
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"created_at": stamp, "config": config, "summary": summary, "cases": rows}
    (RESULTS_DIR / f"report-{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def render_markdown(report: dict) -> str:
    s, c = report["summary"], report["config"]
    lines = [
        f"# Evaluation {report['created_at']}",
        "",
        f"LLM `{c['llm_model']}` · judge `{c['judge_model']}` · embedding `{c['embedding_model']}`"
        f" · reranker `{c['reranker_model']}` · top_k {c['top_k']} · {s['cases']} cases",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {key} | {value} |" for key, value in s.items()]
    lines += ["", "| # | Status | Faithfulness | Relevance | Correctness | Pages | Question |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for i, r in enumerate(report["cases"], start=1):
        lines.append(
            f"| {i} | {r['status']} | {r['faithfulness']} | {r['context_relevance']} | "
            f"{r['correctness']} | {r['pages']} | {r['question'][:70]} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())