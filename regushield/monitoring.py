"""Query log (JSON Lines) and the numbers shown on the Monitoring page and ``/metrics``.

Each answered question is one line: latency per stage, time to first token, tokens, estimated
cost, top relevance, whether the answer cited its sources, and errors. User feedback is
appended as separate lines that point to the query id.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any


def estimate_cost(prompt_tokens: int, completion_tokens: int, prices: tuple[float, float] | None):
    if prices is None:
        return None
    price_in, price_out = prices
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 8)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


class QueryLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_query(self, record: dict[str, Any]) -> None:
        self._append({"type": "query", **record})

    def log_feedback(self, query_id: str, positive: bool) -> None:
        self._append({"type": "feedback", "query_id": query_id, "positive": positive})

    def read(self) -> list[dict[str, Any]]:
        """Queries with their latest feedback attached; malformed lines are skipped."""
        if not self.path.exists():
            return []
        queries: dict[str, dict[str, Any]] = {}
        feedback: dict[str, bool] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("type") == "query" and item.get("id"):
                queries[item["id"]] = item
            elif item.get("type") == "feedback":
                feedback[item.get("query_id", "")] = bool(item.get("positive"))
        for query_id, positive in feedback.items():
            if query_id in queries:
                queries[query_id]["feedback"] = positive
        return list(queries.values())


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    ok = [r for r in records if r.get("status") != "error"]
    answered = [r for r in records if r.get("status") == "answered"]
    latencies = [r["latency_ms"]["total"] for r in ok if r.get("latency_ms")]
    ttfts = [r["ttft_ms"] for r in answered if r.get("ttft_ms") is not None]
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in answered if "prompt_tokens" in r]
    rated = [r for r in records if "feedback" in r]
    return {
        "queries": total,
        "answered": len(answered),
        "not_found_rate": _rate(sum(r.get("status") == "not_found" for r in records), total),
        # "not found" decided by the relevance gate, before (and instead of) an LLM call
        "gated_rate": _rate(sum(r.get("gate", {}).get("passed") is False for r in records), total),
        "error_rate": _rate(total - len(ok), total),
        "uncited_rate": _rate(sum(not r.get("cited") for r in answered), len(answered)),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "ttft_p50_ms": percentile(ttfts, 50),
        "avg_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 8) if costs else None,
        "feedback_count": len(rated),
        "positive_feedback_rate": _rate(sum(r["feedback"] for r in rated), len(rated)),
    }