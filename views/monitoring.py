"""Monitoring: latency, cost, grounding and feedback, computed from the query log."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from regushield.monitoring import summarize
from views import common

STAGES = {"bm25": "BM25", "vector": "Vector", "rerank": "Rerank", "llm": "LLM"}
MIN_POINTS_FOR_CHART = 5  # a trend line over one or two points says nothing


def kpis(summary: dict) -> None:
    row = st.columns(4)
    row[0].metric("Pertanyaan", summary["queries"])
    row[1].metric("Latency p50", common.seconds(summary["latency_p50_ms"]))
    row[2].metric("Latency p95", common.seconds(summary["latency_p95_ms"]))
    row[3].metric("Token pertama p50", common.seconds(summary["ttft_p50_ms"]))
    row = st.columns(4)
    row[0].metric(
        "Tidak ditemukan",
        common.percent(summary["not_found_rate"]),
        help="Pertanyaan yang dijawab 'tidak ditemukan' (gerbang relevansi atau LLM).",
    )
    row[1].metric(
        "Ditolak gerbang",
        common.percent(summary["gated_rate"]),
        help="Pertanyaan di luar topik dokumen yang dijawab 'tidak ditemukan' tanpa memanggil LLM.",
    )
    row[2].metric(
        "Tanpa sitasi",
        common.percent(summary["uncited_rate"]),
        help="Jawaban yang tidak menyebut nomor kutipan: tanda jawaban perlu dicek.",
    )
    row[3].metric(
        "Feedback positif",
        common.percent(summary["positive_feedback_rate"]),
        help=f"Dari {summary['feedback_count']} jawaban yang diberi feedback.",
    )
    row = st.columns(4)
    row[0].metric("Rata-rata token", summary["avg_tokens"] or "-")
    row[1].metric("Estimasi biaya / pertanyaan", common.usd(summary["avg_cost_usd"]))
    row[2].metric("Estimasi total biaya", common.usd(summary["total_cost_usd"]))
    row[3].metric("Error", common.percent(summary["error_rate"]))


def main() -> None:
    engine = common.get_engine()
    common.header(
        "Monitoring",
        "Kesehatan sistem dari log setiap pertanyaan: kecepatan, biaya, "
        "dan seberapa sering jawaban menyebut sumbernya.",
    )
    records = engine.log.read()
    if not records:
        st.info(
            "Belum ada pertanyaan yang tercatat. Ajukan pertanyaan di halaman Assistant.",
            icon=":material/query_stats:",
        )
        return

    kpis(summarize(records))
    prices = engine.settings.prices_for(engine.chat.model if engine.chat else None)
    st.caption(
        "Di Groq Free tier tidak ada tagihan (akun tanpa metode pembayaran). Angka biaya adalah "
        "estimasi jika akun memakai paket berbayar: token × harga list "
        + (
            f"(\\${prices[0]} / \\${prices[1]} per 1 juta token input/output)."
            if prices
            else "(harga model ini belum diatur)."
        )
    )

    frame = pd.DataFrame(
        [
            {
                "Waktu": r.get("ts", "")[:19].replace("T", " "),
                "Status": r.get("status"),
                "Pertanyaan": r.get("question") or "(tidak dicatat)",
                "Latency (dtk)": round(r.get("latency_ms", {}).get("total", 0) / 1000, 2),
                "Token": r.get("prompt_tokens", 0) + r.get("completion_tokens", 0),
                "Biaya (USD)": r.get("cost_usd"),
                "Relevansi teratas": r.get("top_relevance"),
                "Dikutip": r.get("cited"),
                "Feedback": {True: "positif", False: "negatif"}.get(r.get("feedback"), ""),
                **{
                    label: round(r.get("latency_ms", {}).get(key, 0) / 1000, 3)
                    for key, label in STAGES.items()
                },
            }
            for r in records
        ]
    )

    st.markdown("**Rata-rata waktu per tahap**")
    averages = frame[list(STAGES.values())].mean()
    slowest = averages.idxmax()
    row = st.columns(len(STAGES))
    for column, (label, value) in zip(row, averages.items(), strict=True):
        column.metric(
            label, f"{value:.2f} dtk", help="Tahap paling lambat" if label == slowest else None
        )
    st.caption(f"Tahap paling lambat: **{slowest}**. Bandingkan setelah mengubah pengaturan.")

    if len(frame) >= MIN_POINTS_FOR_CHART:
        st.markdown("**Latency per pertanyaan (detik)**")
        series = frame[["Latency (dtk)"]].reset_index(names="Pertanyaan ke-")
        series["Pertanyaan ke-"] += 1
        st.line_chart(series, x="Pertanyaan ke-", y="Latency (dtk)", height=220)

    st.markdown("**Pertanyaan terakhir**")
    st.dataframe(
        frame.drop(columns=list(STAGES.values())).iloc[::-1].head(50),
        hide_index=True,
        width="stretch",
        column_config={
            "Biaya (USD)": st.column_config.NumberColumn(format="%.6f"),
            "Relevansi teratas": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    st.caption(f"Log: `{engine.settings.log_path}` (JSON Lines, satu baris per pertanyaan).")


main()