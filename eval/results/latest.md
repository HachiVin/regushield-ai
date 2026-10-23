# Evaluation 20261024-010520

LLM `openai/gpt-oss-20b` · judge `openai/gpt-oss-120b` · embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` · reranker `ms-marco-MultiBERT-L-12` · top_k 4 · 8 cases

| Metric | Value |
|---|---|
| cases | 8 |
| faithfulness | 0.875 |
| context_relevance | 0.863 |
| correctness | 0.938 |
| judge_failures | 0 |
| page_hit_rate | 0.75 |
| cited_rate | 0.875 |
| not_found | 1 |
| latency_p50_ms | 2807.9 |
| latency_p95_ms | 6451.9 |
| total_cost_usd | 0.001099 |

| # | Status | Faithfulness | Relevance | Correctness | Pages | Question |
|---|---|---|---|---|---|---|
| 1 | answered | 1.0 | 1.0 | 1.0 | [12, 12, 9, 9, 12] | Apa nama saluran atau sistem whistleblowing yang disediakan oleh Bank  |
| 2 | answered | 1.0 | 1.0 | 1.0 | [12, 12, 12, 12, 13, 11] | Bagaimana klasifikasi atau kategori sanksi bagi pegawai yang terbukti  |
| 3 | answered | 1.0 | 1.0 | 1.0 | [9, 12, 9, 12] | Siapa saja pihak yang diperbolehkan untuk menyampaikan laporan dugaan  |
| 4 | answered | 1.0 | 0.9 | 0.5 | [3, 3, 3, 3] | Bagaimana aturan penerimaan hadiah atau gratifikasi? |
| 5 | answered | 1.0 | 1.0 | 1.0 | [12, 12, 11, 12] | Berapa jumlah pelanggaran kode etik tahun 2024? |
| 6 | answered | 1.0 | 1.0 | 1.0 | [11, 12, 11, 13] | Apa sanksi untuk pelanggaran kode etik kategori ringan? |
| 7 | answered | 1.0 | 1.0 | 1.0 | [3, 3, 9, 3] | Bolehkah pegawai menerima bingkisan saat Hari Raya Keagamaan? |
| 8 | not_found | 0.0 | 0.0 | 1.0 | [6, 9, 10, 10] | Berapa suku bunga KPR Bank Mandiri? |
