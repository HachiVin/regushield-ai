# ReguShield: Grounded Q&A for Regulatory Documents

ReguShield answers questions about regulatory and compliance PDFs (OJK regulations, internal
SOPs, codes of conduct). Every answer cites the source document and page. When the documents
do not contain the answer, it says so instead of guessing.

**Live demo:** https://regushield-ai-bnf3fmcobcvkp5ne3lg5en.streamlit.app/

[ISI: screenshot of the Assistant page after uploading a real document]

## The problem

Compliance, legal and HR staff answer questions like "may an employee accept a holiday gift
from a client?" by searching long regulations by hand. A general-purpose chatbot answers
faster, but it can invent an article number or a sanction that does not exist, and a wrong
compliance answer is worse than a slow one. ReguShield keeps the speed of a chatbot and adds
three safeguards:

1. It answers only from retrieved passages of *your* documents.
2. It shows those passages with file name and page, so every claim can be checked.
3. It refuses ("Informasi tersebut tidak ditemukan…") when nothing relevant was retrieved.

## Features

- **Hybrid retrieval.** BM25 catches exact terms and article numbers ("Pasal 12", "POJK 51").
  Multilingual vector search catches paraphrases ("hadiah" vs "gratifikasi"). The two
  rankings are merged with Reciprocal Rank Fusion.
- **Cross-encoder reranking, fused with the retrievers.** A multilingual reranker (FlashRank,
  ONNX) rescores the merged candidates. The final order is Reciprocal Rank Fusion over three
  rankings (BM25, vector, reranker), not the reranker score alone: on Indonesian text the
  reranker gave almost every passage ~0.999, and sorting by it alone dropped a passage that BM25
  ranked first ("tahun 2024 ... 1.043 pelanggaran").
- **OCR for text inside images.** Tables pasted as pictures and scanned pages are read with
  RapidOCR (ONNX, CPU, models bundled, no download). The text keeps its place on the page,
  each OCR passage carries the heading above the image, and it is labelled "OCR" in the
  sources and in the prompt. Logos repeated on every page are skipped.
- **Context expansion.** When a retrieved passage stops mid-sentence or mid-list ("...three
  categories: Light,"), the next passage of the same document is added after it and labelled
  as its continuation. The rest of a list rarely shares words with the question, so it would
  never be retrieved on its own. A passage that starts in the middle of a numbered list
  ("2. E-mail ...") gets the passage before it on the same page, where the list begins.
  Section headings ("4. Program Awareness") are not treated as cut-off sentences.
- **Relevance gate.** When the best passage holds too little of the question's words (weighted
  by rarity) and no passage is close in meaning, the LLM is not called. The gate is
  deliberately loose (see trade-offs); the LLM, told to answer only from the passages, stays
  the final judge.
- **Cited, streamed answers.** Groq (`openai/gpt-oss-20b`) streams the answer with `[n]`
  markers. The app checks which sources were actually cited and flags uncited answers.
- **PDF handling.** Pages are numbered 1-based, so citations match the printed page. Tables
  are extracted as Markdown. Running headers, footers and page numbers are removed before
  indexing.
- **Document management.** Upload several PDFs, re-upload a new version (the old one is
  replaced), skip unchanged files, reject duplicates, delete documents.
- **Monitoring.** Every question is logged with latency per stage, time to first token,
  tokens, estimated cost, top relevance, whether sources were cited, errors and 👍/👎
  feedback. The Monitoring page and the `/metrics` endpoint show p50/p95 latency, cost per
  question, and not-found, uncited and error rates.
- **Same engine, three interfaces.** Streamlit UI, REST API (FastAPI) and CLI, packaged with
  Docker Compose.
- **Offline evaluation.** An LLM judge scores faithfulness, context relevance and
  correctness against hand-written answers.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        PDF[PDF upload] --> Parse[PyMuPDF<br/>text + tables<br/>header/footer cleanup]
        PDF --> OCR[RapidOCR<br/>images with text]
        OCR --> Chunk
        Parse --> Chunk[Chunks<br/>one page each]
        Chunk --> Embed[Multilingual<br/>embeddings]
        Embed --> Chroma[(ChromaDB)]
        Chunk --> BM25[(BM25 index)]
    end
    subgraph Ask
        Q[Question] --> S[BM25 top 20]
        Q --> V[Vector top 20]
        S --> RRF[Reciprocal Rank Fusion]
        V --> RRF
        RRF --> RR[Cross-encoder rerank]
        RR --> Fuse[RRF of BM25,<br/>vector, rerank ranks]
        Fuse --> Gate{question words in<br/>a passage, or similar?}
        Gate -- no --> NF[Not found<br/>no LLM call]
        Gate -- yes --> Exp[Context expansion<br/>next chunk of cut-off hits]
        Exp --> LLM[Groq LLM<br/>streamed, cited]
        LLM --> Check[Citation check]
    end
    Chroma -.-> V
    BM25 -.-> S
    NF --> Log[(Query log<br/>JSONL)]
    Check --> Log
    Log --> Mon[Monitoring page<br/>/metrics]
```

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| BM25 + vector search, merged with RRF | Legal text mixes exact identifiers with paraphrased concepts; each method misses what the other finds. RRF merges ranks without calibrating two score scales. | Two indexes to keep in sync. BM25 is rebuilt in memory from the vector store on start-up. |
| Cross-encoder reranker on the top 10 fused candidates | Reads question and passage together, which is more accurate than comparing embeddings. | The slowest stage on CPU. Rescoring 20 candidates at 512 tokens was the bottleneck in local tests, so the default is 10 candidates at 256 tokens; the Monitoring page shows the effect. |
| Final order = RRF(BM25, vector, reranker) | Audit of 53 retrieved passages: every reranker score was between 0.995 and 0.9996, so its order among good candidates was close to noise. In one case it pushed the passage BM25 ranked first out of the top 4 and the answer became "not found". With fusion a passage two rankings agree on stays on top. | The reranker can no longer overrule both retrievers. `REGUSHIELD_FUSE_RERANK=false` restores reranker-only ordering, e.g. for a stronger reranker. |
| Relevance gate by term coverage, not reranker score | The reranker gave an off-topic question ("suku bunga KPR") ~0.999 too, so a score threshold let everything through. Measured on the Bank Mandiri code of conduct: off-topic questions had 0.00–0.60 of their (rarity-weighted) words in one passage ("harga saham" matches the insider-trading text), and in a live audit an answerable question ("karyawan ... hadiah dari vendor", where the document says "gratifikasi") had only 0.18. | The ranges overlap, so no threshold separates them. A first default of 0.20 refused that answerable question, so the default is 0.10: it stops only questions that share almost nothing with the documents ("resep nasi goreng"); the rest go to the LLM, which refuses when the passages do not answer, at the cost of one LLM call. |
| OCR with RapidOCR (ONNX), images upscaled 2× | In the test document the sanctions table, the violation counts and the gift rules were images: without OCR, about half of the content could not be answered. The model ships in the wheel and runs on CPU, so it stays free and offline. Reading the image's own pixels and upscaling them with a cubic filter removed most misread letters seen when rendering the page area. | Slower ingestion (about 40 s for the 9 images of the test document on the test machine's CPU). The recognizer is trained on Chinese/English text, so Indonesian words are sometimes merged ("kepadanasabah"); OCR passages are labelled so users can check them. Needs `libgl1` and `libglib2.0-0` on Linux (`packages.txt`, Dockerfile). |
| ONNX models (FastEmbed, FlashRank), no PyTorch | Much smaller install and faster cold start on free hosting; CPU only. | Limited to the models these libraries ship. |
| `paraphrase-multilingual-MiniLM-L12-v2` embeddings, 500-character chunks | Small (~0.2 GB) and multilingual; the old English-only MiniLM matched Indonesian text poorly. It reads only ~128 tokens, about 400–500 characters: with 800-character chunks the end of a chunk (where "tahun 2024 ... 1.043" was) most likely never reached the vector. | More, smaller chunks, so the LLM gets 5 instead of 4 and context expansion matters more. |
| Chunks never cross a page | Every chunk cites exactly one page. | A sentence spanning two pages is split; context expansion reattaches it. |
| Context expansion only for cut-off hits, and only for the top 3 | Found in testing: "examples of violations" returned *not found* because the passage ended at "three categories: Light," and the list was in the next chunk; and the "Website" item of a list was in the chunk *before* the hit. The next (or previous) chunk is added only when a hit is cut off. | In one audit 17 passages were added and only 1 was cited, so expansion is limited to the 3 best hits. The check is a punctuation heuristic; `REGUSHIELD_CONTEXT_WINDOW=0` turns it off. |
| Groq `gpt-oss-20b`, temperature 0, low reasoning effort | Fast, cheap, deterministic enough for audit use. | Hosted API, so documents' passages leave the machine. |
| JSON Lines query log | Zero infrastructure; easy to inspect, export or ship to a real monitoring stack. | Not a time-series database; fine for a single instance. |

## Evaluation

`python -m eval.evaluate` runs every question in `eval/dataset.json` through the real
pipeline. A judge model (`openai/gpt-oss-120b` by default) then scores three things:

- **Faithfulness:** is every claim supported by the retrieved passages?
- **Context relevance:** do the passages contain what the question needs?
- **Correctness:** does the answer agree with the hand-written ground truth?

If the dataset lists `expected_pages`, the page hit rate is reported too. Judge outputs
that are not valid JSON are reported as missing, never as 0. Results are written to
`eval/results/` as a JSON report and as `latest.md`.

| Metric | Result |
|---|---|
| Faithfulness | [ISI: setelah menjalankan evaluasi] |
| Context relevance | [ISI] |
| Correctness | [ISI] |
| Latency p50 / p95 | [ISI] |
| Cases | [ISI: jumlah pertanyaan di dataset] |

The questions and ground-truth answers are hand-written from a code-of-conduct document
([ISI: nama dan sumber dokumen]). It is small; extending it is the most useful next step for
tuning the threshold and chunk size.

## Cost: free to run

- **Embedding and reranker models** are open-source and run on your own CPU (FastEmbed,
  FlashRank). They are downloaded once from Hugging Face; no account or API key is needed.
- **The LLM** runs on Groq. On the Free tier (an account without a payment method) nothing is
  billed. When a rate limit is reached, Groq refuses the request and the app shows a "batas
  request" message. Charges only start after upgrading to a paid tier and adding a payment
  method, where spend limits can be set.
- The cost figures on the Monitoring page are *estimates* of what the same usage would cost
  on a paid tier, for capacity planning. They are not charges.

## Run it

### Local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env              # then put your GROQ_API_KEY in .env
streamlit run app.py
```

The embedding and reranker models (a few hundred MB) are downloaded on first use into
`data/models/`, so the first upload or question takes longer.

### Docker Compose (UI + API)

```bash
docker compose up --build
# UI  http://localhost:8501
# API http://localhost:8000/docs
```

Both services share one volume, so a document uploaded in the UI can be queried through the
API. ChromaDB runs embedded in each process; for many concurrent writers, run ChromaDB as a
server instead.

### Streamlit Community Cloud

Set `GROQ_API_KEY` under *Settings → Secrets* (see `.streamlit/secrets.toml.example`).
`packages.txt` installs `libgl1` and `libglib2.0-0`, which OpenCV (used by the OCR) needs; if
OCR is unavailable, the Documents page shows the exact import error. Storage there is temporary, so uploaded
documents and logs are lost when the app restarts.

### REST API

```bash
curl -F "file=@kode_etik.pdf" http://localhost:8000/documents
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "Apa saja jenis sanksi pelanggaran kode etik?"}'
curl http://localhost:8000/metrics
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, number of documents and chunks, whether the LLM is configured |
| `GET /documents` · `POST /documents` · `DELETE /documents/{name}` | Manage indexed PDFs |
| `POST /query` | Answer with sources, latency breakdown, tokens and cost |
| `GET /metrics` | Monitoring summary |

Set `REGUSHIELD_API_KEY` to require an `X-API-Key` header on everything except `/health`.

### CLI

```bash
python -m regushield ingest docs/*.pdf
python -m regushield ask "Kepada siapa benturan kepentingan dilaporkan?"
python -m regushield metrics
```

## Configuration

All settings are environment variables (or `.env`). Only `GROQ_API_KEY` is required.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | – | Groq key; without it the app still retrieves passages but does not answer |
| `REGUSHIELD_LLM_MODEL` | `openai/gpt-oss-20b` | Answer model |
| `REGUSHIELD_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Changing it builds a new index |
| `REGUSHIELD_RERANKER_MODEL` | `ms-marco-MultiBERT-L-12` | `REGUSHIELD_RERANK=false` disables reranking |
| `REGUSHIELD_RERANK_CANDIDATES` / `REGUSHIELD_RERANK_MAX_TOKENS` | `10` / `256` | Passages the reranker rescores, and tokens per pair: the main latency knobs on CPU |
| `REGUSHIELD_FUSE_RERANK` | `true` | Final order from RRF of BM25, vector and reranker ranks |
| `REGUSHIELD_MIN_TERM_COVERAGE` / `REGUSHIELD_MIN_SIMILARITY` | `0.10` / `0.60` | Relevance gate: share of question words in the best passage, or vector similarity |
| `REGUSHIELD_MIN_RELEVANCE` | `0.10` | Gate on the reranker score (rarely triggers, see trade-offs) |
| `REGUSHIELD_TOP_K` / `REGUSHIELD_CANDIDATES` | `5` / `20` | Passages sent to the LLM / taken from each retriever |
| `REGUSHIELD_CONTEXT_WINDOW` / `REGUSHIELD_EXPAND_TOP` | `1` / `3` | Passages added around a cut-off hit (`0` = off) / only for the best N hits |
| `REGUSHIELD_CHUNK_SIZE` / `REGUSHIELD_CHUNK_OVERLAP` | `500` / `80` | Characters; changing them asks to re-upload documents |
| `REGUSHIELD_OCR` / `REGUSHIELD_OCR_MAX_IMAGES` | `true` / `50` | OCR for images in PDFs / maximum images read per document |
| `REGUSHIELD_PRICE_INPUT_PER_M` / `..._OUTPUT_PER_M` | list price | USD per 1M tokens, used for the cost estimate |
| `REGUSHIELD_LOG_QUESTIONS` | `true` | `false` keeps question text out of the log |
| `REGUSHIELD_API_KEY` | – | Protects the REST API |
| `REGUSHIELD_DATA_DIR` | `data` | Uploaded PDFs, index, logs, models |

## Project structure

```
.
├── app.py                 # Streamlit entry point (navigation)
├── views/                 # Streamlit pages: assistant, documents, monitoring
├── regushield/            # Engine, independent of the UI
│   ├── config.py          # Settings from environment
│   ├── pdf.py             # PDF -> pages (text, tables, OCR segments, cleanup)
│   ├── ocr.py             # RapidOCR engine, rows/columns from OCR boxes
│   ├── chunking.py        # Page-bounded chunks
│   ├── models.py          # Embedding and reranker models (ONNX)
│   ├── store.py           # ChromaDB store
│   ├── retrieval.py       # BM25, RRF, rerank fusion, context expansion, term coverage
│   ├── prompts.py         # Grounded prompt, citation parsing
│   ├── llm.py             # Groq streaming client
│   ├── engine.py          # Ingest / ask pipeline, relevance gate, logging
│   ├── monitoring.py      # Query log and summary metrics
│   ├── api.py             # FastAPI service
│   └── __main__.py        # CLI
├── eval/                  # Dataset and LLM-judge evaluation
├── tests/                 # pytest suite (no network: fake models and LLM)
├── Dockerfile, docker-compose.yml
├── packages.txt           # System library for OCR on Streamlit Community Cloud
└── requirements.txt, requirements-dev.txt, pyproject.toml
```

## Tests

```bash
pytest
ruff check .
```

The suite runs offline in about 40 seconds. Deterministic fakes replace the embedding model,
reranker, OCR and LLM, including a reranker that scores everything ~0.999 like the real one
did. It covers PDF parsing (generated PDFs with tables, headers and images), OCR placement and
the real OCR model when installed, chunking, BM25/RRF/rerank fusion, context expansion, the
relevance gate, citation checks, error handling, monitoring, the REST API, the CLI and the
Streamlit pages end to end.

## Limitations

- OCR text can contain merged words or misread letters, and handwriting is not supported.
  Ingesting a PDF with many images takes minutes on a small CPU.
- Retrieval quality depends on the threshold and chunk size, which should be tuned with a
  larger evaluation set than the one included.
- The reranker's scores saturate near 1.00 on Indonesian text, so the relevance shown per
  source says little; ranking relies on fusion and the gate on term coverage.
- The relevance gate is lexical: an off-topic question that shares words with the document
  ("harga saham") still reaches the LLM, which then has to refuse.
- Questions phrased with different words than the document ("wajib mematuhi" vs "berlaku
  untuk") depend on the vector model and can still miss.
- Documents indexed with other settings (older versions, chunk size, OCR off) are listed on
  the Documents page; upload the same file again to re-index it.
- The cost figure is an estimate from token counts and list prices, which change; check
  Groq's pricing.
- Passages are sent to Groq's hosted API. Do not upload confidential documents to the public
  demo.
- Single-instance design: embedded ChromaDB and a local log file. Scaling out needs a
  ChromaDB server and a shared log or metrics store.

## Tech stack

Python · Streamlit · FastAPI · ChromaDB · rank-bm25 · FastEmbed · FlashRank · PyMuPDF · RapidOCR · Groq ·
Docker · pytest · ruff