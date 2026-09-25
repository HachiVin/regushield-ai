# ReguShield - Regulatory Intelligence Platform
ReguShield adalah aplikasi *Retrieval-Augmented Generation* (RAG) berbasis kecerdasan buatan untuk menganalisis dokumen regulasi, kepatuhan hukum, dan SOP perusahaan tanpa halusinasi.

## Fitur Utama
* **Anti-Halusinasi (High Faithfulness):** Sistem akan menjawab "Informasi tidak ditemukan" jika fakta tidak ada di dokumen.
* **Hybrid Rerank Search:** Menggabungkan pencarian kata kunci presisi (BM25 Regex) dan pencarian makna (ChromaDB + MiniLM).
* **Audit Trail Transparan:** Setiap jawaban AI dilengkapi dengan kutipan pasal aslinya (sitasi) beserta nomor halamannya.
* **Complex Table Parsing:** Mampu membaca tabel sanksi/hukuman kompleks menggunakan PyMuPDF.

## Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **LLM Engine:** Groq (LLaMA-3 / GPT-OSS 120b)
* **Vector Store:** ChromaDB

## Live Demo
Aplikasi ini dapat diakses secara langsung melalui tautan berikut: 