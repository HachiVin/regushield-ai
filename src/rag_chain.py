import os
import sys
from typing import Dict, Any, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.retriever import HybridRerankRetriever

load_dotenv()

class FinancialRAGChain:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY tidak ditemukan di file .env!")
        
        print(f"Menginisialisasi LLM ({model_name})...")
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.0,
            groq_api_key=groq_api_key
        )

        self.retriever = HybridRerankRetriever()

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Anda adalah Asisten Pakar Regulasi, Kepatuhan, dan Finansial Korporat.
Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan konteks dokumen resmi yang disediakan di bawah ini.

Pedoman Penulisan & Format:
1. Jawab secara faktual, lugas, profesional, dan terstruktur.
2. Jika jawaban tidak tercantum di dalam teks konteks, nyatakan secara tegas: "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia." DILARANG berasumsi atau mengarang informasi.
3. Selalu sebutkan dokumen rujukan, nomor pasal/bab, dan nomor halaman yang tertera di metadata.
4. ATURAN TABEL & LIST:
   - DILARANG menggunakan tag HTML mentah seperti `<br>` atau `<br/>` di dalam tabel maupun teks.
   - Jika ingin membuat daftar poin larangan/kewajiban yang panjang, gunakan format list biasa (bullet points `-`) di luar tabel, ATAU pisahkan setiap poin menjadi baris (*row*) tersendiri di dalam tabel.

Konteks Dokumen:
{context}
"""),
            ("human", "{question}")
        ])

    def _format_context(self, docs: List[Document]) -> str:
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Dokumen")
            page = doc.metadata.get("page", "?")
            formatted_parts.append(
                f"[Kutipan {i}] (Sumber: {source} | Halaman {page})\n{doc.page_content}"
            )
        return "\n\n".join(formatted_parts)

    def answer_query(self, query: str, filter_source: Optional[str] = None) -> Dict[str, Any]:
        """Pencarian context hybrid + FlashRank rerank + inferensi LLM."""
        retrieved_docs = self.retriever.retrieve(query, top_n=3, filter_source=filter_source)
        
        if not retrieved_docs:
            return {
                "answer": "Informasi tidak ditemukan dalam dokumen yang tersedia.",
                "sources": []
            }
        
        context_text = self._format_context(retrieved_docs)
        
        prompt = self.prompt_template.format_messages(
            context=context_text,
            question=query
        )
        response = self.llm.invoke(prompt)
        
        sources = [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "relevance_score": float(doc.metadata.get("rerank_score", 0.0)),
                "snippet": doc.page_content
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": response.content,
            "sources": sources
        }