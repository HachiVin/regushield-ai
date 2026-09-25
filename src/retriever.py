import os
import re
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

def get_embeddings():
    """Menggunakan MiniLM embedding lokal tanpa API key."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def tokenize_indo(text: str) -> List[str]:
    """Membersihkan tanda baca (koma, tanda tanya, titik) agar pencocokan BM25 100% akurat."""
    return re.findall(r'\w+', text.lower())

class HybridRerankRetriever:
    def __init__(self, vector_db_path: str = os.path.join("data", "vector_db")):
        self.embeddings = get_embeddings()
        self.vector_db_path = vector_db_path
        self.load_index()

    def load_index(self):
        """Memuat Vector DB dan menginisialisasi BM25 corpus dengan Regex."""
        self.db = Chroma(
            persist_directory=self.vector_db_path, 
            embedding_function=self.embeddings
        )
        
        raw_data = self.db.get()
        self.all_docs: List[Document] = []
        
        if raw_data and raw_data.get("documents"):
            for i, text in enumerate(raw_data["documents"]):
                metadata = raw_data["metadatas"][i] if raw_data.get("metadatas") else {}
                self.all_docs.append(Document(page_content=text, metadata=metadata))
        
        if self.all_docs:
            print(f"Menyiapkan index BM25 ({len(self.all_docs)} potongan teks dengan Regex Tokenizer)...")
            tokenized_corpus = [tokenize_indo(doc.page_content) for doc in self.all_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve(self, query: str, top_n: int = 5, filter_source: Optional[str] = None) -> List[Document]:
        """Pencarian Hybrid (BM25 + Chroma) dengan ketahanan terhadap tanda baca."""
        if not self.all_docs or self.bm25 is None:
            return []

        chroma_filter = {"source": filter_source} if filter_source and filter_source != "Semua Dokumen" else None

        # 1. Sparse Keyword Search (BM25) - dengan Tokenizer Regex
        tokenized_query = tokenize_indo(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        
        sparse_results = []
        for idx in top_bm25_indices:
            doc = self.all_docs[idx]
            if filter_source and filter_source != "Semua Dokumen":
                if doc.metadata.get("source") != filter_source:
                    continue
            sparse_results.append(doc)

        # 2. Dense Semantic Search (ChromaDB)
        dense_results = self.db.similarity_search(query, k=10, filter=chroma_filter)

        # 3. Gabungkan hasil dan eliminasi duplikasi
        unique_chunks = {}
        for doc in sparse_results + dense_results:
            if doc.page_content not in unique_chunks:
                doc.metadata["rerank_score"] = 1.0 
                unique_chunks[doc.page_content] = doc

        final_docs = list(unique_chunks.values())[:top_n]
        return final_docs