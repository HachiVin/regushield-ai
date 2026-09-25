import os
import sys
from typing import List, Optional

# Tambahkan root directory ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.rag_chain import FinancialRAGChain

app = FastAPI(
    title="Financial & Regulatory RAG API",
    description="REST API untuk analisis dokumen regulasi dan finansial berbasis Hybrid RAG.",
    version="1.0.0"
)

# Inisialisasi model saat startup
rag_chain = None

@app.on_event("startup")
def startup_event():
    global rag_chain
    rag_chain = FinancialRAGChain(model_name="openai/gpt-oss-20b")

# Skema Request & Response dengan Pydantic
class QueryRequest(BaseModel):
    query: str = Field(..., example="Kapan batas akhir penyampaian laporan bulanan?")

class SourceCitation(BaseModel):
    source: Optional[str]
    page: Optional[int]
    relevance_score: Optional[float]
    snippet: Optional[str]

class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    sources: List[SourceCitation]

@app.get("/api/v1/health")
def health_check():
    """Endpoint untuk mengecek status kesehatan layanan."""
    return {"status": "healthy", "service": "Financial RAG Engine"}

@app.post("/api/v1/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """Endpoint inferensi untuk mengajukan pertanyaan regulasi."""
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Engine RAG belum siap.")
    
    try:
        result = rag_chain.answer_query(request.query)
        return QueryResponse(
            status="success",
            query=request.query,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)