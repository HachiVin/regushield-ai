"""REST API for the same engine the UI uses. Run: ``uvicorn regushield.api:app --port 8000``.

Set ``REGUSHIELD_API_KEY`` to require an ``X-API-Key`` header on every endpoint except /health.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from regushield.config import load_settings
from regushield.engine import IngestError, RAGEngine
from regushield.monitoring import summarize


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000, examples=["Apa sanksi berat bagi pegawai?"]
    )
    document: str | None = Field(None, description="Limit the search to one indexed PDF.")


class SourceOut(BaseModel):
    number: int
    source: str
    page: int
    kind: str
    relevance: float | None
    cited: bool
    text: str


class QueryResponse(BaseModel):
    id: str
    status: str
    answer: str
    sources: list[SourceOut]
    latency_ms: dict[str, float]
    ttft_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    model: str | None


def create_app(engine: RAGEngine | None = None) -> FastAPI:
    """``engine`` is injected by tests; otherwise it is built from the environment at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            load_dotenv()
            app.state.engine = RAGEngine.from_settings(load_settings())
        else:
            app.state.engine = engine
        yield

    app = FastAPI(
        title="ReguShield API",
        version="2.0.0",
        description="Grounded, cited answers over regulatory and compliance PDFs.",
        lifespan=lifespan,
    )

    def get_engine(request: Request) -> RAGEngine:
        return request.app.state.engine

    def require_key(
        request: Request, x_api_key: Annotated[str | None, Header()] = None
    ) -> RAGEngine:
        eng = get_engine(request)
        expected = eng.settings.api_key
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
        return eng

    Engine = Annotated[RAGEngine, Depends(require_key)]

    @app.get("/health")
    def health(request: Request) -> dict:
        eng = get_engine(request)
        return {
            "status": "ok",
            "documents": len(eng.documents()),
            "chunks": eng.store.count(),
            "llm_configured": eng.chat is not None,
        }

    @app.get("/documents")
    def list_documents(eng: Engine) -> list[dict]:
        return [doc.__dict__ for doc in eng.documents()]

    @app.post("/documents", status_code=201)
    async def upload_document(eng: Engine, file: Annotated[UploadFile, File()]) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only PDF files are accepted.")
        try:
            report = eng.ingest(file.filename or "document.pdf", await file.read())
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return report.__dict__

    @app.delete("/documents/{name}", status_code=204)
    def delete_document(name: str, eng: Engine) -> None:
        if not eng.delete(name):
            raise HTTPException(status_code=404, detail="Document not found.")

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest, eng: Engine) -> QueryResponse:
        if body.document and body.document not in {d.source for d in eng.documents()}:
            raise HTTPException(status_code=404, detail="Document not found.")
        answer = eng.ask_sync(body.question, body.document)
        return QueryResponse(
            id=answer.id,
            status=answer.status,
            answer=answer.text,
            sources=[SourceOut(**s.__dict__) for s in answer.sources],
            latency_ms=answer.latency_ms,
            ttft_ms=answer.ttft_ms,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            cost_usd=answer.cost_usd,
            model=answer.model,
        )

    @app.get("/metrics")
    def metrics(eng: Engine) -> dict:
        return summarize(eng.log.read())

    return app


app = create_app()
