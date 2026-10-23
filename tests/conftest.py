import uuid

import pytest

from regushield.config import Settings
from regushield.engine import RAGEngine
from regushield.monitoring import QueryLog
from regushield.store import VectorStore
from tests.fakes import FakeChat, OverlapReranker, SynonymEmbedder, make_pdf

KODE_ETIK = [
    "Pasal 1. Pegawai dilarang menerima gratifikasi dalam bentuk apa pun dari nasabah atau "
    "vendor, termasuk pada hari raya keagamaan.",
    "Pasal 2. Sanksi pelanggaran kode etik terdiri atas sanksi ringan, sedang, dan berat.",
    "Pasal 3. Pegawai wajib melaporkan benturan kepentingan kepada atasan langsung.",
]
SANKSI_TABLE = [
    ["Jenis Sanksi", "Contoh"],
    ["Ringan", "Teguran lisan"],
    ["Berat", "Pemutusan hubungan kerja"],
]


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    for name in ("GROQ_API_KEY", "REGUSHIELD_API_KEY", "REGUSHIELD_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    return Settings(data_dir=tmp_path / "data", min_relevance=0.2, chunk_size=400, chunk_overlap=60)


@pytest.fixture
def kode_etik_pdf() -> bytes:
    return make_pdf(
        KODE_ETIK,
        header="Kode Etik PT Contoh 2025",
        table_on=2,
        table=SANKSI_TABLE,
        page_numbers=True,
    )


@pytest.fixture
def make_engine(settings):
    def build(chat=None, reranker="default", settings_override=None) -> RAGEngine:
        s = settings_override or settings
        store = VectorStore(None, SynonymEmbedder(), f"test-{uuid.uuid4().hex[:8]}")
        rr = OverlapReranker() if reranker == "default" else reranker
        return RAGEngine(s, store, rr, chat, QueryLog(s.log_path))

    return build


@pytest.fixture
def engine(make_engine, kode_etik_pdf) -> RAGEngine:
    eng = make_engine(chat=FakeChat("Pegawai dilarang menerima gratifikasi [1]."))
    eng.ingest("kode etik.pdf", kode_etik_pdf)
    return eng
