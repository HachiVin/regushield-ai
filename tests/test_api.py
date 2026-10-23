import dataclasses

import pytest
from fastapi.testclient import TestClient

from regushield.api import create_app
from tests.fakes import FakeChat, make_pdf


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine)) as c:
        yield c


def test_health_is_public_and_reports_the_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 1
    assert body["chunks"] > 0 and body["llm_configured"] is True


def test_query_returns_answer_sources_and_metrics(client):
    response = client.post(
        "/query", json={"question": "Bolehkah pegawai menerima hadiah dari nasabah?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered" and body["answer"].endswith("[1].")
    assert body["sources"][0]["page"] == 1 and body["sources"][0]["cited"] is True
    assert body["latency_ms"]["total"] > 0 and body["prompt_tokens"] == 120


def test_query_validation_and_unknown_document(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    response = client.post("/query", json={"question": "sanksi", "document": "tidak-ada.pdf"})
    assert response.status_code == 404


def test_documents_upload_list_and_delete(client, kode_etik_pdf):
    files = {"file": ("baru.pdf", b"", "application/pdf")}
    assert client.post("/documents", files=files).status_code == 422  # empty file
    other = make_pdf(["Pasal 7. Pegawai wajib mengikuti pelatihan anti pencucian uang."])
    response = client.post("/documents", files={"file": ("baru.pdf", other, "application/pdf")})
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "indexed"
    assert {d["source"] for d in client.get("/documents").json()} == {"kode etik.pdf", "baru.pdf"}
    assert client.delete("/documents/baru.pdf").status_code == 204
    assert client.delete("/documents/baru.pdf").status_code == 404


def test_only_pdf_files_are_accepted(client):
    response = client.post("/documents", files={"file": ("notes.txt", b"teks", "text/plain")})
    assert response.status_code == 415


def test_metrics_summarize_the_log(client):
    client.post("/query", json={"question": "Bolehkah pegawai menerima hadiah dari nasabah?"})
    client.post("/query", json={"question": "Siapa presiden Amerika?"})
    body = client.get("/metrics").json()
    assert body["queries"] == 2 and body["not_found_rate"] == 0.5


def test_api_key_is_required_when_configured(make_engine, kode_etik_pdf, settings):
    eng = make_engine(
        chat=FakeChat(), settings_override=dataclasses.replace(settings, api_key="rahasia")
    )
    eng.ingest("kode etik.pdf", kode_etik_pdf)
    with TestClient(create_app(eng)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/documents").status_code == 401
        assert c.get("/documents", headers={"X-API-Key": "salah"}).status_code == 401
        assert c.get("/documents", headers={"X-API-Key": "rahasia"}).status_code == 200
