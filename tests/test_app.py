"""End-to-end tests of the Streamlit pages (AppTest) with a fake engine.

AppTest fails to re-run a page after ``st.feedback`` has rendered, so each test that needs a
second page starts a fresh AppTest instead of switching pages in the same one.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.fakes import make_pdf

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def use_engine(monkeypatch, engine):
    st.cache_resource.clear()
    monkeypatch.setattr("regushield.engine.RAGEngine.from_settings", lambda *a, **k: engine)
    yield
    st.cache_resource.clear()


def run(page: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    if page:
        app.switch_page(page)
    return app.run()


def texts(elements) -> list[str]:
    return [e.value for e in elements]


def test_assistant_answers_with_sources_and_metrics(engine):
    app = run()
    assert not app.exception
    app.chat_input[0].set_value("Bolehkah pegawai menerima hadiah dari nasabah?").run()

    assert not app.exception
    markdown = texts(app.markdown)
    assert "Pegawai dilarang menerima gratifikasi [1]." in markdown
    assert any("**[1]** kode etik.pdf · halaman 1" in m and "dikutip" in m for m in markdown)
    assert any(c.startswith("Total ") and "150 token" in c for c in texts(app.caption))
    assert engine.log.read()[0]["status"] == "answered"


def test_assistant_without_documents_points_to_the_documents_page(engine):
    engine.delete("kode etik.pdf")
    app = run()
    assert any("Belum ada dokumen" in i for i in texts(app.info))
    assert not app.chat_input


def test_assistant_warns_when_no_api_key(engine):
    engine.chat = None
    app = run()
    assert any("GROQ_API_KEY belum diatur" in w for w in texts(app.warning))


def test_scope_selector_lists_documents(engine):
    engine.ingest("lain.pdf", make_pdf(["Pasal 9. Cuti tahunan dua belas hari."]))
    app = run()
    assert app.selectbox[0].options == ["Semua dokumen", "kode etik.pdf", "lain.pdf"]


def test_documents_page_lists_indexed_files():
    app = run("views/documents.py")
    assert not app.exception
    table = app.dataframe[0].value
    assert list(table["Dokumen"]) == ["kode etik.pdf"]
    assert int(table["Tabel"][0]) == 1


def test_documents_page_delete_needs_confirmation(engine):
    app = run("views/documents.py")
    assert app.button(key="delete_button").disabled
    app.checkbox(key="delete_confirm").check().run()
    app.button(key="delete_button").click().run()
    assert engine.documents() == []
    assert any("dihapus" in s for s in texts(app.success))


def test_monitoring_empty_state():
    app = run("views/monitoring.py")
    assert any("Belum ada pertanyaan" in i for i in texts(app.info))


def test_monitoring_shows_kpis_from_the_log(engine):
    answer = engine.ask_sync("Bolehkah pegawai menerima hadiah dari nasabah?")
    engine.feedback(answer.id, positive=True)
    engine.ask_sync("Siapa presiden Amerika?")
    app = run("views/monitoring.py")
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Pertanyaan"] == "2"
    assert metrics["Tidak ditemukan"] == "50%"
    assert metrics["Feedback positif"] == "100%"
    assert len(app.dataframe) == 1


def test_example_question_button_asks_it(engine):
    app = run()
    app.button(key="example_1").click().run()
    assert not app.exception
    assert "Bagaimana aturan penerimaan hadiah atau gratifikasi?" in texts(app.markdown)
    assert (
        engine.log.read()[0]["question"] == "Bagaimana aturan penerimaan hadiah atau gratifikasi?"
    )


def test_continuation_source_is_labelled_instead_of_scored(engine):
    engine.ingest(
        "sanksi.pdf",
        make_pdf(
            [
                "Jenis sanksi pelanggaran kode etik diklasifikasikan ke dalam tiga kategori:",
                "Ringan, Sedang, dan Berat. Contoh pelanggaran berat adalah menerima suap.",
            ]
        ),
    )
    engine.retriever.top_k = 1  # so page 2 can only arrive as the continuation of page 1
    app = run()
    app.chat_input[0].set_value("Apa saja jenis sanksi pelanggaran kode etik?").run()
    assert not app.exception
    labels = [m for m in texts(app.markdown) if "sanksi.pdf · halaman 2" in m]
    assert labels and "lanjutan dari [" in labels[0] and "relevansi" not in labels[0]