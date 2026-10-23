import json

import pytest

from regushield.__main__ import main
from tests.fakes import make_pdf


@pytest.fixture(autouse=True)
def use_engine(monkeypatch, engine):
    monkeypatch.setattr("regushield.engine.RAGEngine.from_settings", lambda *a, **k: engine)


def test_ingest_docs_ask_and_metrics(tmp_path, capsys):
    pdf = tmp_path / "sop.pdf"
    pdf.write_bytes(make_pdf(["Laporan bulanan disampaikan paling lambat tanggal 10."]))
    assert main(["ingest", str(pdf), str(tmp_path / "missing.pdf")]) == 1
    out = capsys.readouterr()
    assert "indexed   sop.pdf: 1 pages" in out.out and "failed    missing.pdf" in out.err

    assert main(["docs"]) == 0
    assert "sop.pdf: 1 pages" in capsys.readouterr().out

    assert main(["ask", "Bolehkah pegawai menerima hadiah dari nasabah?"]) == 0
    out = capsys.readouterr().out
    assert "gratifikasi [1]." in out and "*[1] kode etik.pdf p.1" in out

    assert main(["metrics"]) == 0
    assert json.loads(capsys.readouterr().out)["queries"] == 1
