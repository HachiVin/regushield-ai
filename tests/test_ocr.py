import pytest

from regushield import ocr
from regushield.chunking import chunk_pages
from regushield.ocr import OCR_MARKER, OcrLine, lines_to_text
from regushield.pdf import PDFError, read_pdf
from tests.fakes import FakeOcr, make_image_pdf, text_png

TABLE = [["Jenis Sanksi", "2024"], ["Ringan", "291"], ["Total", "1.043"]]


def line(text, x0, y0, x1=None, height=20):
    return OcrLine(text, x0, y0, x1 if x1 is not None else x0 + 10 * len(text), y0 + height, 0.99)


def test_lines_on_one_row_are_joined_and_far_apart_cells_get_a_separator():
    lines = [
        line("Ringan", 10, 52),
        line("Jenis Sanksi", 10, 10),
        line("2024", 400, 12),
        line("291", 400, 50),
        line("Teguran", 100, 52, 170),
        line("Tertulis", 176, 51),
    ]
    assert lines_to_text(lines) == "Jenis Sanksi | 2024\nRingan Teguran Tertulis | 291"


def test_no_lines_give_no_text():
    assert lines_to_text([]) == ""


def test_image_text_keeps_its_place_between_the_paragraphs_around_it():
    pdf = make_image_pdf(
        [
            [
                ("text", "Jenis Sanksi Pelanggaran Kode Etik"),
                ("image", text_png("tabel")),
                ("text", "Penutup halaman."),
            ]
        ]
    )
    page = read_pdf(pdf, FakeOcr(TABLE))[0]
    assert [s.kind for s in page.segments] == ["text", "ocr", "text"]
    assert page.segments[1].text == "Jenis Sanksi | 2024\nRingan | 291\nTotal | 1.043"
    assert page.segments[1].context == "Jenis Sanksi Pelanggaran Kode Etik"
    assert page.ocr_images == 1
    assert "Ringan" not in page.text  # the text layer stays separate


def test_ocr_chunks_say_where_they_come_from_and_which_heading_they_belong_to():
    pdf = make_image_pdf(
        [[("text", "Jenis Sanksi Pelanggaran Kode Etik"), ("image", text_png("tabel"))]]
    )
    chunks = chunk_pages(read_pdf(pdf, FakeOcr(TABLE)), "k.pdf", 400, 60)
    assert [c.kind for c in chunks] == ["text", "ocr"]
    assert chunks[1].text.startswith(f"{OCR_MARKER} · Jenis Sanksi Pelanggaran Kode Etik\n")
    assert "Total | 1.043" in chunks[1].text
    assert [c.seq for c in chunks] == [0, 1]


def test_small_images_and_images_repeated_on_most_pages_are_not_read():
    logo = text_png("LOGO", 200, 80)
    banner = text_png("KOP SURAT")
    texts = ["Pasal satu tentang cuti.", "Pasal dua tentang gaji.", "Pasal tiga tentang sanksi."]
    pages = [[("logo", logo), ("image", banner), ("text", text)] for text in texts]
    fake = FakeOcr(TABLE)
    result = read_pdf(make_image_pdf(pages), fake)
    assert fake.calls == 0
    assert [p.ocr_images for p in result] == [0, 0, 0]


def test_ocr_stops_after_the_image_budget():
    pages = [[("image", text_png(f"tabel {n}")), ("text", "Teks.")] for n in range(3)]
    fake = FakeOcr(TABLE)
    result = read_pdf(make_image_pdf(pages[:2]), fake, max_ocr_images=1)
    assert fake.calls == 1 and [p.ocr_images for p in result] == [1, 0]


def test_a_scanned_pdf_is_readable_with_ocr_and_explained_without():
    scanned = make_image_pdf([[("image", text_png("Pasal 1"))]])
    assert read_pdf(scanned, FakeOcr([["Pasal 1. Pegawai wajib jujur."]]))[0].ocr_images == 1
    with pytest.raises(PDFError, match="Aktifkan OCR"):
        read_pdf(scanned)


def test_without_ocr_images_are_ignored_as_before():
    pdf = make_image_pdf([[("text", "Judul."), ("image", text_png("tabel"))]])
    page = read_pdf(pdf)[0]
    assert [s.kind for s in page.segments] == ["text"] and page.ocr_images == 0


@pytest.mark.skipif(not ocr.available(), reason="rapidocr-onnxruntime is not installed")
def test_real_ocr_reads_a_table_pasted_as_an_image():
    picture = text_png("Sanksi Ringan: Teguran Tertulis Pertama\nTotal pelanggaran 1.043", 520)
    pdf = make_image_pdf([[("text", "Jenis Sanksi Pelanggaran Kode Etik"), ("image", picture)]])
    text = read_pdf(pdf, ocr.RapidOcrEngine())[0].segments[1].text
    assert "Teguran" in text and "1.043" in text


@pytest.mark.skipif(not ocr.available(), reason="rapidocr-onnxruntime is not installed")
def test_transparent_pixels_become_white_not_black():
    import numpy as np

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)  # fully transparent black
    assert (ocr._to_bgr_on_white(rgba, np) == 255).all()
    grey = np.full((2, 2), 7, dtype=np.uint8)
    assert ocr._to_bgr_on_white(grey, np).shape == (2, 2, 3)


def test_the_reason_ocr_is_unavailable_is_reported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("libgthread-2.0.so.0: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert "libgthread-2.0.so.0" in ocr.unavailable_reason()
    assert not ocr.available()