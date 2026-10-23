import pytest

from regushield.chunking import chunk_pages, split_table, split_text
from regushield.pdf import PageContent, PDFError, read_pdf
from tests.conftest import KODE_ETIK, SANKSI_TABLE
from tests.fakes import make_pdf


# --- PDF reading ----------------------------------------------------------------------------
def test_pages_are_numbered_from_one(kode_etik_pdf):
    pages = read_pdf(kode_etik_pdf)
    assert [p.number for p in pages] == [1, 2, 3]
    assert pages[0].text.startswith("Pasal 1.")
    assert pages[2].text.startswith("Pasal 3.")


def test_table_is_extracted_as_markdown_and_not_duplicated_in_text(kode_etik_pdf):
    page = read_pdf(kode_etik_pdf)[1]
    assert len(page.tables) == 1
    table = page.tables[0]
    assert table.startswith("|") and "|---|---|" in table
    assert "Pemutusan hubungan kerja" in table
    assert "Pemutusan" not in page.text


def test_running_header_and_page_numbers_are_removed(kode_etik_pdf):
    for page in read_pdf(kode_etik_pdf):
        assert "Kode Etik PT Contoh" not in page.text
        assert not page.text.rstrip().endswith(str(page.number))


def test_header_is_kept_when_the_document_is_too_short_to_tell():
    pages = read_pdf(make_pdf(KODE_ETIK[:2], header="Kode Etik PT Contoh 2025"))
    assert "Kode Etik PT Contoh" in pages[0].text


def test_layout_line_breaks_inside_a_paragraph_are_joined():
    long = "Pegawai wajib menjaga kerahasiaan data nasabah " * 6
    text = read_pdf(make_pdf([long.strip()]))[0].text
    assert "\n" not in text
    assert "  " not in text


@pytest.mark.parametrize("data", [b"", b"bukan pdf", b"%PDF-1.4 rusak"])
def test_invalid_pdf_is_rejected(data):
    with pytest.raises(PDFError, match="bukan PDF"):
        read_pdf(data)


def test_pdf_without_text_is_rejected():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().draw_rect(pymupdf.Rect(50, 50, 200, 200))
    with pytest.raises(PDFError, match="scan"):
        read_pdf(doc.tobytes())


# --- chunking -------------------------------------------------------------------------------
def test_short_text_is_one_chunk():
    assert split_text("Pasal 1. Singkat.", 100, 20) == ["Pasal 1. Singkat."]


def test_long_text_is_split_within_the_size_and_overlaps():
    sentences = [f"Kalimat nomor {i} membahas kewajiban pegawai." for i in range(40)]
    pieces = split_text(" ".join(sentences), 300, 60)
    assert len(pieces) > 3
    assert all(len(p) <= 300 + 60 + 1 for p in pieces)
    for previous, current in zip(pieces, pieces[1:], strict=False):
        assert current.split()[0] in previous  # starts with the tail of the previous piece


def test_sentences_keep_their_punctuation():
    pieces = split_text("Satu dua tiga. Empat lima enam. Tujuh delapan sembilan.", 20, 0)
    assert pieces[0].endswith(".")


def test_a_single_huge_word_is_hard_split():
    assert all(len(p) <= 50 for p in split_text("x" * 175, 50, 0))


def test_empty_text_gives_no_chunks():
    assert split_text("   ", 100, 10) == []


def test_chunks_never_cross_pages_and_carry_page_numbers():
    pages = [
        PageContent(1, "A " * 300),
        PageContent(2, "B " * 10, tables=["|a|b|\n|---|---|\n|1|2|"]),
    ]
    chunks = chunk_pages(pages, "doc.pdf", 200, 20)
    assert {c.page for c in chunks} == {1, 2}
    assert all(set(c.text.split()) <= {"A"} for c in chunks if c.page == 1)
    assert [c.kind for c in chunks if c.page == 2] == ["text", "table"]


def test_chunks_record_their_reading_order():
    pages = [
        PageContent(1, "A " * 300, tables=["|a|b|\n|---|---|\n|1|2|"]),
        PageContent(2, "B " * 10),
    ]
    chunks = chunk_pages(pages, "doc.pdf", 200, 20)
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert [c.kind for c in chunks][-2:] == ["table", "text"]


def test_chunk_ids_are_stable_and_unique():
    pages = [PageContent(1, "Pasal 1. Isi."), PageContent(2, "Pasal 1. Isi.")]
    first = chunk_pages(pages, "doc.pdf", 100, 0)
    assert [c.id for c in first] == [c.id for c in chunk_pages(pages, "doc.pdf", 100, 0)]
    assert len({c.id for c in first}) == 2


def test_long_tables_are_split_by_rows_repeating_the_header():
    rows = "\n".join(f"|Sanksi {i}|Keterangan panjang nomor {i}|" for i in range(60))
    table = "|Jenis|Keterangan|\n|---|---|\n" + rows
    parts = split_table(table, 300)
    assert len(parts) > 1
    assert all(p.startswith("|Jenis|Keterangan|\n|---|---|") for p in parts)
    assert sum(p.count("|Sanksi ") for p in parts) == 60


def test_table_rows_from_the_fixture_stay_together():
    table = "\n".join("|" + "|".join(row) + "|" for row in SANKSI_TABLE)
    assert split_table(table, 400) == [table]