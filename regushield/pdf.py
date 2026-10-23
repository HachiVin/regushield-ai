"""PDF -> per-page text and tables (PyMuPDF), with light cleaning of real-world PDFs.

* Page numbers are 1-based, as printed on the page, so citations point to the right page.
* Tables are detected with ``find_tables()`` and kept as Markdown, so rows and columns of a
  sanctions table stay together instead of being flattened into a word soup.
* Running headers/footers that repeat on most pages, and bare page numbers, are removed so
  they do not pollute keyword search.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf


class PDFError(Exception):
    """The file cannot be read as a PDF, is encrypted, or has no extractable text."""


@dataclass
class PageContent:
    number: int  # 1-based
    text: str
    tables: list[str] = field(default_factory=list)  # Markdown


_PAGE_NUMBER = re.compile(
    r"^\s*(?:hal(?:aman)?\.?\s*)?-?\s*\d{1,4}\s*-?\s*(?:/\s*\d{1,4})?\s*$", re.I
)


def _normalize_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _clean_block(text: str) -> str:
    """Join lines broken by the PDF layout inside a paragraph and collapse whitespace."""
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"[ \t]+", " ", " ".join(line for line in lines if line)).strip()


def _page_blocks(page: pymupdf.Page) -> tuple[list[str], list[str]]:
    tables, table_rects = [], []
    try:
        for table in page.find_tables().tables:
            if table.row_count >= 2 and table.col_count >= 2:
                tables.append(table.to_markdown().strip())
                table_rects.append(pymupdf.Rect(table.bbox))
    except Exception:  # table detection is best-effort; text extraction still works
        tables, table_rects = [], []

    blocks = []
    for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:  # images
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if any(rect.intersects(t) for t in table_rects):
            continue  # already captured as a table
        cleaned = _clean_block(text)
        if cleaned:
            blocks.append(cleaned)
    return blocks, tables


def _repeated_lines(pages_blocks: list[list[str]]) -> set[str]:
    """Lines (digits ignored) that appear at the top/bottom of at least 60% of pages."""
    if len(pages_blocks) < 3:
        return set()
    counts: Counter[str] = Counter()
    for blocks in pages_blocks:
        edges = {_normalize_line(b) for b in blocks[:2] + blocks[-2:]}
        counts.update(edges)
    threshold = 0.6 * len(pages_blocks)
    return {line for line, n in counts.items() if n >= threshold and len(line) <= 120}


def read_pdf(data: bytes) -> list[PageContent]:
    """Extract every page. Raises ``PDFError`` for unreadable or text-less files."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFError("File bukan PDF yang valid atau rusak.") from exc
    with doc:
        if doc.needs_pass:
            raise PDFError("PDF terkunci dengan password.")
        extracted = [_page_blocks(page) for page in doc]

    repeated = _repeated_lines([blocks for blocks, _ in extracted])
    pages = []
    for number, (blocks, tables) in enumerate(extracted, start=1):
        kept = [
            b for b in blocks if _normalize_line(b) not in repeated and not _PAGE_NUMBER.match(b)
        ]
        pages.append(PageContent(number=number, text="\n\n".join(kept), tables=tables))

    if not any(p.text or p.tables for p in pages):
        raise PDFError("Tidak ada teks yang bisa dibaca (kemungkinan PDF hasil scan/gambar).")
    return pages
