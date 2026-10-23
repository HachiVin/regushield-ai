"""PDF -> per-page text, tables and OCR text (PyMuPDF), with light cleaning of real-world PDFs.

* Page numbers are 1-based, as printed on the page, so citations point to the right page.
* Tables are detected with ``find_tables()`` and kept as Markdown, so rows and columns of a
  sanctions table stay together instead of being flattened into a word soup.
* Running headers/footers that repeat on most pages, and bare page numbers, are removed so
  they do not pollute keyword search.
* Images large enough to hold text (a table pasted as a picture, a scanned page) are read with
  OCR when an OCR engine is given. Their text keeps its place in the reading order of the page.
  Images repeated on most pages (logos, letterheads) are skipped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from regushield.ocr import OcrEngine, lines_to_text


class PDFError(Exception):
    """The file cannot be read as a PDF, is encrypted, or has no extractable text."""


@dataclass(frozen=True)
class Segment:
    kind: str  # "text" or "ocr"
    text: str
    context: str = ""  # OCR only: the text line just before the image, usually its heading


@dataclass
class PageContent:
    number: int  # 1-based
    text: str  # the page's own text layer (without OCR)
    tables: list[str] = field(default_factory=list)  # Markdown
    segments: list[Segment] = field(default_factory=list)  # text and OCR in reading order
    ocr_images: int = 0

    def reading_order(self) -> list[Segment]:
        return self.segments or ([Segment("text", self.text)] if self.text else [])


@dataclass
class _Image:
    y0: float
    rect: pymupdf.Rect
    zoom: float
    digest: bytes
    xref: int = 0  # 0 = inline image, rendered from the page instead


_PAGE_NUMBER = re.compile(
    r"^\s*(?:hal(?:aman)?\.?\s*)?-?\s*\d{1,4}\s*-?\s*(?:/\s*\d{1,4})?\s*$", re.I
)
# An image smaller than this (fraction of the page) is a logo, icon or bullet, not text.
MIN_IMAGE_WIDTH = 0.25
MIN_IMAGE_AREA = 0.02
MIN_IMAGE_HEIGHT_PT = 30.0


def _normalize_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _clean_block(text: str) -> str:
    """Join lines broken by the PDF layout inside a paragraph and collapse whitespace."""
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"[ \t]+", " ", " ".join(line for line in lines if line)).strip()


def _page_blocks(page: pymupdf.Page) -> tuple[list[tuple[float, str]], list[str]]:
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
            blocks.append((y0, cleaned))
    return blocks, tables


def _page_images(page: pymupdf.Page) -> list[_Image]:
    """Images on the page big enough to contain readable text."""
    page_rect = page.rect
    images = []
    try:
        infos = page.get_image_info(hashes=True, xrefs=True)
    except Exception:
        return []
    for info in infos:
        rect = pymupdf.Rect(info["bbox"]) & page_rect
        if rect.is_empty:
            continue
        big_enough = (
            rect.width >= MIN_IMAGE_WIDTH * page_rect.width
            and rect.height >= MIN_IMAGE_HEIGHT_PT
            and rect.get_area() >= MIN_IMAGE_AREA * page_rect.get_area()
        )
        if not big_enough:
            continue
        # Render at the image's own resolution (no blurry resampling); the OCR engine then
        # upscales it with a sharp filter.
        native_width = float(info.get("width") or rect.width)
        zoom = min(4.0, max(1.0, native_width / rect.width))
        images.append(
            _Image(rect.y0, rect, zoom, info.get("digest") or b"", int(info.get("xref") or 0))
        )
    return images


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


def _repeated_images(pages_images: list[list[_Image]]) -> set[bytes]:
    """Images (by content hash) shown on at least 60% of pages: logos and letterheads."""
    if len(pages_images) < 3:
        return set()
    counts: Counter[bytes] = Counter()
    for images in pages_images:
        counts.update({image.digest for image in images if image.digest})
    return {digest for digest, n in counts.items() if n >= 0.6 * len(pages_images)}


def _image_png(page: pymupdf.Page, image: _Image) -> bytes:
    """The image's own pixels when it is stored as an object (sharpest for OCR), otherwise the
    page area rendered at the image's resolution. Transparency is kept for the OCR engine."""
    if image.xref:
        try:
            doc = page.parent
            pixmap = pymupdf.Pixmap(doc, image.xref)
            if pixmap.colorspace and pixmap.colorspace.n not in (1, 3):  # CMYK and the like
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
            smask = doc.extract_image(image.xref).get("smask") or 0
            if smask and not pixmap.alpha:
                pixmap = pymupdf.Pixmap(pixmap, pymupdf.Pixmap(doc, smask))
            return pixmap.tobytes("png")
        except Exception:
            pass  # unusual encodings: fall back to rendering the page area
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(image.zoom, image.zoom), clip=image.rect)
    return pixmap.tobytes("png")


def _ocr_image(page: pymupdf.Page, image: _Image, ocr: OcrEngine) -> str:
    return lines_to_text(ocr.read(_image_png(page, image)))


def _last_sentence(text: str, limit: int = 100) -> str:
    """The last sentence (or heading) of a text block, at most ``limit`` characters."""
    sentence = re.split(r"(?<=[.!?])\s+", text.strip())[-1]
    if len(sentence) > limit:
        sentence = sentence[-limit:].split(" ", 1)[-1]
    return sentence


def _segments(items: list[tuple[float, int, str, str]]) -> list[Segment]:
    """Merge text blocks and OCR text by vertical position into reading-order segments."""
    segments: list[Segment] = []
    texts: list[str] = []
    for _y, _order, kind, text in sorted(items, key=lambda item: (item[0], item[1])):
        if kind == "text":
            texts.append(text)
            continue
        context = ""
        if texts:
            segments.append(Segment("text", "\n\n".join(texts)))
            context = _last_sentence(texts[-1])
            texts = []
        segments.append(Segment("ocr", text, context))
    if texts:
        segments.append(Segment("text", "\n\n".join(texts)))
    return segments


def read_pdf(
    data: bytes, ocr: OcrEngine | None = None, max_ocr_images: int = 50
) -> list[PageContent]:
    """Extract every page. Raises ``PDFError`` for unreadable or text-less files.

    With ``ocr``, up to ``max_ocr_images`` images per document are read as text.
    """
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFError("File bukan PDF yang valid atau rusak.") from exc
    with doc:
        if doc.needs_pass:
            raise PDFError("PDF terkunci dengan password.")
        extracted = [_page_blocks(page) for page in doc]
        images = [_page_images(page) if ocr else [] for page in doc]
        repeated = _repeated_lines([[text for _y, text in blocks] for blocks, _ in extracted])
        logos = _repeated_images(images)

        pages = []
        budget = max_ocr_images
        for index, ((blocks, tables), page_images) in enumerate(
            zip(extracted, images, strict=True)
        ):
            items = [
                (y, 1, "text", text)
                for y, text in blocks
                if _normalize_line(text) not in repeated and not _PAGE_NUMBER.match(text)
            ]
            read = 0
            for image in page_images:
                if image.digest in logos or budget <= 0:
                    continue
                budget -= 1
                text = _ocr_image(doc[index], image, ocr)  # type: ignore[arg-type]
                if text.strip():
                    items.append((image.y0, 0, "ocr", text))
                    read += 1
            segments = _segments(items)
            page_text = "\n\n".join(s.text for s in segments if s.kind == "text")
            pages.append(PageContent(index + 1, page_text, tables, segments, read))

    if not any(p.text or p.tables or p.ocr_images for p in pages):
        hint = "" if ocr else " Aktifkan OCR untuk membaca PDF hasil scan."
        raise PDFError(
            f"Tidak ada teks yang bisa dibaca (kemungkinan PDF hasil scan/gambar).{hint}"
        )
    return pages