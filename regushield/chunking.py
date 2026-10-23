"""Split pages into retrieval chunks.

Chunks never cross a page boundary, so every chunk cites exactly one page. Tables become their
own chunks (split by rows, repeating the header, when they are long). Each chunk also records its
position in the document (``seq``), so retrieval can add the chunk that follows a hit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from regushield.pdf import PageContent

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str  # file name
    page: int  # 1-based
    kind: str = "text"  # "text" or "table"
    seq: int = -1  # reading order within the document; -1 = unknown (indexed before v2.1)


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursively split on the coarsest separator that yields pieces no longer than ``size``."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, finer = separators[0], separators[1:]
    if sep not in text:
        return _split(text, size, finer)

    pieces = text.split(sep)
    if sep.strip():  # keep punctuation (". ", "; ", ", ") with the piece it ends
        pieces = [p + sep.rstrip() for p in pieces[:-1]] + [pieces[-1]]
        joiner = " "
    else:
        joiner = sep

    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        if len(piece) <= size:
            current = piece
        else:
            sub = _split(piece, size, finer)
            parts.extend(sub[:-1])
            current = sub[-1]
    if current:
        parts.append(current)
    return parts


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into pieces of at most ~``size`` characters, each starting with the last
    ``overlap`` characters of the previous piece (cut at a word boundary)."""
    text = text.strip()
    if not text:
        return []
    pieces = _split(text, size, _SEPARATORS)
    if overlap <= 0 or len(pieces) == 1:
        return [p.strip() for p in pieces if p.strip()]
    out = [pieces[0].strip()]
    for previous, piece in zip(pieces, pieces[1:], strict=False):
        tail = previous[-overlap:]
        tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
        out.append(f"{tail} {piece}".strip())
    return out


def split_table(markdown: str, size: int) -> list[str]:
    lines = markdown.strip().splitlines()
    if len(markdown) <= size * 2 or len(lines) <= 3:
        return [markdown.strip()]
    header, rows = lines[:2], lines[2:]
    parts, current = [], []
    for row in rows:
        if current and len("\n".join(header + current + [row])) > size * 2:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(header + current))
    return parts


def _chunk_id(source: str, page: int, kind: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{kind}|{index}|{text}".encode()).hexdigest()
    return digest[:20]


def chunk_pages(pages: list[PageContent], source: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        pieces = [("text", t) for t in split_text(page.text, size, overlap)]
        for table in page.tables:
            pieces += [("table", t) for t in split_table(table, size)]
        for index, (kind, text) in enumerate(pieces):
            chunks.append(
                Chunk(
                    _chunk_id(source, page.number, kind, index, text),
                    text,
                    source,
                    page.number,
                    kind,
                    seq=len(chunks),
                )
            )
    return chunks