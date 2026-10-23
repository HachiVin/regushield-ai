"""OCR for images inside PDFs (tables or text pasted as pictures, scanned pages).

RapidOCR runs PP-OCR models through ONNX Runtime on CPU, and its models ship inside the wheel,
so there is nothing to download and nothing to pay for. Its default recognizer is trained on
Chinese and English; Indonesian uses the same Latin letters, but words are sometimes merged
("kepadanasabah"), which is why OCR text is marked as such wherever it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Protocol

OCR_MARKER = "[Teks dari gambar (OCR)]"
UPSCALE_BELOW_PX = 1600


@dataclass(frozen=True)
class OcrLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


class OcrEngine(Protocol):
    def read(self, png: bytes) -> list[OcrLine]:
        """Text lines found in a PNG image, with their boxes in image pixels."""
        ...


class RapidOcrEngine:
    """RapidOCR (ONNX). The angle classifier is off: rendered PDF images are never rotated,
    and with it on, some lines came back upside down as gibberish."""

    def __init__(self, min_score: float = 0.6) -> None:
        self.min_score = min_score
        self._engine = None

    def _load(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, png: bytes) -> list[OcrLine]:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            return []
        image = _to_bgr_on_white(image, np)
        scale = 1.0
        if image.shape[1] < UPSCALE_BELOW_PX:
            # Small text: twice the size with a sharp (cubic) filter. In testing this recovered
            # lines the detector skipped and fixed most misread letters ("Dilarong").
            scale = 2.0
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result, _elapsed = self._load()(image, use_cls=False)
        lines = []
        for box, text, score in result or []:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            if float(score) >= self.min_score and text.strip():
                lines.append(
                    OcrLine(text.strip(), min(xs), min(ys), max(xs), max(ys), float(score))
                )
        return lines


def _to_bgr_on_white(image, np):
    """Grey or colour, with or without transparency -> 3-channel image on a white background
    (transparent pixels would otherwise turn black and hide dark text)."""
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if image.shape[2] == 2:  # grey + alpha
        alpha = image[:, :, 1:2].astype(np.float32) / 255.0
        grey = image[:, :, 0:1].astype(np.float32)
        return np.repeat(grey * alpha + 255.0 * (1.0 - alpha), 3, axis=2).astype(np.uint8)
    return image


def unavailable_reason() -> str | None:
    """Why OCR cannot run here (None when it can): a missing package, or OpenCV missing a
    system library such as libGL or libglib, which a plain import error would hide."""
    try:
        import cv2  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def available() -> bool:
    """True when the OCR dependencies are installed and importable."""
    return unavailable_reason() is None


def section_labels(lines: list[OcrLine]) -> list[str]:
    """The labels of the side column (section names), top to bottom; empty for other layouts."""
    if not lines:
        return []
    line_height = median(line.height for line in lines) or 1.0
    labels = _side_labels(lines)
    rows = _rows([line for line in lines if line not in labels], line_height)
    texts = [_join_row(row, line_height) for row in rows]
    return [text for _index, text in _place_labels(labels, rows, texts, line_height)]


def lines_to_text(lines: list[OcrLine]) -> str:
    """Rebuild reading order: lines whose boxes share a row are joined left to right, with
    " | " between cells that are far apart (table columns), rows top to bottom.

    A narrow label column beside body text (the code of conduct's "3 | Penyalahgunaan Jabatan
    dan Gratifikasi" boxes) is kept out of the body rows. Joined into them, its words landed in
    the middle of sentences ("... Hari Raya Keagamaan Gratifikasi | dari nasabah ..."). Each
    label is put on its own line before the paragraph where its section starts.
    """
    if not lines:
        return ""
    line_height = median(line.height for line in lines) or 1.0
    labels = _side_labels(lines)
    rows = _rows([line for line in lines if line not in labels], line_height)
    texts = [_join_row(row, line_height) for row in rows]
    before: dict[int, list[str]] = {}
    for index, label in _place_labels(labels, rows, texts, line_height):
        before.setdefault(index, []).append(label)
    out = []
    for index, text in enumerate(texts):
        out += before.get(index, [])
        out.append(text)
    out += before.get(len(texts), [])
    return "\n".join(out)


def _rows(lines: list[OcrLine], line_height: float) -> list[list[OcrLine]]:
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: item.center_y):
        if rows and abs(line.center_y - _row_center(rows[-1])) <= line_height / 2:
            rows[-1].append(line)
        else:
            rows.append([line])
    return rows


def _join_row(row: list[OcrLine], line_height: float) -> str:
    row = sorted(row, key=lambda item: item.x0)
    text = row[0].text
    for previous, line in zip(row, row[1:], strict=False):
        gap = line.x0 - previous.x1
        text += (" | " if gap > 2 * line_height else " ") + line.text
    return text


def _side_labels(lines: list[OcrLine]) -> list[OcrLine]:
    """Lines that sit entirely left of a column of long prose lines, when they are few.

    Tables are left alone: their cells are short, so there is no prose column, or their first
    column has a cell on most rows (more than 30% of the lines)."""
    width = max(line.x1 for line in lines) - min(line.x0 for line in lines)
    prose = [line for line in lines if line.x1 - line.x0 >= 0.5 * width]
    if len(prose) < 0.3 * len(lines):
        return []
    body_left = min(line.x0 for line in prose)
    labels = [line for line in lines if line.x1 < body_left]
    return labels if 0 < len(labels) <= 0.3 * len(lines) else []


def _place_labels(
    labels: list[OcrLine], rows: list[list[OcrLine]], texts: list[str], line_height: float
) -> list[tuple[int, str]]:
    """For each group of nearby labels (one table cell), the body row it goes before: the top
    of the image for the first group (sections start there), otherwise the last paragraph start
    at or above the label's top. A label is centred in its cell, so its top is a little below
    the first line of its section."""
    groups: list[list[OcrLine]] = []
    for line in sorted(labels, key=lambda item: item.y0):
        if groups and line.y0 - max(item.y1 for item in groups[-1]) <= 3 * line_height:
            groups[-1].append(line)
        else:
            groups.append([line])
    starts = [0] + [
        i for i in range(1, len(texts)) if texts[i - 1].rstrip().endswith((".", ":", ";", ")"))
    ]
    placed: list[tuple[int, str]] = []
    last = -1
    for number, group in enumerate(groups):
        top = min(item.y0 for item in group) + line_height / 2
        candidates = [i for i in starts if i > last and _row_center(rows[i]) <= top]
        later = [i for i in starts if i > last]
        if number == 0:
            index = 0
        else:
            index = candidates[-1] if candidates else (later[0] if later else len(texts))
        # The section number ("3") is its own column: put it first, then the name in order.
        numbers = [item for item in group if item.text.strip().isdigit()]
        names = [item for item in group if item not in numbers]
        text = " ".join(
            [item.text for item in sorted(numbers, key=lambda item: item.y0)]
            + [_join_row(row, line_height) for row in _rows(names, line_height)]
        )
        placed.append((index, text))
        last = index
    return placed


def _row_center(row: list[OcrLine]) -> float:
    return sum(line.center_y for line in row) / len(row)