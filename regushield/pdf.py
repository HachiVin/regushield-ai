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


def available() -> bool:
    """True when the OCR dependencies are installed."""
    try:
        import cv2  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:  # ImportError, or OpenCV missing a system library such as libGL
        return False
    return True


def lines_to_text(lines: list[OcrLine]) -> str:
    """Rebuild reading order: lines whose boxes share a row are joined left to right, with
    " | " between cells that are far apart (table columns), rows top to bottom."""
    if not lines:
        return ""
    line_height = median(line.height for line in lines) or 1.0
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: item.center_y):
        if rows and abs(line.center_y - _row_center(rows[-1])) <= line_height / 2:
            rows[-1].append(line)
        else:
            rows.append([line])

    out = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        text = row[0].text
        for previous, line in zip(row, row[1:], strict=False):
            gap = line.x0 - previous.x1
            text += (" | " if gap > 2 * line_height else " ") + line.text
        out.append(text)
    return "\n".join(out)


def _row_center(row: list[OcrLine]) -> float:
    return sum(line.center_y for line in row) / len(row)