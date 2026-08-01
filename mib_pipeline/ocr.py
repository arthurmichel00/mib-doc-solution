"""OCR of scanned pages: rendering, preprocessing ladder, line recovery.

The engine is abstracted behind OcrEngine so the pytesseract subprocess
backend used in development can be swapped for tesserocr (C API, model
loaded once per worker) in the Docker build without touching callers.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pymupdf

from .model import Line, Page, Source
from .visibility import redact_hidden_text

RENDER_DPI = 288
_MIN_WORD_CONF = 30          # drop OCR words below this confidence
_LOW_CONTRAST_RANGE = 120    # 2..98 percentile spread that triggers CLAHE
_DESKEW_MIN_ANGLE = 0.4      # degrees


@dataclass(frozen=True)
class OcrWord:
    text: str
    conf: float               # 0..1
    line_key: tuple[int, int, int]
    x: float


class OcrEngine(Protocol):
    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        """Recognize words with confidences on a grayscale/binary image.

        sparse=True switches to sparse-text segmentation (Tesseract psm 11),
        which recovers isolated rows on noisy pages where block segmentation
        discards them.
        """


class PytesseractEngine:
    """Development backend; shells out to the tesseract binary per call."""

    def __init__(self, psm: int = 6, sparse_psm: int = 11) -> None:
        self._psm = psm
        self._sparse_psm = sparse_psm

    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        import pytesseract

        config = f"--oem 1 --psm {self._sparse_psm if sparse else self._psm}"
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT
        )
        out = []
        for text, conf, block, par, line, left in zip(
            data["text"], data["conf"], data["block_num"], data["par_num"],
            data["line_num"], data["left"],
        ):
            text = text.strip()
            conf = float(conf)
            if text and conf >= _MIN_WORD_CONF:
                out.append(OcrWord(text, conf / 100.0, (block, par, line), float(left)))
        return out


class TesserocrEngine:
    """Docker backend; binds the tesseract C++ API so the model is loaded
    once per worker process instead of spawning a subprocess per call."""

    def __init__(self, psm: int = 6, sparse_psm: int = 11) -> None:
        import tesserocr

        self._tess = tesserocr
        # tesserocr enum members are plain ints; PSM(6)-style construction
        # is not supported, so keep the raw psm numbers.
        self._psm = psm
        self._sparse_psm = sparse_psm
        path = os.environ.get("TESSDATA_PREFIX")
        kwargs = {"path": path} if path else {}
        self._api = tesserocr.PyTessBaseAPI(
            lang="eng", oem=tesserocr.OEM.LSTM_ONLY, psm=self._psm, **kwargs
        )

    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        from PIL import Image

        ril = self._tess.RIL
        self._api.SetPageSegMode(self._sparse_psm if sparse else self._psm)
        self._api.SetImage(Image.fromarray(np.ascontiguousarray(image)))
        self._api.Recognize()
        it = self._api.GetIterator()
        if it is None:
            return []
        out = []
        block = par = line = 0
        for word in self._tess.iterate_level(it, ril.WORD):
            if word.IsAtBeginningOf(ril.BLOCK):
                block += 1
            if word.IsAtBeginningOf(ril.PARA):
                par += 1
            if word.IsAtBeginningOf(ril.TEXTLINE):
                line += 1
            try:
                text = (word.GetUTF8Text(ril.WORD) or "").strip()
            except RuntimeError:
                continue
            conf = float(word.Confidence(ril.WORD))
            if text and conf >= _MIN_WORD_CONF:
                bbox = word.BoundingBox(ril.WORD)
                out.append(OcrWord(text, conf / 100.0, (block, par, line),
                                   float(bbox[0]) if bbox else 0.0))
        return out


_ENGINE: OcrEngine | None = None


def default_engine() -> OcrEngine:
    """Per-process cached engine (a tesserocr API instance must be reused,
    not rebuilt per PDF). MIB_OCR_ENGINE=tesserocr|pytesseract forces a
    backend; the default prefers tesserocr when installed (the Docker image)
    and falls back to the pytesseract subprocess backend (development)."""
    global _ENGINE
    if _ENGINE is None:
        choice = os.environ.get("MIB_OCR_ENGINE", "auto")
        if choice == "pytesseract":
            _ENGINE = PytesseractEngine()
        else:
            try:
                _ENGINE = TesserocrEngine()
            except Exception:
                if choice == "tesserocr":
                    raise
                _ENGINE = PytesseractEngine()
    return _ENGINE


_RAPID_SINGLETON = None

# PP-OCRv5/v6 detection and recognition normalize the input with ImageNet
# statistics (from each model's PaddleOCR inference.yml), whereas the
# rapidocr-onnxruntime wheel defaults to [0.5, 0.5, 0.5] for its bundled
# PP-OCRv4 models. Feeding the wrong mean/std collapses the detector on our
# faint scans, so the values travel with the local-model configuration.
_PPOCR_MEAN = [0.485, 0.456, 0.406]
_PPOCR_STD = [0.229, 0.224, 0.225]

# DBPostProcess parameters from each generation's inference.yml. v5/v6 were
# trained/tuned with these; using them keeps recall on degraded text without
# hand-tuning.
_PPOCR_DET = {
    "v5": {"det_thresh": 0.3, "det_box_thresh": 0.6, "det_unclip_ratio": 1.5},
    "v6": {"det_thresh": 0.2, "det_box_thresh": 0.45, "det_unclip_ratio": 1.4},
}


def _models_dir() -> Path:
    """Root of the bundled ONNX models (COPYed to /app/models in Docker)."""
    override = os.environ.get("MIB_MODELS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "models"


def _rapid_reader():
    """Per-process PP-OCR fallback recognizer (rapidocr-onnxruntime 1.4.4).

    By default this loads the PP-OCRv5-mobile classical det+rec ONNX bundled
    under models/ppocrv5 by explicit local path — the rapidocr-onnxruntime
    runner never touches the network, a hard requirement under --network none
    (the newer unified rapidocr package downloads models on first use and
    would die offline). MIB_RAPID_MODEL=v4 falls back to the PP-OCRv4 ONNX
    bundled inside the wheel, and =v6 to models/ppocrv6, for A/B measurement.
    """
    global _RAPID_SINGLETON
    if _RAPID_SINGLETON is None:
        from rapidocr_onnxruntime import RapidOCR

        gen = os.environ.get("MIB_RAPID_MODEL", "v5")
        if gen == "v4":
            _RAPID_SINGLETON = RapidOCR(intra_op_num_threads=1)
        else:
            base = _models_dir() / ("ppocrv6" if gen == "v6" else "ppocrv5")
            _RAPID_SINGLETON = RapidOCR(
                det_model_path=str(base / "det.onnx"),
                rec_model_path=str(base / "rec.onnx"),
                cls_model_path=str(base / "cls.onnx"),
                rec_keys_path=str(base / "rec_keys.txt"),
                det_mean=_PPOCR_MEAN,
                det_std=_PPOCR_STD,
                intra_op_num_threads=1,
                **_PPOCR_DET["v6" if gen == "v6" else "v5"],
            )
    return _RAPID_SINGLETON


def rapid_lines(gray: np.ndarray, page_index: int) -> list[Line]:
    """Second-opinion pass for pages Tesseract cannot read.

    PP-OCR's detector+recognizer survives degradation profiles (washout,
    JPEG mush) that defeat Tesseract's block segmentation. Boxes are grouped
    into reading-order lines; the recognizer tends to drop spaces, which the
    anchor matcher's no-space fallback absorbs. Field reads only — findings
    and other tier-1 evidence still require the primary path's provenance.
    """
    try:
        result, _ = _rapid_reader()(gray)
    except Exception:
        return []
    if not result:
        return []
    rows: list[tuple[float, float, str, float]] = []
    for box, text, score in result:
        if not text.strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        rows.append((sum(ys) / len(ys), min(xs), text.strip(), float(score)))
    rows.sort(key=lambda r: (r[0], r[1]))
    lines: list[Line] = []
    current: list[tuple[float, float, str, float]] = []
    for row in rows:
        if current and abs(row[0] - current[0][0]) > 18:
            lines.append(_merge_rapid_row(current, page_index))
            current = []
        current.append(row)
    if current:
        lines.append(_merge_rapid_row(current, page_index))
    return lines


# PP-OCR's recognizer scores run systematically hotter than Tesseract's
# word confidences for comparable read quality; scale them onto the same
# axis so cross-engine voting stays fair.
_RAPID_CONF_SCALE = 0.85


def _merge_rapid_row(row: list[tuple[float, float, str, float]],
                     page_index: int) -> Line:
    row.sort(key=lambda r: r[1])
    text = " ".join(r[2] for r in row)
    conf = float(np.mean([r[3] for r in row])) * _RAPID_CONF_SCALE
    return Line(text=text, page_index=page_index, source=Source.OCR, conf=conf)


# Letter at 288 DPI is ~7.8 MP; the cap only reins in pathological page
# sizes that would otherwise allocate multi-GiB pixmaps.
_MAX_RENDER_PIXELS = 40_000_000


def render_gray(pdf_page: pymupdf.Page, page: Page, dpi: int = RENDER_DPI) -> np.ndarray:
    """Rasterize the page with hidden text removed first.

    Deleting hidden spans before rendering guarantees no threshold or
    contrast step downstream can resurrect white-on-white text.
    """
    redact_hidden_text(pdf_page, page.hidden_spans)
    rect = pdf_page.rect
    pixels = abs(rect) * (dpi / 72.0) ** 2
    if pixels > _MAX_RENDER_PIXELS:
        dpi = max(72, int(dpi * (_MAX_RENDER_PIXELS / pixels) ** 0.5))
    pix = pdf_page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def _otsu(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (2, 98))
    if hi - lo < _LOW_CONTRAST_RANGE:
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 3), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10,
    )


def _stretch(gray: np.ndarray) -> np.ndarray:
    """Contrast-stretch for washed-out pages (ink barely below paper)."""
    lo, hi = np.percentile(gray, (1, 60))
    if hi - lo < 1:
        return gray
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo),
                   0, 255).astype(np.uint8)


def _sauvola(gray: np.ndarray, win: int = 25, k: float = 0.2) -> np.ndarray:
    """Local (Sauvola) threshold: survives gradients that defeat global Otsu."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win))
    sq = cv2.boxFilter(g * g, -1, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + k * (std / 128.0 - 1))
    return ((g > thr) * 255).astype(np.uint8)


def _divblur(gray: np.ndarray) -> np.ndarray:
    """Divide-by-background: lifts ghosted/overprinted text off the paper."""
    bg = cv2.GaussianBlur(gray, (0, 0), 21)
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def _up2(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, None, fx=2.0, fy=2.0,
                      interpolation=cv2.INTER_LANCZOS4)


def _bgsub(gray: np.ndarray) -> np.ndarray:
    """Median-background subtraction with residual amplification.

    Lifts faint ink sitting just below a translucent overlay or wash —
    the damage class _divblur's divide-normalization leaves under
    threshold (human review, MIB-000051: sponsor digit at ~10 grey
    levels below a box fill of ~237). Median (not Gaussian) background
    keeps glyph strokes out of the estimate, so thin ink survives."""
    f = cv2.medianBlur(gray, 3).astype(np.float32)
    bg = cv2.medianBlur(gray, 61).astype(np.float32)
    residual = np.clip(bg - f, 0, None)
    return (255.0 - np.clip(residual * 8.0, 0, 255)).astype(np.uint8)


# Last-resort escalation ladder, ordered by measured cost. Each variant
# recovered a different washed receipt on train (334 divblur/up2, 107
# sauvola, 404 up2+sauvola); 0 wrong verdicts across the 136-case
# R8_fee_unread population, so escalating cannot flip a read badly.
ESCALATION_VARIANTS = ("divblur", "sauvola", "up2_raw", "up2_sauvola")
# Optional tail of the ladder (MIB_BGSUB=1): runs only on cases the four
# shipped variants + PP-OCR still leave under-determined, so its cost is
# confined to the damaged tail and bounded by the escalation soft budget.
BGSUB_VARIANTS = ("bgsub", "up2_bgsub")
_ESCALATION_FNS = {
    "divblur": _divblur,
    "sauvola": _sauvola,
    "up2_raw": _up2,
    "up2_sauvola": lambda g: _sauvola(_up2(g)),
    "bgsub": _bgsub,
    "up2_bgsub": lambda g: _bgsub(_up2(g)),
}


def pad_for_ocr(gray: np.ndarray, margin: int = 30) -> np.ndarray:
    """White border pad: tesseract degrades on glyphs touching the image
    edge (clipped ascenders on rotated note pages read as noise)."""
    return cv2.copyMakeBorder(gray, margin, margin, margin, margin,
                              cv2.BORDER_CONSTANT, value=255)


def escalation_lines(gray: np.ndarray, page_index: int, variant: str,
                     engine: OcrEngine) -> list[Line]:
    img = _ESCALATION_FNS[variant](gray)
    lines = _words_to_lines(engine.words(img), page_index)
    lines.extend(_words_to_lines(engine.words(img, sparse=True), page_index))
    return lines


# --- Cut-strip weld (sponsor-only) -----------------------------------------
# Damage model identified by human review: the generator displaces rectangular
# patches of page CONTENT — a text row is cut horizontally mid-glyph and the
# top strip drawn shifted (weld family), or broken into full-height segments
# with per-segment offsets (staircase family). Welding the strips back makes
# the row OCR-readable again (verified: MIB-000027 'Sponsor ID: SPN-1345').
# Corpus measurement (450 miss cases): recoveries concentrate in sponsor rows
# (4 correct / 1 wrong / 0 revoked-list wrongs), so only sponsor-bearing weld
# lines are emitted; a welded misread must never mint an R4 denial, so
# revoked-list values are dropped (worst case = the field stays unread).

_WELD_MIN_DX = 12
_WELD_MAX_DX = 900
_WELD_MARGIN = 1.15
_WELD_BAND_MIN_H = 14


def _weld_bands(ink: np.ndarray) -> list[tuple[int, int]]:
    prof = ink.sum(axis=1).astype(float)
    if prof.max() <= 0:
        return []
    on = prof > prof.max() * 0.04
    bands: list[list[int]] = []
    start = None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start >= _WELD_BAND_MIN_H:
                bands.append([max(0, start - 4), min(len(on), y + 4)])
            start = None
    if start is not None and len(on) - start >= _WELD_BAND_MIN_H:
        bands.append([max(0, start - 4), len(on)])
    merged: list[list[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 8:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(b[0], b[1]) for b in merged]


def _weld_row_candidates(band_ink: np.ndarray, topk: int = 3) -> list[np.ndarray]:
    h, w = band_ink.shape
    if h < _WELD_BAND_MIN_H or band_ink.sum() < 3000:
        return []
    scored = []
    for cut in range(6, h - 5, 2):
        top, bot = band_ink[:cut], band_ink[cut:]
        if top.sum() < 1500 or bot.sum() < 1500:
            continue
        tedge = top[-3:].sum(axis=0).astype(float)
        bedge = bot[:3].sum(axis=0).astype(float)
        base = float(np.minimum(tedge, bedge).sum()) + 1e-6
        for sign in (1, -1):
            for dx in range(_WELD_MIN_DX, _WELD_MAX_DX, 2):
                d = sign * dx
                shifted = np.roll(tedge, -d)
                if d > 0:
                    shifted[-d:] = 0
                else:
                    shifted[:-d] = 0
                cont = float(np.minimum(shifted, bedge).sum())
                if cont > base * _WELD_MARGIN:
                    scored.append((cont, cut, d))
    scored.sort(reverse=True)
    out: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for cont, cut, d in scored:
        key = (cut // 5, d // 16)
        if key in seen:
            continue
        seen.add(key)
        canvas = np.zeros_like(band_ink, dtype=float)
        canvas[cut:] = band_ink[cut:]
        M = np.float32([[1, 0, -d], [0, 1, 0]])
        canvas[:cut] = cv2.warpAffine(
            band_ink[:cut].astype(float), M, (band_ink.shape[1], cut))
        out.append(255 - np.clip(canvas, 0, 255).astype(np.uint8))
        if len(seen) >= topk:
            break
    return out


def weld_sponsor_lines(gray: np.ndarray, page_index: int,
                       engine: OcrEngine,
                       deadline: float | None = None) -> list[Line]:
    """Weld displaced sponsor rows back together and read only SPN values.

    Emits a Line only when the welded row's OCR yields a well-formed,
    NON-revoked sponsor id — every other read is discarded, so the worst a
    bad weld can do is leave the field unread exactly as it was.

    deadline (time.monotonic timestamp) bounds the band scan: the weld is
    a last-resort engine and must never drive a heavy packet into the hard
    per-case deadline, where the fallback row discards won evidence.
    """
    import time as _time

    from . import policy
    from . import vocab

    lines: list[Line] = []
    for k in range(4):
        if deadline is not None and _time.monotonic() > deadline:
            return lines
        rot = np.ascontiguousarray(np.rot90(gray, k=k)) if k else gray
        ink = (255 - rot).astype(np.uint16)
        for y0, y1 in _weld_bands(ink):
            if deadline is not None and _time.monotonic() > deadline:
                return lines
            band = ink[y0:y1].astype(np.uint8)
            for img in _weld_row_candidates(band):
                up = cv2.resize(img, None, fx=2, fy=2,
                                interpolation=cv2.INTER_LANCZOS4)
                words = engine.words(up)
                if not words:
                    continue
                text = " ".join(w.text for w in words)
                spn = vocab.repair_sponsor_id(text)
                if spn is None or spn in policy.REVOKED_SPONSORS:
                    continue
                conf = float(np.mean([w.conf for w in words])) * 0.85
                # Extraction-only: cap below the affirmative-read threshold
                # (0.55) so a weld fill can populate the FIELD but never set
                # known=True — a wrong well-formed weld could otherwise skip
                # R12_sponsor_unread and open the approve gate (external
                # review reproduced that path; adjudication must never rest
                # on a reconstructed read).
                conf = min(conf, 0.50)
                lines.append(Line(text=text, page_index=page_index,
                                  source=Source.OCR, conf=conf))
    return lines


def _remove_ruled_lines(bw: np.ndarray) -> np.ndarray:
    """Erase the notebook ruling that fragments OCR on table-style scans."""
    inverted = cv2.bitwise_not(bw)
    horizontal = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    vertical = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical),
                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.bitwise_or(bw, lines)


def _estimate_skew(gray: np.ndarray) -> float:
    """Skew angle from the ruled notebook lines (near-horizontal Hough)."""
    small = cv2.resize(gray, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=120,
        minLineLength=small.shape[1] // 3, maxLineGap=8,
    )
    if lines is None:
        return 0.0
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)
        if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 20
    ]
    return float(np.median(angles)) if angles else 0.0


def _rotate_small(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _best_orientation(gray: np.ndarray, engine: OcrEngine) -> int:
    """Pick the 90-degree orientation with the strongest OCR response.

    Template label words weigh heavily: the always-upright footer gives
    every rotation a confident-word floor, but only the true orientation
    reads the body's vocabulary.
    """
    small = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    best_rot, best_score = 0, -1.0
    for rot in (0, 1, 2, 3):
        img = np.rot90(small, k=rot) if rot else small
        words = engine.words(_otsu(np.ascontiguousarray(img)))
        score = sum(w.conf for w in words if w.conf > 0.5)
        score += 4.0 * sum(
            1 for w in words if w.conf > 0.5 and _ANCHORISH_RE.search(w.text)
        )
        if score > best_score:
            best_rot, best_score = rot, score
    return best_rot


def _words_to_lines(words: list[OcrWord], page_index: int) -> list[Line]:
    groups: dict[tuple[int, int, int], list[OcrWord]] = {}
    for word in words:
        groups.setdefault(word.line_key, []).append(word)
    lines = []
    for group in groups.values():
        group.sort(key=lambda w: w.x)
        text = " ".join(w.text for w in group)
        conf = float(np.mean([w.conf for w in group]))
        lines.append(Line(text=text, page_index=page_index, source=Source.OCR, conf=conf))
    return lines


_FOOTER_WORDS = frozenset(
    "packet page synthetic hiring challenge document".split()
)
_ANCHORISH_RE = re.compile(
    r"(?i)\b(applicant|species|sponsor|arrival|observed|finding|registry|"
    r"visa|home|purpose|amount|waiver|status|receipt|intake|biometric|"
    r"attestation|adjudicator)\b"
)


def _content_words(words: list[OcrWord]) -> int:
    """Confident words excluding the always-upright digital footer.

    The footer is vector text stamped on every page, so it OCRs cleanly
    even when the scanned content is rotated 90/180/270 degrees — counting
    it would mask exactly the pages that need the orientation retry.
    """
    return sum(
        1 for w in words
        if w.conf > 0.5 and w.text.lower() not in _FOOTER_WORDS
        and not w.text.upper().startswith("MIB-")
    )


def _looks_readable(words: list[OcrWord]) -> bool:
    """True when the upright pass found real template vocabulary.

    Rotated pages still emit plenty of confident junk (vertical glyph
    columns read as short high-confidence "words"), so only genuine label
    words count — every template prints several.
    """
    anchorish = sum(
        1 for w in words if w.conf > 0.5 and _ANCHORISH_RE.search(w.text)
    )
    return anchorish >= 2


@dataclass
class ScanOcrResult:
    lines: list[Line]
    gray: np.ndarray          # orientation-fixed, deskewed render
    upright: bool             # False when a 90/180/270 rotation was applied
    best_rot: int = 0         # estimated np.rot90 k for non-upright pages
    # True when the upright pass did not read as a known template and the
    # orientation estimator had to guess (upright=True + True means the
    # estimator scored k=0 highest). Metadata only — no ladder behavior
    # depends on it; the A2 rotation probe uses it as its trigger.
    orientation_estimated: bool = False


def ocr_scan_page(gray: np.ndarray, page_index: int,
                  engine: OcrEngine) -> ScanOcrResult:
    """Run the preprocessing ladder and pool recovered lines.

    Different variants recover different lines on damaged pages (verified:
    faint gray text survives the raw pass but dies under Otsu, washed-out
    lines need contrast stretching or adaptive thresholding, and ruled
    notebook lines fragment table rows), so results are pooled and
    deduplicated rather than picking a single winning variant.
    """
    raw_words = engine.words(gray)
    upright = True
    orientation_estimated = False
    pooled = _words_to_lines(raw_words, page_index)

    # Orientation handling is ADDITIVE (v1.2 lesson: replacing the base
    # image on an estimator's say-so poisons the whole ladder when the
    # estimator is wrong). The upright passes always run; when the page
    # does not read as a known template, every counter-rotation gets its
    # own passes pooled on top — choosing a single "best" rotation from a
    # downscaled probe mis-picks on faint pages, and pooled junk from the
    # wrong rotations is discarded by the vocabulary thresholds anyway.
    if not _looks_readable(raw_words):
        orientation_estimated = True
        best_rot = _best_orientation(gray, engine)
        if best_rot:
            upright = False
        for rot in (1, 2, 3):
            rotated = np.ascontiguousarray(np.rot90(gray, k=rot))
            passes = ((rotated, False), (rotated, True))
            if rot == best_rot:
                passes += ((_otsu(rotated), False),)
            for img, sparse in passes:
                pooled.extend(_words_to_lines(engine.words(img, sparse=sparse),
                                              page_index))

    otsu = _otsu(gray)
    for img in (otsu, _adaptive(gray), _otsu(_stretch(gray))):
        pooled.extend(_words_to_lines(engine.words(img), page_index))
    pooled.extend(_words_to_lines(engine.words(gray, sparse=True), page_index))

    # Fine deskew is additive, never a replacement: a bad Hough estimate on
    # a decoy-line page must not poison the whole ladder.
    skew = _estimate_skew(gray)
    if abs(skew) >= _DESKEW_MIN_ANGLE:
        deskewed = _rotate_small(gray, skew)
        pooled.extend(_words_to_lines(engine.words(deskewed), page_index))
        pooled.extend(_words_to_lines(engine.words(deskewed, sparse=True),
                                      page_index))
        pooled.extend(_words_to_lines(engine.words(_otsu(deskewed)), page_index))

    strong_lines = sum(1 for l in pooled if l.conf > 0.6 and len(l.text) > 6)
    if strong_lines < 12:
        pooled.extend(_words_to_lines(
            engine.words(_remove_ruled_lines(otsu)), page_index))

    # Every scan template draws its field rows in the top band of the page;
    # re-OCRing that band as a crop beats full-page passes on damaged pages
    # (verified: a receipt whose rows all six full-page variants missed
    # reads "Fee Status: paid" at 0.95 from the crop). Upright pages only —
    # counter-rotated content does not sit at template coordinates.
    if upright:
        scale = RENDER_DPI / 72.0
        band = gray[int(30 * scale):int(350 * scale),
                    int(40 * scale):int(500 * scale)]
        if band.size:
            band = np.ascontiguousarray(band)
            for img, sparse in ((band, False), (_otsu(band), False),
                                (_adaptive(band), True)):
                pooled.extend(_words_to_lines(engine.words(img, sparse=sparse),
                                              page_index))
            band_yield = sum(
                1 for l in pooled if l.conf > 0.6 and len(l.text) > 6)
            if band_yield < 14:
                # faint crops sometimes only resolve after interpolation
                big = cv2.resize(band, None, fx=1.6, fy=1.6,
                                 interpolation=cv2.INTER_LANCZOS4)
                pooled.extend(_words_to_lines(engine.words(_otsu(big)),
                                              page_index))

    best: dict[str, Line] = {}
    for line in pooled:
        key = " ".join(line.text.lower().split())
        if key not in best or line.conf > best[key].conf:
            best[key] = line
    return ScanOcrResult(lines=list(best.values()), gray=gray, upright=upright,
                         best_rot=0 if upright else best_rot,
                         orientation_estimated=orientation_estimated)
