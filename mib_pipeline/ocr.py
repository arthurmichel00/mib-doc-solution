"""OCR of scanned pages: rendering, preprocessing ladder, line recovery.

The engine is abstracted behind OcrEngine so the pytesseract subprocess
backend used in development can be swapped for tesserocr (C API, model
loaded once per worker) in the Docker build without touching callers.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pymupdf

from . import row_restore
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
    # Full word box (same pixel space as the image the pass read). Both
    # backends always report it; the defaults only keep old positional
    # constructions (tests/fixtures) valid. Consumed by the MIB_CTCFILL
    # label locator (ctcfill.py) — line assembly still keys on x alone.
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


class OcrEngine(Protocol):
    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        """Recognize words with confidences on a grayscale/binary image.

        sparse=True switches to sparse-text segmentation (Tesseract psm 11),
        which recovers isolated rows on noisy pages where block segmentation
        discards them.
        """


class PytesseractEngine:
    """Development backend; shells out to the tesseract binary per call.

    config_extra is appended verbatim to every call's config string (used
    by the MIB_USERWORDS engine to pass --user-words/--user-patterns).

    lang/tessdata_dir select a non-default recognition model (the
    MIB_TESSFT fine-tuned engine); both stay unset for the stock engine, so
    the emitted tesseract command line is unchanged.
    """

    def __init__(self, psm: int = 6, sparse_psm: int = 11,
                 config_extra: str = "", lang: str | None = None,
                 tessdata_dir: str | None = None) -> None:
        self._psm = psm
        self._sparse_psm = sparse_psm
        self._lang = lang
        if tessdata_dir:
            config_extra = f'--tessdata-dir "{tessdata_dir}" {config_extra}'
        config_extra = config_extra.strip()
        self._config_extra = f" {config_extra}" if config_extra else ""

    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        import pytesseract

        config = (f"--oem 1 --psm {self._sparse_psm if sparse else self._psm}"
                  f"{self._config_extra}")
        extra = {"lang": self._lang} if self._lang else {}
        data = pytesseract.image_to_data(
            image, config=config, output_type=pytesseract.Output.DICT, **extra
        )
        out = []
        for text, conf, block, par, line, left, top, width, height in zip(
            data["text"], data["conf"], data["block_num"], data["par_num"],
            data["line_num"], data["left"], data["top"], data["width"],
            data["height"],
        ):
            text = text.strip()
            conf = float(conf)
            if text and conf >= _MIN_WORD_CONF:
                out.append(OcrWord(text, conf / 100.0, (block, par, line),
                                   float(left), float(top), float(width),
                                   float(height)))
        return out


class TesserocrEngine:
    """Docker backend; binds the tesseract C++ API so the model is loaded
    once per worker process instead of spawning a subprocess per call.

    init_variables are handed to PyTessBaseAPI's `variables` kwarg, which
    tesserocr applies DURING Init — required for init-only params such as
    user_words_file/user_patterns_file (the MIB_USERWORDS engine).

    lang/path select a non-default recognition model: `path` is the
    tessdata directory to resolve `<lang>.traineddata` in, overriding
    TESSDATA_PREFIX (the MIB_TESSFT fine-tuned engine loads lang="mib" out
    of models/tessdata). Both default to the stock eng model under
    TESSDATA_PREFIX, so the stock engine's Init is unchanged.
    """

    def __init__(self, psm: int = 6, sparse_psm: int = 11,
                 init_variables: dict[str, str] | None = None,
                 lang: str = "eng", path: str | None = None) -> None:
        import tesserocr

        self._tess = tesserocr
        # tesserocr enum members are plain ints; PSM(6)-style construction
        # is not supported, so keep the raw psm numbers.
        self._psm = psm
        self._sparse_psm = sparse_psm
        path = path or os.environ.get("TESSDATA_PREFIX")
        kwargs = {"path": path} if path else {}
        if init_variables:
            kwargs["variables"] = dict(init_variables)
        self._api = tesserocr.PyTessBaseAPI(
            lang=lang, oem=tesserocr.OEM.LSTM_ONLY, psm=self._psm, **kwargs
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
                bbox = word.BoundingBox(ril.WORD)   # (x1, y1, x2, y2)
                if bbox:
                    x, y = float(bbox[0]), float(bbox[1])
                    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
                else:
                    x = y = w = h = 0.0
                out.append(OcrWord(text, conf / 100.0, (block, par, line),
                                   x, y, w, h))
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


# --- Vocabulary-constrained escalation pass (MIB_USERWORDS) -----------------
# Every decision input the pipeline reads comes from a closed vocabulary
# (vocab.py) printed after a fixed set of form labels. Tesseract's LSTM beam
# search consults word dawgs, so loading those strings as user words biases
# recognition on damaged rows toward exactly the tokens the downstream
# matchers can consume — without touching a single matcher threshold. The
# word list is built at import time from the closed sets; the dawg files are
# written to tmpfiles once per process, on first use.

# Form labels every template prints in front of its values.
_USERWORDS_LABELS = (
    "Home World", "Visa Class", "Sponsor ID", "Arrival Date",
    "Declared Purpose", "Species Code", "Fee Status", "Observed Flags",
    "FINDING", "Reason",
)


def _build_userwords() -> tuple[str, ...]:
    """Ordered, de-duplicated word list from vocab.py's closed sets.

    Tesseract user-words dawgs are word-level: multi-token phrases must be
    split into their tokens. Species codes keep the printed underscore form
    AND contribute their space-variant tokens (OCR reads both).
    """
    from . import vocab

    words: dict[str, None] = {}          # insertion-ordered de-dupe

    def add_tokens(phrase: str) -> None:
        for token in str(phrase).split():
            if token:
                words.setdefault(token, None)

    for code in vocab.SPECIES_CODES:
        words.setdefault(code, None)     # printed underscore form
        add_tokens(code.replace("_", " "))
    for world in vocab.HOME_WORLDS:
        add_tokens(world)
    for cls in vocab.VISA_CLASSES:
        words.setdefault(cls, None)
    for purpose in vocab.PURPOSES:
        add_tokens(purpose)
    for status in vocab.FEE_STATUSES:
        words.setdefault(status, None)
    for label in _USERWORDS_LABELS:
        add_tokens(label)
    return tuple(words)


USERWORDS = _build_userwords()

# trie.cpp pattern syntax (\d = digit unichar): sponsor ids and ISO dates,
# the two decision-bearing pattern-shaped values.
USERPATTERNS = (r"SPN-\d\d\d\d", r"\d\d\d\d-\d\d-\d\d")

_USERWORDS_PATH: str | None = None
_USERPATTERNS_PATH: str | None = None


def _dict_tmpfile(entries: tuple[str, ...], prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(entries) + "\n")
    return path


def userwords_path() -> str:
    """Per-process user-words tmpfile (one word per line), written once."""
    global _USERWORDS_PATH
    if _USERWORDS_PATH is None:
        _USERWORDS_PATH = _dict_tmpfile(USERWORDS, "mib_userwords_")
    return _USERWORDS_PATH


def userpatterns_path() -> str:
    """Per-process user-patterns tmpfile, written once."""
    global _USERPATTERNS_PATH
    if _USERPATTERNS_PATH is None:
        _USERPATTERNS_PATH = _dict_tmpfile(USERPATTERNS, "mib_userpatterns_")
    return _USERPATTERNS_PATH


_USERWORDS_ENGINE: OcrEngine | None = None


def userwords_engine() -> OcrEngine:
    """Per-process engine with the closed-vocabulary dictionaries loaded.

    Same backend selection as default_engine. user_words_file /
    user_patterns_file are init-only tesseract params, so this is a second
    API instance, cached like the default one (verified in-container:
    tesserocr 2.8.0 applies the `variables` kwarg during Init and tesseract
    5.3's LSTM beam search consults the resulting dawgs).
    """
    global _USERWORDS_ENGINE
    if _USERWORDS_ENGINE is None:
        words, patterns = userwords_path(), userpatterns_path()
        choice = os.environ.get("MIB_OCR_ENGINE", "auto")

        def _pytess() -> PytesseractEngine:
            return PytesseractEngine(
                config_extra=f"--user-words {words} "
                             f"--user-patterns {patterns}")

        if choice == "pytesseract":
            _USERWORDS_ENGINE = _pytess()
        else:
            try:
                _USERWORDS_ENGINE = TesserocrEngine(init_variables={
                    "user_words_file": words,
                    "user_patterns_file": patterns,
                })
            except Exception:
                if choice == "tesserocr":
                    raise
                _USERWORDS_ENGINE = _pytess()
    return _USERWORDS_ENGINE


def userwords_lines(gray: np.ndarray, page_index: int) -> list[Line]:
    """ONE vocabulary-constrained pass over one escalation view.

    Emitted lines are ordinary OCR lines: they rejoin the exact same
    anchoring/vocabulary/reconciliation path as every escalation variant
    and are protected by the same escalation restore guard (fill unread
    fields only; an affirmative read is never out-voted).
    """
    return _words_to_lines(userwords_engine().words(gray), page_index)


# --- Fine-tuned-font escalation pass (MIB_TESSFT) ---------------------------
# A second tesseract LSTM whose recognizer was fine-tuned on the challenge
# generator's own font (artifact by Shrey Shingala, MIT — see
# ATTRIBUTION.md; held-out CER 9.59% -> 0.19% on that font). It is a real
# recognizer, not a reconstruction, so its reads flow through the ordinary
# pooling and confidence thresholds like any other engine's.
#
# It is NOT a corpus-wide second engine: the pass fires only from the
# lazily-escalated tail in pipeline.py, on cases where both shipped engines
# and the whole preprocessing ladder left a decision input unread.
#
# And within those cases it does not read PAGES, it reads label-anchored
# VALUE STRIPS for the specific fields still unread (locator mechanism from
# ctcfill.locate_strips, reusing the word-box geometry the ladder already
# stashed — no new OCR pass to find them).
#
# That is the whole reason the lever is affordable. The arithmetic, against
# a 6.00 s/PDF hard-DQ contract the build already sits at 5.81 under, i.e.
# 0.19 s/PDF of headroom:
#
#   one fine-tuned pass over a full 288-DPI render  0.371 s (stock 0.185 s)
#   escalation-eligible ceiling                     851/1000 packets
#     (149 carry no scan page and can never fire)
#   scan pages per scan-bearing packet              ~2.3
#   => PAGE-level cost   0.851 x 2.3 x 0.371     =  ~0.73 s/PDF   4x headroom
#
#   fine-tuned pass over one located strip          0.0167-0.0200 s
#   strips per escalated packet (measured)          1.35
#   => STRIP-level cost  0.851 x ~0.023          =  ~0.020 s/PDF
#
# The strip figure holds at the CEILING of the trigger rate, so it does not
# depend on estimating how often escalation actually fires — which is just
# as well, since that is bounded below at 320/1000 (the unread-path
# population) and estimated at 60-75% of scan-bearing packets, but was
# never directly censused. A strip is ~1/50th of a page, and that ratio is
# the lever.
#
# Strip anchoring is also the safety mechanism. The locator only emits a
# strip when the field's printed LABEL was read at conf >= 0.60, so the
# crop provably contains generator text — the model is never handed the
# blank/noisy regions where (see below) it hallucinates confidently.
#
# NOT A REPEAT OF MIB_USERWORDS (ledgered NO-SHIP at -0.22, research/
# LEDGER.md:392-399, whose anti-repeat reads "user-words/user-patterns on
# the escalation ladder measured NET-NEGATIVE; do not revisit without a
# different integration design"). That lever biased a general recognizer's
# DECODER toward the closed vocabulary, and its own post-mortem named the
# mechanism: "vocab-biased decoding corrupts reads" — extraction fell 45.30
# -> 45.18. This lever changes the MODEL, not the decode: a second LSTM
# whose weights were fit to the glyph shapes, with no vocabulary prior
# imposed at all. The two differ in what they are allowed to get wrong. A
# vocabulary prior can only pull a read TOWARD a legal string, so when it
# is wrong it is wrong in exactly the direction the matchers cannot catch;
# a recognizer that misreads a glyph produces a string the matchers reject.
# The shared failure mode is nonetheless real and is what the strip-level
# tests guard: on a CLEAN row the fine-tuned pass must not out-vote a
# correct stock read. Pooling keeps the stock line, the restore guard puts
# affirmative reads back, and the 0.75 scale caps what an invented read can
# claim.
#
# Two further inherited cautions from that post-mortem, both for the A/B to
# adjudicate rather than anything native measurement can settle:
#   - Its 8 changed rows clustered in the corpus tail and were attributed
#     to an extra pass perturbing per-worker tesseract state, i.e. an extra
#     pass can move cases it never targeted. This pass runs on its own
#     PyTessBaseAPI instance under OEM.LSTM_ONLY, which should not share
#     adaptive state, but "should not" is a hypothesis; the corpus A/B is
#     what tests it.
#   - Its capability was verified in-image and it still lost. Passing a
#     test battery predicts nothing about the score.
#
# CONFOUND worth naming before anyone credits the fine-tuning. The
# artifact's version string is
# "4.00.00alpha:eng:synth20170629:[...Lfx512...]" — it is a fine-tune of
# tessdata_best eng (the FLOAT model), whereas our stock engine is the
# int-quantized Lfx192 build of that same lineage. So any quality gap
# measured here has two candidate causes, the font fine-tuning and the
# bigger/unquantized base, and this work does not separate them. If the
# A/B pays, the cheap follow-up is to run plain tessdata_best eng as the
# second engine: same ~2x cost, no third-party artifact, and it would say
# how much of the gain the fine-tuning is actually responsible for.
#
# Measured yield is NOT yet proven, and the strip design is the reason to
# be suspicious: requiring a confident label biases the located strips
# toward rows the stock engine already reads. On 60 escalation-population
# packets / 61 strips the two engines agreed on almost everything, with 2
# label recoveries for the fine-tuned model ("Sponsor 1D" -> "Sponsor ID",
# same value both ways) against 1 regression ("Visa Class" -> "Via
# Class") — a wash at that sample size, and zero value-level recoveries.
# Two caveats cut the other way: those runs approximated the word stash
# with ONE stock pass instead of the ladder's five, so the real locator
# sees more labels, and they sampled cases the ladder had NOT yet failed
# on, which is an easier population than the one this pass actually meets.
# Treat the cost numbers as measured and the benefit as an open question
# for the A/B — that is what the flag is for.
#
# FIT/SERVE: every number in this comment was measured natively (host
# tesseract 5.5.1) and the image runs tesseract 5.3.4, a gap documented
# five times in this project and sign-mixed each time — the note-band
# rescue mints under 5.5 and not under 5.3, while MIB-001000 reads under
# 5.3 and not under 5.5. So treat the figures below as DIRECTION, not as
# evidence about the image; the in-container A/B is the only arbiter.
# One 5.3-specific mechanism is already handled rather than hoped about:
# 5.3 clips glyphs touching a crop edge (measured 72.5 -> 94.3 word conf on
# the note band from padding alone), and every strip this pass reads is a
# raw crop, so tessft_lines runs pad_for_ocr on all of them.
#
# Conf scale, measured on 8 train packets / 14 scan pages (2026-08-03):
#
#   Genuine reads are NOT inflated. Over 409 words where both engines
#   returned the same string in the same box, mean conf was 92.68 stock vs
#   91.96 fine-tuned (mean ratio 1.0006, median 1.0000, fine-tuned higher
#   on 50.1% — a coin flip). So this is NOT a miscalibrated engine and the
#   scale is not a calibration correction.
#
#   What IS shifted is hallucination on content holding no generator text
#   at all. On textureless noise strips the fine-tuned model invents 1.75x
#   to 7x more words than stock, at materially higher confidence: stock
#   emitted nothing at sigma=45 and peaked at 50.55 elsewhere, while the
#   fine-tuned model emitted 7-13 words peaking at 63.57 / 67.67 / 73.20.
#   That is the font fit working against us — it is trained to see this
#   font, so paper grain resolves into confident glyphs.
#
#   Stock's junk therefore cannot reach fields._KNOWN_MIN_OCR_CONF (0.55)
#   and the fine-tuned model's can, which is the whole exposure: an
#   invented read crossing the affirmative-read line could mint a value on
#   a decision field that was unread — exactly the field this pass targets.
#   The scale is sized to close precisely that gap: 0.55 / 0.7320 = 0.7514,
#   so 0.75 puts the worst observed hallucination at 54.9, just under the
#   line, while the average genuine read (92.0) lands at 69.0 and stays
#   comfortably affirmative. Reads still pool and vote normally; they just
#   cannot out-vote a stock engine on invented evidence.
TESSFT_LANG = "mib"
_TESSFT_CONF_SCALE = 0.75

_TESSFT_ENGINE: OcrEngine | None = None
_TESSFT_UNAVAILABLE = False


def tessft_dir() -> Path:
    """Directory holding <TESSFT_LANG>.traineddata (COPYed to /app/models)."""
    override = os.environ.get("MIB_TESSFT_DIR")
    return Path(override) if override else _models_dir() / "tessdata"


def tessft_engine() -> OcrEngine | None:
    """Per-process engine bound to the fine-tuned traineddata, or None.

    Returns None (warning once) when the artifact or a working backend is
    absent rather than raising: the pass is an optional escalation engine
    and a missing model must degrade to "no extra pass", never break a
    case. Warning rather than silent, because a model that quietly stops
    loading in the image would look exactly like a lever that stopped
    paying (the failure mode called out in the source project's own
    Dockerfile).
    """
    global _TESSFT_ENGINE, _TESSFT_UNAVAILABLE
    if _TESSFT_ENGINE is None and not _TESSFT_UNAVAILABLE:
        directory = tessft_dir()
        model = directory / f"{TESSFT_LANG}.traineddata"
        if not model.is_file():
            _warn_tessft(f"fine-tuned model not found at {model}")
            return None
        choice = os.environ.get("MIB_OCR_ENGINE", "auto")

        def _pytess() -> PytesseractEngine:
            return PytesseractEngine(lang=TESSFT_LANG,
                                     tessdata_dir=str(directory))

        try:
            if choice == "pytesseract":
                _TESSFT_ENGINE = _pytess()
            else:
                try:
                    _TESSFT_ENGINE = TesserocrEngine(lang=TESSFT_LANG,
                                                     path=str(directory))
                except Exception:
                    if choice == "tesserocr":
                        raise
                    _TESSFT_ENGINE = _pytess()
        except Exception as exc:                     # pragma: no cover
            _warn_tessft(f"fine-tuned engine failed to load: {exc!r}")
            return None
    return _TESSFT_ENGINE


def _warn_tessft(message: str) -> None:
    global _TESSFT_UNAVAILABLE
    import sys

    _TESSFT_UNAVAILABLE = True
    print(f"[mib] MIB_TESSFT disabled: {message}", file=sys.stderr)


# Printed label token runs anchoring the value strip of each field the
# escalation gate (pipeline._under_determined) can be waiting on. Matched
# case-insensitively after punctuation stripping, like ctcfill's locator.
TESSFT_LABELS: dict[str, tuple[tuple[str, ...], ...]] = {
    "fee_status": (("Fee", "Status"),),
    "visa_class": (("Visa", "Class"),),
    "home_world": (("Home", "World"),),
    "sponsor_id": (("Sponsor", "ID"),),
    "arrival_date": (("Arrival", "Date"),),
    "risk_flags": (("Observed", "flags"),),
}

# Locator gates, mirroring ctcfill's (which were calibrated on the
# menu-sweep population): every label token needs this confidence, the
# value region runs this far right of the label's left edge, and each row
# gets this much vertical padding. _TESSFT_MAX_STRIPS bounds the per-page
# work so a page whose labels re-read many times cannot blow the budget.
_TESSFT_LOC_MIN_CONF = 0.60
_TESSFT_STRIP_W = 680
_TESSFT_STRIP_Y_PAD = 6
_TESSFT_MAX_STRIPS = 6


def tessft_strips(result: "ScanOcrResult",
                  wanted: tuple[str, ...]) -> list[np.ndarray]:
    """Label-anchored value strips for the still-unread fields of one page.

    Reads ScanOcrResult.words — geometry the ladder already produced — so
    locating costs no OCR. A label token run must match exactly after
    punctuation stripping, sit on one row reading left-to-right, and carry
    conf >= _TESSFT_LOC_MIN_CONF. Rows are deduped across fields (one strip
    is read once even if two labels resolve to it). Empty list = abstain;
    there is deliberately no full-page fallback search.

    Consequence worth knowing: the stash is read on the UNROTATED render,
    so a page with baked-in rotation yields no confident label run and this
    pass abstains on it entirely. That is the intended trade — the rotation
    families are the escalation ladder's and the A2 probe's job, and a
    fallback search over rotated views would cost exactly the per-page time
    this design exists to avoid.
    """
    words = result.words or []
    if not words:
        return []
    gray = result.gray
    hits: dict[int, tuple[float, tuple[int, int, int, int]]] = {}
    for field in wanted:
        for pattern in TESSFT_LABELS.get(field, ()):
            n = len(pattern)
            for i in range(len(words) - n + 1):
                run = words[i:i + n]
                if any(w.h <= 0 or w.w <= 0 for w in run):
                    continue        # legacy stash without geometry
                if any(_clean_label_token(w.text) != t.lower()
                       for w, t in zip(run, pattern)):
                    continue
                if any(w.conf < _TESSFT_LOC_MIN_CONF for w in run):
                    continue
                ys = [w.y for w in run]
                if max(ys) - min(ys) > max(w.h for w in run):
                    continue        # not one row
                if not all(a.x + a.w * 0.5 < b.x < a.x + a.w + 3.0 * a.h
                           for a, b in zip(run, run[1:])):
                    continue
                y0 = int(max(0, min(ys) - _TESSFT_STRIP_Y_PAD))
                y1 = int(min(gray.shape[0],
                             max(w.y + w.h for w in run) + _TESSFT_STRIP_Y_PAD))
                x0 = int(max(0, run[0].x - 2))
                x1 = int(min(gray.shape[1], run[0].x + _TESSFT_STRIP_W))
                if y1 - y0 < 8 or x1 - x0 < 60:
                    continue
                key = (y0 + y1) // 24        # dedupe re-reads of one row
                conf = min(w.conf for w in run)
                held = hits.get(key)
                if held is None or conf > held[0]:
                    hits[key] = (conf, (y0, y1, x0, x1))
    return [np.ascontiguousarray(gray[y0:y1, x0:x1])
            for _, (y0, y1, x0, x1) in
            sorted(hits.values(), key=lambda h: -h[0])[:_TESSFT_MAX_STRIPS]]


def _clean_label_token(text: str) -> str:
    return text.strip().strip("|:;.,'\"`!‘’“”[](){}").lower()


def tessft_lines(result: "ScanOcrResult", page_index: int,
                 wanted: tuple[str, ...]) -> list[Line]:
    """Fine-tuned-model reads of one page's unread-field value strips.

    Emitted lines are ordinary OCR lines carrying scaled confidences: they
    rejoin the exact same anchoring/vocabulary/reconciliation path as every
    escalation variant and sit inside the same escalation restore guard
    (fill unread fields only; an affirmative read is never out-voted).
    Each strip keeps its label, so the lines read "Fee Status: paid" —
    exactly the shape the anchor matchers already consume.
    """
    engine = tessft_engine()
    if engine is None:
        return []
    lines: list[Line] = []
    for strip in tessft_strips(result, wanted):
        # Strips are raw crops whose glyphs can touch the edge, which
        # tesseract clips (the note-band lever measured 72.5 -> 94.3 word
        # conf from the white pad alone).
        lines.extend(_words_to_lines(engine.words(pad_for_ocr(strip)),
                                     page_index))
    return [replace(l, conf=l.conf * _TESSFT_CONF_SCALE) for l in lines]


# --- Quarantined note-band re-read (MIB_NOTE_RESCUE) ------------------------
# The adjudicator note prints its Finding/Reason rows in a fixed band at the
# top of the page. render_gray rasterizes at 288 DPI — a 2x upsample of the
# 144 DPI embedded scan — and on a note's condensed micro-text the upsample's
# ringing welds adjacent glyphs, so the Reason sentence reads ~30 points
# below the template bars (diagnosed on MIB-001000). Resampling the band back
# to native resolution and up to 1.5x restores the x-height Tesseract wants
# without reopening the PDF.
_NOTE_BAND_PT = (30, 350, 20, 560)      # top, bottom, left, right (points)


def note_band_lines(gray: np.ndarray, page_index: int,
                    engine: OcrEngine) -> list[Line]:
    """Native-resolution sparse re-read of the note header band.

    Callers MUST keep these lines quarantined: they exist solely for
    fields.note_template_finding's N1-only probe and never join page.lines
    or any other consumer.
    """
    s = RENDER_DPI / 72.0
    t, b, l, r = _NOTE_BAND_PT
    band = gray[int(t * s):int(b * s), int(l * s):int(r * s)]
    if band.size == 0:
        return []
    band = np.ascontiguousarray(band)
    native = cv2.resize(band, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    img = cv2.resize(native, None, fx=1.5, fy=1.5,
                     interpolation=cv2.INTER_CUBIC)
    # The band is a raw crop, so its rows can touch the image edge, and
    # tesseract 5.3 clips crop-edge glyphs (measured: 72.5 -> 94.3 word
    # conf on MIB-001000's Reason line with the standard white pad).
    out = _words_to_lines(engine.words(pad_for_ocr(img), sparse=True),
                          page_index)
    out.extend(_words_to_lines(engine.words(pad_for_ocr(_divblur(img)),
                                            sparse=True), page_index))
    return out


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


# --- Frame-fiducial row registration (MIB_ROWRESTORE) -----------------------
# One extra pooled ladder pass over a geometry-repaired crop, on the scan
# pages that carry a measured horizontal band translation. Every other
# ladder pass is photometric (threshold, contrast, upsample) and none of
# them can read a row whose glyphs were cut and slid sideways; row_restore
# puts the strip back on the form's own frame baseline. See row_restore.py
# for the mechanism and ATTRIBUTION.md for its source.

# Reconstructed geometry is never allowed to carry an affirmative read.
# Same cap and same reason as weld_sponsor_lines: a remap is a hypothesis
# about where the ink belonged, and adjudication must not rest on one.
# Measured on train renders (see the lever's analysis): a registered band
# on MIB-000057 p2 turned a correct `Code ORION_GRAYS` at 0.73 into
# `Code: ORIOH_GRAYS` at 0.84 — uncapped, the corruption outranks the
# truth in the pool. Capped below fields._KNOWN_MIN_OCR_CONF it can fill
# an unread field and nothing else.
_ROWRESTORE_CONF_CAP = 0.50


def row_restore_lines(gray: np.ndarray, page_index: int, engine: OcrEngine,
                      dpi: float = RENDER_DPI) -> list[Line]:
    """Read the frame-registered band, or nothing when the page is clean.

    The gate lives in row_restore.restored_band: an undamaged page pays a
    strided frame trace and returns here empty, having run no OCR at all.
    Only the displaced band is re-read, not the page — the crop keeps the
    cost proportional to the damage. Emitted lines rejoin the same
    pooling/anchoring/vocabulary path as every other ladder pass, at
    capped confidence.
    """
    reg = row_restore.register(gray, dpi)
    if reg is None:
        return []
    top = min(b.top for b in reg.repaired)
    bottom = max(b.bottom for b in reg.repaired)
    margin = max(8, int(dpi / 4))
    crop = reg.image[max(0, top - margin):min(gray.shape[0], bottom + margin)]
    if crop.size == 0:
        return []
    # Crop rows can touch the image edge, where tesseract 5.3 clips glyphs
    # (the same white-pad fix note_band_lines needed).
    img = pad_for_ocr(np.ascontiguousarray(crop))
    lines = _words_to_lines(engine.words(img), page_index)
    lines.extend(_words_to_lines(engine.words(img, sparse=True), page_index))
    return [Line(text=l.text, page_index=l.page_index, source=l.source,
                 conf=min(l.conf, _ROWRESTORE_CONF_CAP)) for l in lines]


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
    # The RAW render this page was read from, exactly as passed in. It is
    # NOT deskewed and NOT orientation-corrected: ocr_scan_page's rotation
    # and deskew work is ADDITIVE (extra OCR passes over transformed
    # copies, pooled into `lines`) and never replaces this image. Consumers
    # that do their own geometry — ctcfill's grid locator, absynth's
    # registration — must straighten it themselves. `upright` / `best_rot`
    # below report what the estimator believed, not what was applied here.
    gray: np.ndarray
    upright: bool             # False when a 90/180/270 rotation was applied
    best_rot: int = 0         # estimated np.rot90 k for non-upright pages
    # True when the upright pass did not read as a known template and the
    # orientation estimator had to guess (upright=True + True means the
    # estimator scored k=0 highest). Metadata only — no ladder behavior
    # depends on it; the A2 rotation probe uses it as its trigger.
    orientation_estimated: bool = False
    # Word boxes pooled from the ladder passes whose pixel space equals
    # `gray` (raw/otsu/adaptive/stretch/sparse). Zero extra OCR cost — the
    # passes already run; only their geometry was previously discarded.
    # Deskewed/rotated/band-crop passes are excluded: their coordinates
    # do not live in `gray`'s space. Consumed by the MIB_CTCFILL locator.
    words: list[OcrWord] = None  # set in ocr_scan_page; None keeps old ctors


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
    # Geometry stash for MIB_CTCFILL: words from every pass that reads
    # `gray`'s own pixel space (raw + the thresholded variants + sparse).
    word_stash = list(raw_words)

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
        words = engine.words(img)
        word_stash.extend(words)
        pooled.extend(_words_to_lines(words, page_index))
    sparse_words = engine.words(gray, sparse=True)
    word_stash.extend(sparse_words)
    pooled.extend(_words_to_lines(sparse_words, page_index))

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

    # Free-form re-read of the registered band (MIB_ROWRESTORE_FREEFORM=1,
    # default OFF and MEASURED NO-SHIP — see row_restore.FREEFORM_DEFAULT.
    # MIB_ROWRESTORE alone does NOT reach here: the shipped channel reads
    # the registered page with the closed-menu CTC scorer instead, so the
    # arm never pays this block's ~204 ms per triggered page).
    # Upright pages only: the fiducial is the form's LEFT frame rule, which
    # is only on the left when the page is the right way up, and the
    # counter-rotated views are already pooled above. Additive like every
    # other pass — the restored crop's lines join `pooled` and win a line
    # only by out-confidencing it, never by replacing a read.
    # Word geometry is deliberately NOT stashed: the crop's coordinates do
    # not live in `gray`'s pixel space (same exclusion as the band passes).
    if row_restore.freeform_enabled() and upright:
        pooled.extend(row_restore_lines(gray, page_index, engine))

    best: dict[str, Line] = {}
    for line in pooled:
        key = " ".join(line.text.lower().split())
        if key not in best or line.conf > best[key].conf:
            best[key] = line
    return ScanOcrResult(lines=list(best.values()), gray=gray, upright=upright,
                         best_rot=0 if upright else best_rot,
                         orientation_estimated=orientation_estimated,
                         words=word_stash)
