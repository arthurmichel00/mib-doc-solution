"""Closed-vocab fee-value shape classifier (spec §9, D-FEE reading extension).

The fee receipt template prints its three value fields in 9 pt regular
Helvetica at fixed positions (dev-time span probe over digital receipts,
byte-identical geometry, spec §9.1). Each value is ONE word from a closed
vocabulary, so damaged rows that defeat character OCR ("naird" for "paid")
can still be read holistically: correlate the whole value region against
exact-font templates rendered by pymupdf — the same base-14 Helvetica
metrics that rendered the corpus.

Abstention-first: a read is returned only when the best template clears an
absolute score bar AND a margin over the runner-up, and survives two hard
vetoes that close the substring hazard ("unpaid" contains "paid" as a
sub-image, the same failure family as fields._PHRASE_MIN_COVER): the crop's
ink width must match the template's, and the aligned template must explain
most of the crop's ink mass. Thresholds are tuned on synthetic degradations
only (JPEG q~58, blur, skew — tools-level sweep; spec §9.2); gold labels are
never used for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

STATUS_VOCAB = ("paid", "waived", "unpaid", "unknown")
AMOUNT_VOCAB = ("$809.00", "$0.00")
WAIVER_VOCAB = ("N/A", "DIP-WAIVER")

# Value-row regions in template points (y0, y1, x0, x1), padded ±6 pt for
# the corpus skew band (±0.8 degrees). Geometry provenance: spec §9.1.
REGIONS_PT = {
    "fee_status": (142, 163, 232, 345),
    "fee_amount": (166, 187, 232, 350),
    "waiver_code": (190, 211, 232, 360),
}
_PAGE_PT = (612.0, 792.0)
_VALUE_BASELINE_PT = 157.0   # any row works; templates are position-trimmed
_VALUE_X_PT = 238.0
_VALUE_FONT_SIZE = 9.0
_TEMPLATE_DPI = 576

# Ink extraction and vetoes.
_INK_STRONG = 60             # strong-ink threshold after background removal
_MIN_INK_PIXELS = 60         # fewer strong pixels than this = blank region
_HEIGHT_ABS_PX = (18, 160)   # plausible ink heights at ~576 DPI
_HEIGHT_REL = (0.45, 2.2)    # crop ink height vs template ink height
_WIDTH_REL = (0.70, 1.38)    # crop ink width vs scaled template width
_RESIDUAL_MAX = 0.25         # ink mass the aligned template may leave over
_SCALE_JITTER = (0.90, 1.00, 1.10)
_BLUR_SIGMAS = (0.0, 2.0, 4.0)
# Per-glyph cell verification. Holistic correlation cannot separate
# near-glyph forgeries or lookalike words ("$500.00" vs "$809.00", "$80.00"
# swallowing a slid "$0.00", blurred "void" vs "paid"), and no absolute
# NCC bar can either (a clean forged '5' correlates 0.66 against the '8'
# template while a degraded genuine '8' can score less). Each wide-enough
# glyph cell must instead CLASSIFY as its expected glyph against a candidate
# alphabet — rank-based, so degradation that lowers every candidate equally
# does not break genuine reads. An expected glyph over empty paper fails
# outright (catches a template hanging past the crop's ink, e.g. "$809.00"
# matched onto a $-less "809.00").
# Amounts (the decisive corroborator, forgeable by digit swaps): expected
# glyph must be STRICT argmax over digits + '$'. Status words (real corpus
# damage mangles letter strokes): expected letter must rank in the TOP-2
# over the status vocabulary's own alphabet — one notch of damage tolerance,
# zero failed cells allowed.
_GLYPH_MIN_WIDTH_PX = 12
_GLYPH_BLANK_FRAC = 0.08    # cell ink below this fraction of template = paper
_DIGITS = "0123456789"
_AMOUNT_GLYPH_SET = _DIGITS + "$"
_STATUS_GLYPH_SET = "".join(sorted(set("".join(STATUS_VOCAB))))

# Acceptance bars — validated on the synthetic degradation sweep
# (tools/tune_fee_shape.py, 2026-07-29): 0 wrong accepts across the full
# grid (596 adversarial probes abstained: cross-vocabulary words, digit
# forgeries, $-less amounts, junk, noise, blank), genuine accepts 350/384
# with observed correct-accept score minimum 0.587 — the bars sit a safety
# notch below the genuine frontier and well above every probe survivor.
SCORE_MIN = 0.55
MARGIN_MIN = 0.10


@dataclass(frozen=True)
class ShapeRead:
    value: str
    score: float
    margin: float


def crop_regions(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Cut the three value-row regions out of a full-page rasterization.

    Scale is derived from the image itself so any render DPI works; the
    caller is responsible for the page being upright (the discharge head
    rotates by the scan's estimated orientation before calling).
    """
    sy = gray.shape[0] / _PAGE_PT[1]
    sx = gray.shape[1] / _PAGE_PT[0]
    return {
        name: np.ascontiguousarray(
            gray[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)])
        for name, (y0, y1, x0, x1) in REGIONS_PT.items()
    }


@lru_cache(maxsize=None)
def _template(word: str) -> np.ndarray:
    """Ink image of one vocabulary word, rendered like the corpus prints it."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=_PAGE_PT[0], height=_PAGE_PT[1])
    page.insert_text((_VALUE_X_PT, _VALUE_BASELINE_PT), word,
                     fontsize=_VALUE_FONT_SIZE, fontname="helv")
    pix = page.get_pixmap(dpi=_TEMPLATE_DPI, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width).copy()
    doc.close()
    ink = (255 - gray).astype(np.uint8)
    ys, xs = np.nonzero(ink > _INK_STRONG)
    return np.ascontiguousarray(
        ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


@lru_cache(maxsize=None)
def _glyph_bounds_px(word: str) -> tuple[float, ...]:
    """Per-glyph x boundaries inside the trimmed template, in template px.

    Helvetica advance widths are exact (same metrics the corpus renderer
    used), so glyph cells need no ink segmentation: pen offsets are mapped
    into the trimmed template by the '$'-edge trim offset.
    """
    import pymupdf

    font = pymupdf.Font("helv")
    scale = _TEMPLATE_DPI / 72.0
    pen = [font.text_length(word[:i], fontsize=_VALUE_FONT_SIZE) * scale
           for i in range(len(word) + 1)]
    # trim offset: ink starts at the first glyph's left side bearing; the
    # template was trimmed to its ink bbox, so shift pen offsets to that box
    doc = pymupdf.open()
    page = doc.new_page(width=_PAGE_PT[0], height=_PAGE_PT[1])
    page.insert_text((_VALUE_X_PT, _VALUE_BASELINE_PT), word,
                     fontsize=_VALUE_FONT_SIZE, fontname="helv")
    pix = page.get_pixmap(dpi=_TEMPLATE_DPI, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width).copy()
    doc.close()
    xs = np.nonzero((255 - gray).max(axis=0) > _INK_STRONG)[0]
    trim = float(xs.min()) - _VALUE_X_PT * scale
    return tuple(p - trim for p in pen)


def _glyphs_verify(window: np.ndarray, word: str, scale: float,
                   sigma: float, alphabet: str, top_k: int) -> bool:
    """Every wide-enough glyph cell must rank its expected glyph in the
    alphabet's top-k; a checked cell over empty paper fails outright."""
    bounds = [b * scale for b in _glyph_bounds_px(word)]
    for ch, left, right in zip(word, bounds, bounds[1:]):
        if ch not in alphabet:
            continue
        a = max(0, int(left) - 3)
        b = min(window.shape[1], int(right) + 3)
        if b - a < _GLYPH_MIN_WIDTH_PX:
            continue
        cell = np.ascontiguousarray(window[:, a:b])
        expected = cv2.resize(_template(ch), None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        if float((cell > _INK_STRONG).sum()) < \
                _GLYPH_BLANK_FRAC * float((expected > _INK_STRONG).sum()):
            return False
        scores: dict[str, float] = {}
        for cand in alphabet:
            tmpl = cv2.resize(_template(cand), None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA).astype(np.float32)
            if sigma:
                tmpl = cv2.GaussianBlur(tmpl, (0, 0), sigma)
            if tmpl.shape[0] > cell.shape[0] or tmpl.shape[1] > cell.shape[1]:
                continue
            res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
            scores[cand] = float(res.max())
        if ch not in scores:
            return False
        rank = sum(1 for v in scores.values() if v > scores[ch])
        if rank >= top_k:
            return False
    return True


def _crop_ink(crop: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """(continuous ink image, strong-ink bbox) or None for a blank region."""
    if crop.size == 0:
        return None
    bg = float(np.percentile(crop, 80))
    ink = np.clip(bg - crop.astype(np.float32), 0, 255)
    strong = ink > _INK_STRONG
    if int(strong.sum()) < _MIN_INK_PIXELS:
        return None
    ys, xs = np.nonzero(strong)
    return ink, (int(ys.min()), int(ys.max()) + 1,
                 int(xs.min()), int(xs.max()) + 1)


def classify(crop: np.ndarray, vocab: tuple[str, ...],
             glyphs: tuple[str, int] | None = None) -> ShapeRead | None:
    extracted = _crop_ink(crop)
    if extracted is None:
        return None
    ink, (y0, y1, x0, x1) = extracted
    h_c, w_c = y1 - y0, x1 - x0
    if not _HEIGHT_ABS_PX[0] <= h_c <= _HEIGHT_ABS_PX[1]:
        return None
    strong = ink > _INK_STRONG
    total_ink = float(ink[strong].sum())

    scores: dict[str, float] = {}
    for word in vocab:
        base = _template(word)
        h_t = base.shape[0]
        if not _HEIGHT_REL[0] <= h_c / h_t <= _HEIGHT_REL[1]:
            continue
        best = 0.0
        for jitter in _SCALE_JITTER:
            s = h_c / h_t * jitter
            tmpl = cv2.resize(base, None, fx=s, fy=s,
                              interpolation=cv2.INTER_AREA).astype(np.float32)
            th, tw = tmpl.shape
            if th > ink.shape[0] or tw > ink.shape[1]:
                continue
            if not _WIDTH_REL[0] <= w_c / tw <= _WIDTH_REL[1]:
                continue
            for sigma in _BLUR_SIGMAS:
                probe = cv2.GaussianBlur(tmpl, (0, 0), sigma) if sigma else tmpl
                res = cv2.matchTemplate(ink, probe, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val <= best:
                    continue
                # Residual-ink veto: the aligned template must explain the
                # crop's ink; a "paid" window over the "-paid" suffix of
                # "unpaid" leaves the "un" ink outside and is rejected.
                mx0, mx1 = loc[0], loc[0] + tw
                outside = float(ink[:, :mx0][strong[:, :mx0]].sum()
                                + ink[:, mx1:][strong[:, mx1:]].sum())
                if total_ink > 0 and outside / total_ink > _RESIDUAL_MAX:
                    continue
                if glyphs is not None:
                    window = np.ascontiguousarray(
                        ink[loc[1]:loc[1] + th, mx0:mx1])
                    if not _glyphs_verify(window, word, s, sigma, *glyphs):
                        continue
                best = float(val)
        if best > 0.0:
            scores[word] = best
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    value, score = ranked[0]
    margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if score < SCORE_MIN or margin < MARGIN_MIN:
        return None
    return ShapeRead(value=value, score=score, margin=margin)


def classify_status(crop: np.ndarray) -> ShapeRead | None:
    return classify(crop, STATUS_VOCAB, glyphs=(_STATUS_GLYPH_SET, 2))


def classify_amount(crop: np.ndarray) -> ShapeRead | None:
    return classify(crop, AMOUNT_VOCAB, glyphs=(_AMOUNT_GLYPH_SET, 1))


def classify_waiver(crop: np.ndarray) -> ShapeRead | None:
    return classify(crop, WAIVER_VOCAB)
