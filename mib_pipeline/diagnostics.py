"""Diagnostic-only green-APPROVED-stamp counter (A7; approved 2026-07-29).

Detect-and-log ONLY. Nothing in this module feeds decisions, fields,
confidences, or emitted row content: the single entry point returns None,
mutates no Page/CaseEvidence state, and every step is wrapped so a failure
here can never fail a case. Output goes to stderr and to a diagnostics
file in the working directory — never into the prediction rows.

Detector provenance (research/17-a7-stamp-verify.md): the adjudicator's
green APPROVED stamp is pure-green ink (OpenCV hue 60, sat 253-255) drawn
only on digital Manual Adjudicator Note pages, as two hollow components —
the rotated box outline (~227x97 px @100dpi, fill ~0.08) and the word
(~154x47 px, fill ~0.24). On the 1000-case train corpus the detector below
flags exactly 33 cases, all gold APPROVED (precision 1.000), zero hits on
the red-team corpus. Cost control: pages are RGB-rendered only when the
existing classification says digital note page — the only page type the
survey ever found carrying the stamp.
"""
from __future__ import annotations

import atexit
import json
import os
import sys

import cv2
import numpy as np
import pymupdf

from . import fields
from .model import Page, PageKind

_RENDER_DPI = 100
_HUE_LO, _HUE_HI = 50, 70    # digital ink is exactly 60
_SAT_MIN = 150               # observed 253-255; nearest clutter <= 66
_VAL_LO, _VAL_HI = 60, 200   # observed 128
_MIN_COMPONENT_PX = 400      # observed components 1668-1842 px
_ASPECT_LO, _ASPECT_HI = 1.4, 3.6   # observed 2.34 (box) / 3.28 (word)
_FILL_LO, _FILL_HI = 0.04, 0.40     # hollow strokes; observed 0.08 / 0.24
_MIN_WIDTH_PX = 90           # observed 154-227

_DIAG_FILE_ENV = "MIB_STAMP_DIAG_FILE"
_DIAG_FILE_DEFAULT = "stamp_diagnostics.log"

# Review-sweep page cap, mirroring pdf_loader._MAX_PAGES: bounds the render
# work a pathological many-page input could demand from the diagnostic.
_REVIEW_MAX_PAGES = 20

# Per-process tallies (workers each report their own subtotal at exit).
_counts = {"cases": 0, "note_pages": 0, "review_cases": 0,
           "review_pages": 0, "stamped_cases": 0}
_exit_registered = False


def _diag_path() -> str:
    return os.environ.get(_DIAG_FILE_ENV, _DIAG_FILE_DEFAULT)


def _emit(text: str) -> None:
    """Stderr always; diagnostics file best-effort (cwd may be read-only)."""
    print(text, file=sys.stderr, flush=True)
    try:
        with open(_diag_path(), "a") as f:
            f.write(text + "\n")
    except OSError:
        pass


def _log_exit_total() -> None:
    if not _counts["cases"]:      # nothing processed (e.g. patched-out tests)
        return
    _emit("[stamp-diag] exit subtotal: cases=%d note_pages_rendered=%d "
          "review_cases=%d review_pages_rendered=%d stamped_cases=%d (pid %d)"
          % (_counts["cases"], _counts["note_pages"], _counts["review_cases"],
             _counts["review_pages"], _counts["stamped_cases"], os.getpid()))


def stamp_components(img_rgb: np.ndarray) -> list[dict]:
    """Green-stamp components on one rendered RGB page (color-region only)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((h >= _HUE_LO) & (h <= _HUE_HI) & (s >= _SAT_MIN)
            & (v >= _VAL_LO) & (v <= _VAL_HI)).astype(np.uint8)
    if int(mask.sum()) < _MIN_COMPONENT_PX:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    hits = []
    for i in range(1, n):
        x, y, w, hh, _area = stats[i]
        npx = int(((labels == i) & (mask > 0)).sum())
        if npx < _MIN_COMPONENT_PX or w < _MIN_WIDTH_PX:
            continue
        aspect, fill = w / hh, npx / (w * hh)
        if _ASPECT_LO <= aspect <= _ASPECT_HI and _FILL_LO <= fill <= _FILL_HI:
            hits.append({"bbox": [int(x), int(y), int(w), int(hh)],
                         "px": npx, "aspect": round(aspect, 2),
                         "fill": round(fill, 3)})
    return hits


# The note header exactly as the classifier's header table spells it.
# Digital pages carry pristine vector text, so the exact phrase suffices:
# verified equivalent to fields.detect_doc_type()=="note" on all 2203
# digital pages of the train corpus (162 = 162, 0 disagreements), at
# microseconds instead of the fuzzy matcher's ~10-30 ms per page.
_NOTE_HEADER = next(
    header for header, (doc_type, _tier) in fields._DOC_HEADERS.items()
    if doc_type == "note")


def _digital_note_indexes(pages: list[Page]) -> list[int]:
    """Digital pages typed as adjudicator note (pure read; sets nothing)."""
    return [page.index for page in pages
            if page.kind == PageKind.DIGITAL
            and any(_NOTE_HEADER in line.text for line in page.lines[:10])]


def _render_rgb(doc: pymupdf.Document, index: int) -> np.ndarray:
    pix = doc[index].get_pixmap(dpi=_RENDER_DPI, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3)


def _log_detection(case_id: str, source: str,
                   stamped_pages: dict[int, int]) -> None:
    _counts["stamped_cases"] += 1
    _emit("[stamp-diag] %s green-APPROVED stamp detected: %s"
          % (case_id,
             json.dumps({"source": source, "pages": sorted(stamped_pages),
                         "components": sum(stamped_pages.values())},
                        sort_keys=True)))


def _register_exit() -> None:
    global _exit_registered
    if not _exit_registered:
        atexit.register(_log_exit_total)
        _exit_registered = True


def log_stamp_scan(doc: pymupdf.Document, pages: list[Page],
                   case_id: str) -> None:
    """Count/log green APPROVED stamps for one case. Diagnostic only.

    Renders in RGB only the digital note pages (the sole carrier type in
    the corpus survey); a case without one costs a few dict lookups.
    Never raises: any error is logged and swallowed.
    """
    try:
        _register_exit()
        _counts["cases"] += 1
        stamped_pages: dict[int, int] = {}
        for index in _digital_note_indexes(pages):
            _counts["note_pages"] += 1
            hits = stamp_components(_render_rgb(doc, index))
            if hits:
                stamped_pages[index] = len(hits)
        if stamped_pages:
            _log_detection(case_id, "note_page", stamped_pages)
    except TimeoutError:
        raise  # the case deadline (pipeline._deadline) must never be swallowed
    except Exception as exc:
        try:
            _emit("[stamp-diag] %s scan error (ignored): %r" % (case_id, exc))
        except Exception:
            pass


def log_stamp_scan_review(pdf_path: str, case_id: str) -> dict[int, int]:
    """Post-decision sweep for NEEDS_REVIEW cases: every page, in color.

    The corpus's scan-page images are /DeviceRGB 8-bit JPEGs (verified on
    all 1,956 train scan pages), so green ink CAN survive scanning — a
    scanned adjudicator note would be missed by the digital-header gate
    above. NEEDS_REVIEW is the only population where a missed stamp is
    informative, so only those cases pay for the full-color sweep. The
    in-pipeline document is closed by decision time; the PDF is re-opened
    read-only.

    Returns {page_index: component_count} for detected stamps ({} on any
    error — fail-closed for the MIB_STAMP_RESCUE consumer). The return
    value is purely informational for the default-off rescue flag; with
    the flag unset nothing reads it and the sweep stays diagnostic-only.
    """
    stamped_pages: dict[int, int] = {}
    try:
        _register_exit()
        _counts["review_cases"] += 1
        with pymupdf.open(pdf_path) as doc:
            for index in range(min(doc.page_count, _REVIEW_MAX_PAGES)):
                _counts["review_pages"] += 1
                hits = stamp_components(_render_rgb(doc, index))
                if hits:
                    stamped_pages[index] = len(hits)
        if stamped_pages:
            _log_detection(case_id, "review_scan", stamped_pages)
    except TimeoutError:
        raise  # the case deadline (pipeline._deadline) must never be swallowed
    except Exception as exc:
        try:
            _emit("[stamp-diag] %s review-scan error (ignored): %r"
                  % (case_id, exc))
        except Exception:
            pass
        return {}
    return stamped_pages


def log_rescue(case_id: str, stamped_pages: dict[int, int]) -> None:
    """Audit line for a MIB_STAMP_RESCUE promotion. Never raises."""
    try:
        _emit("[stamp-diag] %s rescued NEEDS_REVIEW->APPROVED "
              "(S1_stamp_approved): %s"
              % (case_id,
                 json.dumps({"pages": sorted(stamped_pages),
                             "components": sum(stamped_pages.values())},
                            sort_keys=True)))
    except Exception:
        pass
