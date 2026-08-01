"""Hidden-text detection.

Splits every text span into visible (candidate evidence) vs hidden
(injection carrier). Hidden text is never evidence, for any purpose.
Spans are classified straight from the appearance stream via
page.get_texttrace(), and hidden spans are physically redacted from a
working copy of the page BEFORE any rasterization so that no image
preprocessing step can resurrect them.
"""
from __future__ import annotations

import pymupdf

from .model import TextSpan

NEAR_WHITE = 0.92     # min channel value that counts as white-on-paper
TINY_FONT_PT = 6.0    # injections observed at 5 pt; real footers are 7 pt


def _span_from_trace(trace: dict) -> TextSpan:
    return TextSpan(
        text="".join(chr(c[0]) for c in trace["chars"]),
        bbox=tuple(trace["bbox"]),
        size=float(trace["size"]),
        color=tuple(trace["color"]) if trace.get("color") else None,
        opacity=float(trace["opacity"]),
        render_type=int(trace["type"]),
        font=str(trace.get("font", "")),
    )


def hidden_reasons(span: TextSpan, page_rect: pymupdf.Rect) -> list[str]:
    reasons = []
    if span.render_type == 3:
        reasons.append("invisible_render_mode")
    if span.opacity == 0:
        reasons.append("zero_opacity")
    if span.size < TINY_FONT_PT:
        reasons.append("tiny_font")
    if span.color is not None and span.color and min(span.color) >= NEAR_WHITE:
        reasons.append("near_white_fill")
    if not pymupdf.Rect(span.bbox).intersects(page_rect):
        reasons.append("outside_cropbox")
    return reasons


def classify_spans(page: pymupdf.Page) -> tuple[list[TextSpan], list[TextSpan]]:
    """Return (visible_spans, hidden_spans) for one page."""
    visible: list[TextSpan] = []
    hidden: list[TextSpan] = []
    for trace in page.get_texttrace():
        span = _span_from_trace(trace)
        if not span.text.strip():
            continue
        if hidden_reasons(span, page.rect):
            hidden.append(span)
        else:
            visible.append(span)
    return visible, hidden


def redact_hidden_text(page: pymupdf.Page, hidden: list[TextSpan]) -> None:
    """Remove hidden spans from the page content before rendering.

    Uses redaction annotations restricted to text only, so the underlying
    scan image and vector graphics are untouched.
    """
    if not hidden:
        return
    for span in hidden:
        rect = pymupdf.Rect(span.bbox) & page.mediabox
        if not rect.is_empty:
            page.add_redact_annot(rect)
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
    )
