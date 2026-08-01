"""PDF loading, page typing, and digital text-line assembly."""
from __future__ import annotations

import re

import pymupdf

from .model import Line, Page, PageKind, Source, TextSpan
from .visibility import classify_spans

# A page is a "scan" when a raster image covers most of it; the only
# trustworthy digital text on such a page is the generator's footer.
_SCAN_IMAGE_COVERAGE = 0.5
_FOOTER_RE = re.compile(
    r"^(Packet MIB-[0-9]{6} / page \d+|Synthetic hiring challenge document)$"
)
_LINE_Y_TOLERANCE = 4.0  # pt
# Observed packets have 3-6 pages; the cap only bounds pathological inputs.
_MAX_PAGES = 20


def _page_kind(page: pymupdf.Page) -> PageKind:
    page_area = abs(page.rect)
    if not page_area:
        return PageKind.DIGITAL
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            if abs(rect) / page_area >= _SCAN_IMAGE_COVERAGE:
                return PageKind.SCAN
    return PageKind.DIGITAL


def assemble_lines(spans: list[TextSpan], page_index: int) -> list[Line]:
    """Group spans that share a baseline into reading-order lines."""
    def y_center(s: TextSpan) -> float:
        return (s.bbox[1] + s.bbox[3]) / 2

    def is_dark(span: TextSpan) -> bool:
        return span.color is None or not span.color or min(span.color) < 0.45

    rows: list[list[TextSpan]] = []
    for span in sorted(spans, key=lambda s: (y_center(s), s.bbox[0])):
        if rows and abs(y_center(rows[-1][0]) - y_center(span)) <= _LINE_Y_TOLERANCE:
            rows[-1].append(span)
        else:
            rows.append([span])
    lines = []
    for row in rows:
        row.sort(key=lambda s: s.bbox[0])
        text = " ".join(s.text.strip() for s in row if s.text.strip())
        if text:
            lines.append(Line(
                text=text,
                page_index=page_index,
                source=Source.DIGITAL,
                conf=0.99,
                italic=any(s.italic for s in row),
                dark=all(is_dark(s) for s in row if s.text.strip()),
            ))
    return lines


def load_pages(doc: pymupdf.Document) -> list[Page]:
    """Classify spans and pages; assemble trusted digital lines.

    On scan pages, only the footer survives as digital text: any other
    text-layer content on a scanned page cannot be seen by a human reading
    the page image and is therefore untrusted by construction.
    """
    pages: list[Page] = []
    for index, pdf_page in enumerate(doc):
        if index >= _MAX_PAGES:
            break
        visible, hidden = classify_spans(pdf_page)
        kind = _page_kind(pdf_page)
        if kind == PageKind.SCAN:
            trusted = [s for s in visible if _FOOTER_RE.match(s.text.strip())]
            untrusted = [s for s in visible if not _FOOTER_RE.match(s.text.strip())]
            hidden = hidden + untrusted
            visible = trusted
        page = Page(index=index, kind=kind, visible_spans=visible, hidden_spans=hidden)
        if kind == PageKind.DIGITAL:
            page.lines = assemble_lines(visible, index)
        pages.append(page)
    return pages
