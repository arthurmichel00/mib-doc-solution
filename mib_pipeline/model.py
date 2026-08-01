"""Shared datatypes for the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PageKind(Enum):
    DIGITAL = "digital"
    SCAN = "scan"


class Source(Enum):
    DIGITAL = "digital"   # visible vector text span
    OCR = "ocr"           # recognized from rendered pixels


@dataclass(frozen=True)
class TextSpan:
    """One text span from the PDF appearance stream."""
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    color: tuple[float, ...] | None
    opacity: float
    render_type: int
    font: str

    @property
    def italic(self) -> bool:
        return "oblique" in self.font.lower() or "italic" in self.font.lower()


@dataclass(frozen=True)
class Line:
    """One reading-order text line, from either text layer or OCR."""
    text: str
    page_index: int
    source: Source
    conf: float           # 0..1; digital lines are ~certain
    italic: bool = False
    # True when the line is printed in near-black ink. OCR lines are dark by
    # construction (recognized from rendered ink); digital lines carry their
    # spans' fill color. Light-grey text that survives the hidden-span filter
    # is still never trusted for tier-1 evidence.
    dark: bool = True
    # False for sub-trusted engines (the candidate-trained CRNN): their reads
    # may anchor ordinary field candidates but can never mint a Finding or
    # stamp — tier-1 evidence requires a proven-precision reader.
    tier1_ok: bool = True


@dataclass
class Page:
    index: int
    kind: PageKind
    visible_spans: list[TextSpan] = field(default_factory=list)
    hidden_spans: list[TextSpan] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    doc_type: str | None = None


@dataclass(frozen=True)
class FieldCandidate:
    """One observed value for one schema field, with provenance."""
    fld: str
    raw: str              # value text as read
    value: str | None     # vocabulary/pattern-corrected value, None if rejected
    tier: float           # FIELD_MANUAL evidence precedence, lower = stronger
    page_index: int
    source: Source
    conf: float
