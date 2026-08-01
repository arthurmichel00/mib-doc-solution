"""MIB Doc Challenge extraction + adjudication pipeline.

Visible-evidence-only document pipeline: hidden text spans are filtered
before any extraction or rasterization, fields are harvested from digital
text layers and OCR of scanned pages, reconciled by the FIELD_MANUAL
evidence precedence, and adjudicated by a deterministic policy engine with
an expected-value decision layer on top.
"""

__version__ = "0.1.0"
