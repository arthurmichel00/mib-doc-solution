"""Structural review-discharge heads (MIB_DISCHARGE=1, default OFF).

Spec: docs/superpowers/specs/2026-07-29-structural-review-discharge-design.md
(§2 heads, §4 CFA firewall, §8 binding acceptance bars). A head never writes
an adjudication: when the policy left a case on one of four under-determined
review paths, the head re-reads that path's ONE blocking evidence item from a
fresh 576-DPI rasterization of the packet's own evidence page (template-band
crop, hidden text redacted before rendering) and upgrades the field to
affirmatively-known only when independent preprocessing views agree on the
parsed value. The unchanged policy cascade then re-runs — approval still
requires every R12 gate to pass, which is what keeps a head misread from ever
reaching a catastrophic false approval on its own.

Head roster (kills are FINAL per the spec): D-FEE on R8_fee_unread, D-ARRIVAL
on R9_arrival_not_visible, D-SOLEGAP on R12_sponsor_unread/R12_visa_unread.
No head exists for R12_flags_unread (the silent-flag trap family lives there)
or R12_world_unread (empty population). Affirmative review paths (N0/N1
notes, printed fee "unknown", R10 review flags) are never triggers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import fields, policy, vocab
from .fields import CaseEvidence
from .model import Page, PageKind

# Discharged rows keep the resulting path's calibrated confidence, capped:
# the evidence was re-read after every primary engine failed on it, so it
# never deserves the path's full ceiling (in-container refit re-estimates).
CONF_CAP = 0.90

_BAND_DPI = 576                      # fresh rasterization, not a resample
_BAND_PT = (30, 350, 40, 500)        # y0, y1, x0, x1 — the template field
#                                      band every scan template prints
#                                      (same coordinates the base ladder
#                                      crops at 288 DPI, ocr.py)
_ALLOWED_FEE_VERDICTS = ("paid", "waived", "unpaid")
_UNREADABLE_MAX_DIST = 2.5           # printed-UNREADABLE marker tolerance

# path -> (head, field). The absence of R12_flags_unread/R12_world_unread
# here IS the D-FLAGS/H3 kill; do not add entries without a new spec ruling.
_TRIGGERS: dict[str, tuple[str, str]] = {
    "R8_fee_unread": ("fee", "fee_status"),
    "R9_arrival_not_visible": ("arrival", "arrival_on_intake"),
    "R12_sponsor_unread": ("solegap", "sponsor_id"),
    "R12_visa_unread": ("solegap", "visa_class"),
}

_HEADS = ("fee", "arrival", "solegap")


@dataclass(frozen=True)
class ShapeVerdict:
    """Fee verdict from the closed-vocab shape classifier (spec §9.4)."""
    value: str
    conf: float
    views: tuple[str, ...]


@dataclass(frozen=True)
class Discharge:
    head: str
    field: str
    value: str
    page_index: int
    views: tuple[str, ...]
    conf: float

    def as_dict(self) -> dict:
        return {"head": self.head, "field": self.field, "value": self.value,
                "page": self.page_index, "views": list(self.views),
                "conf": round(self.conf, 4)}


def _master_on() -> bool:
    return os.environ.get("MIB_DISCHARGE") == "1"


def head_enabled(head: str) -> bool:
    return _master_on() and \
        os.environ.get(f"MIB_DISCHARGE_{head.upper()}", "1") == "1"


def any_enabled() -> bool:
    return any(head_enabled(h) for h in _HEADS)


# ------------------------------------------------------------ rasterization


def _band_views(pdf_path: str, page: Page, scan, budget_left=None):
    """Named preprocessing views of the page's template field band.

    This is the module's only pixel source and the tests' monkeypatch seam.
    Rendering goes through ocr.render_gray, so hidden text is redacted before
    any pixel exists — the muzzle-at-source posture is preserved. Views in
    the same pixel family share a name prefix ("raw"/"raw-sparse"): head
    acceptance requires agreement across DISTINCT families, never across two
    reads of the same pixels.
    """
    import pymupdf

    from . import ocr

    try:
        with pymupdf.open(pdf_path) as doc:
            gray = ocr.render_gray(doc[page.index], page, dpi=_BAND_DPI)
    except Exception:
        return []
    if not scan.upright and scan.best_rot:
        gray = np.ascontiguousarray(np.rot90(gray, k=scan.best_rot))
    scale = _BAND_DPI / 72.0
    y0, y1, x0, x1 = (int(v * scale) for v in _BAND_PT)
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return []
    band = ocr.pad_for_ocr(np.ascontiguousarray(band))
    views = [("raw", band, False), ("raw-sparse", band, True)]
    for name, fn in (("otsu", ocr._otsu), ("divblur", ocr._divblur)):
        try:
            views.append((name, fn(band), False))
        except Exception:
            continue
    # PP-OCR second-engine views (reader tag instead of a sparse flag).
    # Both share the single "ppocr" family: the same recognizer prior can
    # repeat a misread across preprocessings, so PP-OCR may corroborate a
    # Tesseract read but never fire a head on its own agreement.
    views.append(("ppocr-raw", band, "ppocr"))
    try:
        views.append(("ppocr-divblur", ocr._divblur(band), "ppocr"))
    except Exception:
        pass
    return views


def _family(view_name: str) -> str:
    return view_name.split("-", 1)[0]


def _feeshape_enabled() -> bool:
    return os.environ.get("MIB_DISCHARGE_FEESHAPE", "1") == "1"


def _shape_views(pdf_path: str, page: Page, scan):
    """Named pixel-variant crop sets for the shape classifier.

    Same render/rotation/muzzle path as _band_views; divblur is applied to
    the region crops (its 21 px background estimate is local). Tests'
    monkeypatch seam.
    """
    import pymupdf

    from . import fee_shape, ocr

    try:
        with pymupdf.open(pdf_path) as doc:
            gray = ocr.render_gray(doc[page.index], page, dpi=_BAND_DPI)
    except Exception:
        return []
    if not scan.upright and scan.best_rot:
        gray = np.ascontiguousarray(np.rot90(gray, k=scan.best_rot))
    raw = fee_shape.crop_regions(gray)
    out = [("shape-raw", raw)]
    try:
        out.append(("shape-divblur",
                    {name: ocr._divblur(crop) for name, crop in raw.items()}))
    except Exception:
        pass
    return out


def _shape_verdict(pdf_path: str, page: Page, scan) -> ShapeVerdict | None:
    """Fee verdict from whole-word shape reads, or None (abstain).

    Fires only when EVERY pixel variant classifies BOTH status and amount to
    margins and all variants agree on one _receipt_verdict outcome — amount
    corroboration is structural (the amount vocabulary is {809, 0} only, so
    a forged $500.00 abstains inside the classifier).
    """
    from . import fee_shape

    per_variant: list[tuple[str, str, float]] = []
    for name, crops in _shape_views(pdf_path, page, scan):
        status = fee_shape.classify_status(crops["fee_status"])
        amount = fee_shape.classify_amount(crops["fee_amount"])
        if status is None or amount is None:
            return None
        waiver = fee_shape.classify_waiver(crops["waiver_code"])
        verdict = fields._receipt_verdict(
            status.value, 809 if amount.value == "$809.00" else 0,
            waiver.value if waiver else None)
        if verdict not in _ALLOWED_FEE_VERDICTS:
            return None
        per_variant.append((name, verdict, min(status.score, amount.score)))
    if len(per_variant) < 2 or len({v for _, v, _ in per_variant}) != 1:
        return None
    return ShapeVerdict(value=per_variant[0][1],
                        conf=min(c for _, _, c in per_variant),
                        views=tuple(name for name, _, _ in per_variant))


def _view_lines(engine, image, sparse, page_index: int):
    from . import ocr

    if sparse == "ppocr":
        return ocr.rapid_lines(image, page_index)
    return ocr._words_to_lines(engine.words(image, sparse=sparse), page_index)


# ------------------------------------------------------------ page selection


def _candidate_pages(pages: list[Page], scans: dict, case_id: str,
                     doc_types: tuple[str, ...]) -> list[Page]:
    """Scan pages of the wanted template type that belong to THIS case."""
    return [
        p for p in pages
        if p.kind == PageKind.SCAN and p.doc_type in doc_types
        and p.index in scans and not fields._foreign_page(p, case_id)
    ]


# ------------------------------------------------------------------- D-FEE


def _fee_view_verdict(lines) -> tuple[str | None, bool, float] | None:
    """(verdict, corroborated, conf) for one view, or None when unusable.

    corroborated = the verdict is decided by the authoritative amount rule
    ($809 => paid, $0+waiver => waived — 449/449 on train) or the status
    token is an exact vocabulary read. Fuzzy fee matching at 0.34 admits
    unpaid->paid as a 2-deletion boundary case; this bar is what closes it.
    """
    status_raw = amount = waiver = None
    conf = 0.0
    for line in lines:
        anchored = fields._match_anchor(line.text)
        if anchored is None:
            continue
        fld, _, raw = anchored
        if fld == "fee_status" and status_raw is None:
            status_raw = raw
            conf = max(conf, line.conf)
        elif fld == "fee_amount" and amount is None:
            amount = fields._parse_amount(raw)
            conf = max(conf, line.conf)
        elif fld == "waiver_code" and waiver is None:
            waiver = raw
    if status_raw is None and amount is None:
        return None
    if amount is not None and amount not in (809, 0):
        # Genuine receipts print $809.00 or $0.00 only (449/449 on train).
        # An affirmatively-read implausible amount is receipt inconsistency
        # — _receipt_verdict would fall back to the status line, which is
        # exactly the forged-receipt route (paid + wrong amount) — so the
        # whole view abstains instead.
        return None
    status = fields._correct("fee_status", status_raw) if status_raw else None
    verdict = fields._receipt_verdict(status, amount, waiver)
    if verdict is None:
        return None
    waiver_present = waiver is not None and "WAIVER" in waiver.upper()
    decisive_amount = amount == 809 or (amount == 0 and waiver_present)
    exact_status = status_raw is not None and \
        vocab.strip_captions(status_raw).strip(" .,:;|-").lower() == verdict
    return verdict, (decisive_amount or exact_status), conf


def _fee_head(pdf_path, case_id, pages, scans, evidence, engine,
              budget_left) -> Discharge | None:
    receipts = _candidate_pages(pages, scans, case_id, ("fee_receipt",))
    if len(receipts) != 1:      # zero = nothing to re-read; two = conflict
        return None
    page = receipts[0]
    if not budget_left():
        return None
    agree: dict[str, tuple[str, bool, float]] = {}
    for name, image, sparse in _band_views(pdf_path, page, scans[page.index]):
        read = _fee_view_verdict(_view_lines(engine, image, sparse, page.index))
        if read is not None:
            agree[name] = read
    shape = _shape_verdict(pdf_path, page, scans[page.index]) \
        if _feeshape_enabled() else None
    # One verdict across every reader that produced one — an OCR/shape
    # disagreement is a mutual veto, never a vote.
    verdicts = {v for v, _, _ in agree.values()}
    if shape is not None:
        verdicts.add(shape.value)
    if len(verdicts) != 1:
        return None
    verdict = verdicts.pop()
    if verdict not in _ALLOWED_FEE_VERDICTS:
        return None
    families = {_family(n) for n in agree}
    fire_by_ocr = len(families) >= 2 and \
        any(corroborated for _, corroborated, _ in agree.values())
    # The shape read carries its own two-variant, dual-region agreement
    # (spec §9.4), so it may fire alone where character OCR reads nothing.
    if not fire_by_ocr and shape is None:
        return None
    confs = [c for _, _, c in agree.values()]
    if shape is not None:
        confs.append(shape.conf)
    conf = min(confs)
    views = tuple(sorted(agree) + (list(shape.views) if shape else []))
    evidence.values["fee_status"] = verdict
    evidence.known["fee_status"] = True
    evidence.conf["fee_status"] = conf
    return Discharge(head="fee", field="fee_status", value=verdict,
                     page_index=page.index, views=views, conf=conf)


# --------------------------------------------------------------- D-ARRIVAL


def _unreadable_like(raw: str) -> bool:
    return any(
        vocab.weighted_distance(tok.strip(" .,:;|-").lower(), "unreadable")
        <= _UNREADABLE_MAX_DIST
        for tok in raw.split() if 6 <= len(tok) <= 14
    )


def _arrival_view_date(lines) -> tuple[str, float] | str | None:
    """One view's intake arrival read: (date, conf), "poisoned", or None."""
    dates: set[str] = set()
    conf = 0.0
    for line in lines:
        anchored = fields._match_anchor(line.text)
        if anchored is not None and anchored[0] == "arrival_date":
            if _unreadable_like(anchored[2]):
                return "poisoned"
            fixed = vocab.repair_date(anchored[2])
            if fixed:
                dates.add(fixed)
                conf = max(conf, line.conf)
            continue
        if anchored is None and fields._date_word_in(line.text):
            fixed = vocab.repair_date(line.text)
            if fixed:
                dates.add(fixed)
                conf = max(conf, line.conf)
    if len(dates) != 1:
        return None
    return dates.pop(), conf


def _arrival_head(pdf_path, case_id, pages, scans, evidence, engine,
                  budget_left) -> Discharge | None:
    anchor = evidence.value("arrival_date")
    if not anchor:
        return None     # nothing independently recovered to corroborate
    intakes = _candidate_pages(pages, scans, case_id, ("intake",))
    if len(intakes) != 1:
        return None
    page = intakes[0]
    # The primary ladder may already have read the printed UNREADABLE
    # marker — the organizer's residual family. That read is affirmative
    # evidence of invisibility and vetoes any head view outright.
    for line in page.lines:
        anchored = fields._match_anchor(line.text)
        if anchored is not None and anchored[0] == "arrival_date" \
                and _unreadable_like(anchored[2]):
            return None
    if not budget_left():
        return None
    agree: dict[str, tuple[str, float]] = {}
    for name, image, sparse in _band_views(pdf_path, page, scans[page.index]):
        read = _arrival_view_date(_view_lines(engine, image, sparse, page.index))
        if read == "poisoned":
            return None
        if read is not None:
            agree[name] = read
    dates = {d for d, _ in agree.values()}
    if len(dates) != 1 or dates.pop() != anchor:
        return None
    if len({_family(n) for n in agree}) < 2:
        return None
    conf = min(c for _, c in agree.values())
    evidence.arrival_on_intake = True
    return Discharge(head="arrival", field="arrival_on_intake", value=anchor,
                     page_index=page.index, views=tuple(sorted(agree)),
                     conf=conf)


# --------------------------------------------------------------- D-SOLEGAP


def _sponsor_view_read(lines) -> tuple[str, float, tuple[str, ...]] | None:
    """(SPN, conf, raw texts) for one view; None when absent or ambiguous."""
    spns: set[str] = set()
    raws: list[str] = []
    conf = 0.0
    for line in lines:
        anchored = fields._match_anchor(line.text)
        if anchored is not None and anchored[0] == "sponsor_id":
            spn = vocab.repair_sponsor_id(anchored[2])
        else:
            m = fields._ATTEST_SPONSOR_RE.search(line.text)
            spn = f"SPN-{m.group(1)}" if m else None
        if spn:
            spns.add(spn)
            raws.append(line.text)
            conf = max(conf, line.conf)
    if len(spns) != 1:
        return None
    return spns.pop(), conf, tuple(raws)


def _visa_view_read(lines) -> tuple[str, float] | None:
    """Exact-token visa read for one view — no fuzzy minting at all: DIP-1
    and TRANSIT-7 are both decision-sensitive directions."""
    classes: set[str] = set()
    conf = 0.0
    for line in lines:
        anchored = fields._match_anchor(line.text)
        if anchored is None or anchored[0] != "visa_class":
            continue
        token = vocab.strip_captions(anchored[2]).strip(" .,:;|-")
        if token in vocab.VISA_CLASSES:
            classes.add(token)
            conf = max(conf, line.conf)
    if len(classes) != 1:
        return None
    return classes.pop(), conf


def _solegap_head(pdf_path, case_id, pages, scans, evidence, engine,
                  budget_left, field: str) -> Discharge | None:
    candidates = _candidate_pages(pages, scans, case_id,
                                  ("attestation", "intake"))
    accepted: list[tuple[int, str, tuple[str, ...], float]] = []
    for page in candidates:
        if not budget_left():
            return None
        agree: dict[str, tuple] = {}
        for name, image, sparse in _band_views(pdf_path, page,
                                               scans[page.index]):
            lines = _view_lines(engine, image, sparse, page.index)
            read = _sponsor_view_read(lines) if field == "sponsor_id" \
                else _visa_view_read(lines)
            if read is not None:
                agree[name] = read
        values = {r[0] for r in agree.values()}
        if len(values) != 1 or len({_family(n) for n in agree}) < 2:
            continue
        value = values.pop()
        if field == "sponsor_id" and value in policy.REVOKED_SPONSORS:
            # A revoked mint requires the id VERBATIM in every agreeing
            # view's raw text: a repaired near-miss must never create an
            # R4 denial (same posture as the weld's revoked-list drop).
            if not all(any(value in raw for raw in r[2])
                       for r in agree.values()):
                continue
        conf = min(r[1] for r in agree.values())
        accepted.append((page.index, value, tuple(sorted(agree)), conf))
    if len({value for _, value, _, _ in accepted}) != 1:
        return None     # zero pages read it, or two pages disagree
    page_index, value, views, conf = accepted[0]
    evidence.values[field] = value
    evidence.known[field] = True
    evidence.conf[field] = conf
    return Discharge(head="solegap", field=field, value=value,
                     page_index=page_index, views=views, conf=conf)


# ----------------------------------------------------------------- entrypoint


def run_discharge(*, pdf_path: str, case_id: str, pages: list[Page],
                  scans: dict, evidence: CaseEvidence, path: str, engine,
                  budget_left) -> Discharge | None:
    """Run the head owning `path`, if any and enabled; mutates `evidence`
    only on an accepted fire. Returns the provenance record or None."""
    trigger = _TRIGGERS.get(path)
    if trigger is None:
        return None
    head, field = trigger
    if not head_enabled(head) or not budget_left():
        return None
    if head == "fee":
        return _fee_head(pdf_path, case_id, pages, scans, evidence, engine,
                         budget_left)
    if head == "arrival":
        return _arrival_head(pdf_path, case_id, pages, scans, evidence,
                             engine, budget_left)
    return _solegap_head(pdf_path, case_id, pages, scans, evidence, engine,
                         budget_left, field)
