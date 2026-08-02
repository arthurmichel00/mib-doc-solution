"""Quarantined note-band rescue behind MIB_NOTE_RESCUE=1 (Saturday branch).

The 288-DPI render is a 2x upsample of the 144-DPI embedded scan; on a
destroyed note's condensed micro-text the upsample welds glyphs and the
Reason sentence reads ~30 points below the template bars (diagnosed on
MIB-001000). ocr.note_band_lines re-reads the note header band at native
resolution; fields.note_template_finding scores ONLY those lines through
the N1 reason-template probe. The rescue lines never join page.lines or
any other consumer, and pipeline._note_rescue_candidate abstains outright
when two pages mint different labels. With the flag unset the code is
dead: no OCR pass, no behavior change.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mib_pipeline import decision, fields, ocr, pipeline, policy
from mib_pipeline.fields import CaseEvidence
from mib_pipeline.model import Line, Page, PageKind, Source
from mib_pipeline.ocr import OcrWord

# Runtime-observed rescue read of MIB-001000's note band (prototype run:
# mints "Clean or exception-qualified packet." APPROVED at conf 0.654,
# case adjudicates N1_reason_approved 0.828).
RESCUE_1000 = "Reesor Clean cr exceptionquaified packet"
APPROVE_TEMPLATE = "Clean or exception-qualified packet."
DENY_TEMPLATE = "Mandatory fee unpaid."

CASE_ID = "MIB-000123"


@pytest.fixture
def reason_adj_on(monkeypatch):
    monkeypatch.setenv("MIB_REASON_ADJ", "1")


@pytest.fixture
def reason_adj_off(monkeypatch):
    monkeypatch.delenv("MIB_REASON_ADJ", raising=False)


@pytest.fixture
def rescue_on(monkeypatch):
    monkeypatch.setenv("MIB_NOTE_RESCUE", "1")


@pytest.fixture
def rescue_off(monkeypatch):
    monkeypatch.delenv("MIB_NOTE_RESCUE", raising=False)


def _lines(texts, index=0, conf=0.654):
    return [Line(text=t, page_index=index, source=Source.OCR, conf=conf)
            for t in texts]


def _eligible_page(index=0):
    """Sparse untyped scan page: no doc_type, no foreign labels."""
    return Page(index=index, kind=PageKind.SCAN,
                lines=_lines(["Now"], index=index, conf=0.96))


def _registry_page(index=0):
    page = Page(index=index, kind=PageKind.SCAN,
                lines=_lines(["Registry Status: verified"], index=index))
    page.doc_type = "registry"
    return page


class RecordingEngine:
    def __init__(self, canned=None):
        self.calls = []
        self.images = []
        self._canned = list(canned or [])

    def words(self, image, sparse=False):
        self.calls.append((image.shape, sparse))
        self.images.append(image)
        return self._canned.pop(0) if self._canned else []


# ------------------------------------------------- ocr.note_band_lines


class TestNoteBandLines:
    def test_band_geometry_and_sparse_variants(self):
        # 288-DPI letter render: 792x612pt * 4 px/pt.
        gray = np.full((3168, 2448), 255, np.uint8)
        words = [OcrWord("Reesor", 0.7, (1, 1, 1), 10.0),
                 OcrWord("packet", 0.6, (1, 1, 1), 90.0)]
        engine = RecordingEngine(canned=[words, []])
        lines = ocr.note_band_lines(gray, 2, engine)
        # band (30,350,20,560)pt * 4 = 1280x2160 px, downsampled 0.5 to
        # native then upsampled 1.5: 960x1620, plus pad_for_ocr's 30 px
        # white border on every side (tesseract 5.3 clips crop-edge
        # glyphs): 1020x1680. Two sparse passes (raw + divblur), nothing
        # else.
        assert engine.calls == [((1020, 1680), True), ((1020, 1680), True)]
        assert [l.text for l in lines] == ["Reesor packet"]
        assert lines[0].page_index == 2
        assert lines[0].source == Source.OCR

    def test_both_band_reads_are_edge_padded(self):
        """Both sparse reads (raw + divblur) see a white pad_for_ocr
        border, so band content never touches the image edge — the
        tesseract 5.3 crop-edge clipping that motivated the pad."""
        gray = np.zeros((3168, 2448), np.uint8)    # all-ink band content
        engine = RecordingEngine()
        ocr.note_band_lines(gray, 0, engine)
        assert len(engine.images) == 2
        for img in engine.images:
            assert img.shape == (960 + 60, 1620 + 60)
            border = np.concatenate([img[:30].ravel(), img[-30:].ravel(),
                                     img[:, :30].ravel(),
                                     img[:, -30:].ravel()])
            assert (border == 255).all()
        # the raw view's interior is still the (dark) band content
        assert (engine.images[0][30:-30, 30:-30] == 0).all()

    def test_empty_band_reads_nothing(self):
        engine = RecordingEngine()
        assert ocr.note_band_lines(np.full((50, 40), 255, np.uint8),
                                   0, engine) == []
        assert engine.calls == []


# --------------------------------------- fields.note_template_finding


class TestNoteTemplateFinding:
    def test_1000_runtime_read_mints_n1_approved(self, reason_adj_on):
        page = _eligible_page(index=2)
        found = fields.note_template_finding(page, _lines([RESCUE_1000],
                                                          index=2))
        assert found is not None
        assert found.label == "APPROVED"
        assert found.template == APPROVE_TEMPLATE
        assert found.template_only is True
        assert found.conf == pytest.approx(0.654)

    def test_minted_candidate_routes_n1_in_policy(self, reason_adj_on):
        ev = CaseEvidence()   # empty evidence -> R8_fee_unread
        ev.template_finding = fields.note_template_finding(
            _eligible_page(), _lines([RESCUE_1000]))
        assert policy.adjudicate(ev) == ("APPROVED", "N1_reason_approved")

    def test_reason_adj_flag_off_is_inert(self, reason_adj_off):
        assert fields.note_template_finding(
            _eligible_page(), _lines([RESCUE_1000])) is None

    def test_ineligible_page_never_mints(self, reason_adj_on):
        assert fields.note_template_finding(
            _registry_page(), _lines([RESCUE_1000])) is None

    def test_rescue_lines_stay_quarantined(self, reason_adj_on):
        page = _eligible_page()
        before = list(page.lines)
        fields.note_template_finding(page, _lines([RESCUE_1000]))
        assert page.lines == before          # probe never touches the page

    def test_public_alias_is_the_eligibility_gate(self):
        assert fields.finding_eligible is fields._finding_eligible


# --------------------------------- pipeline._note_rescue_candidate


class TestNoteRescueCandidate:
    def _run(self, monkeypatch, band_texts, pages, budget=lambda: True):
        """band_texts: {page_index: [line texts]} served by the fake OCR."""
        queried = []

        def fake_band(gray, page_index, engine):
            queried.append(page_index)
            return _lines(band_texts.get(page_index, []), index=page_index)

        monkeypatch.setattr(ocr, "note_band_lines", fake_band)
        scans = {p.index: SimpleNamespace(gray=np.zeros((4, 4), np.uint8))
                 for p in pages}
        got = pipeline._note_rescue_candidate(pages, scans, engine=None,
                                              budget_left=budget)
        return got, queried

    def test_agreeing_pages_return_the_best_read(self, reason_adj_on,
                                                 monkeypatch):
        pages = [_eligible_page(0), _eligible_page(1)]
        got, queried = self._run(
            monkeypatch,
            {0: [APPROVE_TEMPLATE], 1: [RESCUE_1000]}, pages)
        assert queried == [0, 1]
        assert got is not None and got.label == "APPROVED"
        assert got.conf == pytest.approx(0.654)

    def test_disagreeing_pages_abstain(self, reason_adj_on, monkeypatch):
        pages = [_eligible_page(0), _eligible_page(1)]
        got, _ = self._run(
            monkeypatch,
            {0: [APPROVE_TEMPLATE], 1: [DENY_TEMPLATE]}, pages)
        assert got is None

    def test_ineligible_pages_skip_the_ocr_entirely(self, reason_adj_on,
                                                    monkeypatch):
        pages = [_registry_page(0), _eligible_page(1)]
        got, queried = self._run(
            monkeypatch, {0: [APPROVE_TEMPLATE], 1: [RESCUE_1000]}, pages)
        assert queried == [1]                # typed page never OCR'd
        assert got is not None and got.label == "APPROVED"

    def test_budget_exhausted_reads_nothing(self, reason_adj_on,
                                            monkeypatch):
        pages = [_eligible_page(0)]
        got, queried = self._run(monkeypatch, {0: [APPROVE_TEMPLATE]},
                                 pages, budget=lambda: False)
        assert got is None and queried == []

    def test_pages_are_never_mutated(self, reason_adj_on, monkeypatch):
        pages = [_eligible_page(0)]
        before = list(pages[0].lines)
        self._run(monkeypatch, {0: [RESCUE_1000]}, pages)
        assert pages[0].lines == before


# ------------------------------------------------------- flag gating


class TestFlagGate:
    def test_default_off(self, rescue_off):
        assert pipeline.NOTE_RESCUE_DEFAULT is False
        assert pipeline._note_rescue_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_NOTE_RESCUE", "1")
        assert pipeline._note_rescue_enabled() is True
        monkeypatch.setenv("MIB_NOTE_RESCUE", "0")
        assert pipeline._note_rescue_enabled() is False


# --------------------------------------- pipeline wiring (real PDF, e2e)


def _make_scan_pdf(tmp_path, page_specs, case_id=CASE_ID):
    """Synthetic packet whose pages are full-page raster images (SCAN kind)
    with the genuine vector footer (pattern from test_discharge_heads)."""
    import pymupdf

    doc = pymupdf.open()
    for i, lines in enumerate(page_specs):
        src = pymupdf.open()
        sp = src.new_page(width=612, height=792)
        y = 70
        for line in lines:
            sp.insert_text((60, y), line, fontsize=16, fontname="helv")
            y += 30
        pix = sp.get_pixmap(dpi=180)
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, pixmap=pix)
        page.insert_text((40, 782), f"Packet {case_id} / page {i + 1}",
                         fontsize=6, fontname="helv")
        src.close()
    path = tmp_path / f"{case_id}.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


INTAKE_SPEC = [
    "FORM I-8090: Extraterrestrial Work Authorization Intake",
    f"Case ID: {CASE_ID}",
    "Applicant: Solul Zamora",
    "Species Code: ORION_GRAYS",
    "Home World: Proxima-b",
    "Visa Class: DIP-1",
    "Arrival Date: 2026-05-04",
    "Declared Purpose: research",
]
NOTE_SPEC = [
    "Manual Adjudicator Note",
]


def _quiet_escalation(monkeypatch):
    """Silence the heavy under-determined engines irrelevant to the block
    under test (their own suites cover them)."""
    from mib_pipeline import crnn

    monkeypatch.setattr(ocr, "rapid_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "escalation_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "weld_sponsor_lines", lambda *a, **k: [])
    monkeypatch.setattr(crnn, "crnn_lines", lambda *a, **k: [])


class TestIntegrationPipeline:
    def test_flag_off_never_touches_the_band(self, reason_adj_on,
                                             rescue_off, tmp_path,
                                             monkeypatch):
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC, NOTE_SPEC])
        calls = []
        monkeypatch.setattr(
            ocr, "note_band_lines",
            lambda *a, **k: calls.append(1) or [])
        row = pipeline.process_pdf(pdf)
        assert calls == []                    # inert: zero OCR passes
        assert row["_path"] == "R8_fee_unread"
        assert row["adjudication"] == "NEEDS_REVIEW"

    def test_flag_on_rescues_the_1000_class(self, reason_adj_on, rescue_on,
                                            tmp_path, monkeypatch):
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC, NOTE_SPEC])
        queried = []

        def fake_band(gray, page_index, engine):
            queried.append(page_index)
            return _lines([RESCUE_1000], index=page_index)

        monkeypatch.setattr(ocr, "note_band_lines", fake_band)
        row = pipeline.process_pdf(pdf)
        # only the note-eligible page is re-read; the typed intake is not
        assert queried == [1]
        assert row["_path"] == "N1_reason_approved"
        assert row["adjudication"] == "APPROVED"
        assert row["confidence"] == pytest.approx(
            decision.decide("APPROVED", "N1_reason_approved")[1])
        # quarantine: the rescue read never leaks into extracted fields
        assert all(RESCUE_1000 not in str(v) for v in row.values())

    def test_flag_on_without_reason_adj_mints_nothing(self, reason_adj_off,
                                                      rescue_on, tmp_path,
                                                      monkeypatch):
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC, NOTE_SPEC])
        monkeypatch.setattr(
            ocr, "note_band_lines",
            lambda gray, page_index, engine: _lines([RESCUE_1000],
                                                    index=page_index))
        row = pipeline.process_pdf(pdf)
        assert row["_path"] == "R8_fee_unread"
        assert row["adjudication"] == "NEEDS_REVIEW"
