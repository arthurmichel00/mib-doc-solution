"""Structural review-discharge heads behind MIB_DISCHARGE=1 (A8).

Spec: docs/superpowers/specs/2026-07-29-structural-review-discharge-design.md
(§5 test contract, §8 binding acceptance bars). A head never writes APPROVED:
it upgrades exactly one under-determined evidence item from a cross-view
re-read of the packet's own evidence page, then the unchanged policy cascade
re-runs. With the flag unset, behavior is identical to the shipped code.

The full-corpus sha byte-identity proof and the named-ID sentinel sweep
(spec T3 over the 27 silent-flag + 21+6 path gold-D + pixel-lie sets) are
BLOCKED-ON-CLEARANCE (machine discipline, spec §8.5); their unit-shaped
equivalents live in TestT3SentinelShapes / TestIntegrationPipeline here.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from mib_pipeline import decision, discharge, policy, writer
from mib_pipeline.fields import CaseEvidence
from mib_pipeline.model import Page, PageKind, Source, TextSpan
from mib_pipeline.ocr import OcrWord, ScanOcrResult

CASE_ID = "MIB-000123"


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MIB_DISCHARGE", "1")
    for sub in ("FEE", "ARRIVAL", "SOLEGAP"):
        monkeypatch.delenv(f"MIB_DISCHARGE_{sub}", raising=False)


@pytest.fixture
def flag_off(monkeypatch):
    for name in ("MIB_DISCHARGE", "MIB_DISCHARGE_FEE",
                 "MIB_DISCHARGE_ARRIVAL", "MIB_DISCHARGE_SOLEGAP"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ helpers


def _evidence(**overrides) -> CaseEvidence:
    """Evidence that reaches R11 unless a field is knocked out."""
    ev = CaseEvidence()
    values = {
        "applicant_name": "Solul Zamora", "species_code": "ORION_GRAYS",
        "home_world": "Proxima-b", "visa_class": "XW-1",
        "sponsor_id": "SPN-1234", "arrival_date": "2026-05-04",
        "declared_purpose": "research", "fee_status": "paid",
    }
    values.update({k: v for k, v in overrides.items() if k in values})
    for fld, value in values.items():
        ev.values[fld] = value
        ev.known[fld] = value is not None
        ev.conf[fld] = 0.9 if value is not None else 0.0
    ev.flags = set(overrides.get("flags", set()))
    ev.flags_known = overrides.get("flags_known", True)
    ev.arrival_on_intake = overrides.get("arrival_on_intake", True)
    return ev


def _scan_page(index: int, doc_type: str | None, lines=()) -> Page:
    page = Page(index=index, kind=PageKind.SCAN, lines=list(lines))
    page.doc_type = doc_type
    return page


def _span(text: str) -> TextSpan:
    return TextSpan(text=text, bbox=(40, 770, 300, 780), size=6.0,
                    color=(0.0,), opacity=1.0, render_type=0, font="Helvetica")


def _scans_for(pages) -> dict[int, ScanOcrResult]:
    return {
        p.index: ScanOcrResult(lines=[], gray=np.zeros((8, 8), np.uint8),
                               upright=True)
        for p in pages if p.kind == PageKind.SCAN
    }


def _words(lines: list[str], conf: float = 0.9) -> list[OcrWord]:
    out = []
    for li, text in enumerate(lines):
        for wi, tok in enumerate(text.split()):
            out.append(OcrWord(tok, conf, (0, 0, li), float(wi)))
    return out


class FakeEngine:
    """Scripted OCR: maps (view sentinel image, sparse) -> words."""

    def __init__(self, script: dict):
        self.script = script

    def words(self, image, sparse: bool = False):
        return self.script.get((image, sparse), [])


def _fake_views(monkeypatch, per_page: dict[int, list[tuple[str, str]]]):
    """Monkeypatch the rasterization seam: page.index -> [(name, sentinel)]."""

    def fake(pdf_path, page, scan, budget_left=None):
        return [(name, img, False) for name, img in per_page.get(page.index, [])]

    monkeypatch.setattr(discharge, "_band_views", fake)


def _run(pages, evidence, engine, path=None, budget=lambda: True):
    if path is None:
        _, path = policy.adjudicate(evidence)
    return discharge.run_discharge(
        pdf_path="/nonexistent.pdf", case_id=CASE_ID, pages=pages,
        scans=_scans_for(pages), evidence=evidence, path=path,
        engine=engine, budget_left=budget,
    )


def _snapshot(ev: CaseEvidence):
    return (copy.deepcopy(ev.values), copy.deepcopy(ev.known),
            copy.deepcopy(ev.conf), set(ev.flags), ev.flags_known,
            ev.arrival_on_intake)


RECEIPT_GOOD = ["MIB Fee Receipt", "Case ID: MIB-000123",
                "Fee Status: paid", "Amount: $809.00", "Waiver Code: N/A"]


def _fee_setup(monkeypatch, receipt_lines=RECEIPT_GOOD, view_names=("raw", "otsu"),
               extra_pages=(), receipt_doc_type="fee_receipt"):
    """Standard firing-shaped fee case: R8_fee_unread + one legible receipt."""
    receipt = _scan_page(1, receipt_doc_type)
    pages = [_scan_page(0, "intake"), receipt, *extra_pages]
    _fake_views(monkeypatch, {1: [(n, f"receipt-{n}") for n in view_names]})
    engine = FakeEngine({(f"receipt-{n}", False): _words(receipt_lines)
                         for n in view_names})
    ev = _evidence(fee_status=None)
    return pages, ev, engine


# ------------------------------------------------------------ flag gating


class TestFlagGating:
    def test_default_env_is_fully_off(self, flag_off):
        assert not discharge.any_enabled()

    def test_flag_off_never_fires_even_on_perfect_evidence(
            self, flag_off, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch)
        before = _snapshot(ev)
        assert _run(pages, ev, engine) is None
        assert _snapshot(ev) == before

    def test_master_on_fires(self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.head == "fee"

    def test_sub_flag_zero_disables_one_head_only(self, flag_on, monkeypatch):
        monkeypatch.setenv("MIB_DISCHARGE_FEE", "0")
        pages, ev, engine = _fee_setup(monkeypatch)
        assert _run(pages, ev, engine) is None

    def test_sub_flag_without_master_stays_off(self, flag_off, monkeypatch):
        monkeypatch.setenv("MIB_DISCHARGE_FEE", "1")
        pages, ev, engine = _fee_setup(monkeypatch)
        assert not discharge.any_enabled()
        assert _run(pages, ev, engine) is None

    def test_conf_cap_constant(self):
        assert discharge.CONF_CAP == 0.90


# --------------------------------------------------- D-FEE fire + cascade


class TestFeeHead:
    def test_fires_and_cascade_approves(self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch)
        fired = _run(pages, ev, engine)
        assert fired is not None
        assert (fired.head, fired.field, fired.value) == \
            ("fee", "fee_status", "paid")
        assert ev.value("fee_status") == "paid" and ev.is_known("fee_status")
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("APPROVED", "R11_default_approve")

    def test_waived_receipt_discharges_waived(self, flag_on, monkeypatch):
        lines = ["MIB Fee Receipt", "Fee Status: waived", "Amount: $0.00",
                 "Waiver Code: DIP-WAIVER"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "waived"

    def test_remaining_gap_still_blocks_approval(self, flag_on, monkeypatch):
        # Discharge supplies FEE evidence only; an unread flags line keeps
        # the case at R12 — the firewall the CFA analysis (§4) relies on.
        pages, ev, engine = _fee_setup(monkeypatch)
        ev.flags_known = False
        fired = _run(pages, ev, engine)
        assert fired is not None
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("NEEDS_REVIEW", "R12_flags_unread")


class TestFeeAcceptanceBars:
    def test_single_view_never_fires(self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch, view_names=("raw",))
        assert _run(pages, ev, engine) is None

    def test_disagreeing_views_never_fire(self, flag_on, monkeypatch):
        pages, ev, _ = _fee_setup(monkeypatch)
        engine = FakeEngine({
            ("receipt-raw", False): _words(RECEIPT_GOOD),
            ("receipt-otsu", False): _words(
                ["MIB Fee Receipt", "Fee Status: unpaid", "Amount: $0.00"]),
        })
        assert _run(pages, ev, engine) is None

    def test_unknown_verdict_never_fires(self, flag_on, monkeypatch):
        # paid/waived claim with no money and no waiver = the generator's
        # inconsistent-receipt pattern; a printed unknown is R8_fee_unknown
        # territory and not the head's business either way.
        lines = ["MIB Fee Receipt", "Fee Status: waived", "Amount: $0.00",
                 "Waiver Code: N/A"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        assert _run(pages, ev, engine) is None

    def test_fuzzy_status_without_amount_never_fires(self, flag_on, monkeypatch):
        # "pait" corrects to paid inside the 0.34 budget, but with no parsed
        # amount and a non-exact token the §8.4 corroboration bar holds.
        lines = ["MIB Fee Receipt", "Fee Status: pait"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        assert _run(pages, ev, engine) is None

    def test_exact_status_token_without_amount_fires(self, flag_on, monkeypatch):
        lines = ["MIB Fee Receipt", "Fee Status: unpaid"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "unpaid"

    def test_809_amount_with_garbled_status_fires_paid(self, flag_on, monkeypatch):
        # amount is authoritative (297/297 in train): $809.00 is a paid fee.
        lines = ["MIB Fee Receipt", "Fee Status: p@#d", "Amount: $809.00"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "paid"

    def test_implausible_amount_is_receipt_inconsistency_never_fires(
            self, flag_on, monkeypatch):
        # The forged-receipt route: an exact "paid" token would corroborate,
        # but genuine receipts print $809.00 or $0.00 only — a read $500
        # makes the whole view abstain rather than let the status decide.
        lines = ["MIB Fee Receipt", "Fee Status: paid", "Amount: $500.00"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        assert _run(pages, ev, engine) is None


class TestPpocrViewFamily:
    def test_ppocr_corroborating_tesseract_fires(self, flag_on, monkeypatch):
        from mib_pipeline import ocr as o
        pages = [_scan_page(0, "intake"), _scan_page(1, "fee_receipt")]

        def views(pdf_path, page, scan, budget_left=None):
            return [("raw", "tess-img", False), ("ppocr-raw", "pp-img", "ppocr")]

        monkeypatch.setattr(discharge, "_band_views", views)
        monkeypatch.setattr(
            o, "rapid_lines",
            lambda image, idx: [] if image != "pp-img" else [
                __import__("mib_pipeline.model", fromlist=["Line"]).Line(
                    text=t, page_index=idx, source=Source.OCR, conf=0.8)
                for t in RECEIPT_GOOD])
        engine = FakeEngine({("tess-img", False): _words(RECEIPT_GOOD)})
        ev = _evidence(fee_status=None)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "paid"
        assert set(fired.views) == {"raw", "ppocr-raw"}

    def test_ppocr_alone_never_fires(self, flag_on, monkeypatch):
        # Both PP-OCR views share one family: the same recognizer prior can
        # repeat a misread across preprocessings of the same pixels.
        from mib_pipeline import ocr as o
        from mib_pipeline.model import Line
        pages = [_scan_page(0, "intake"), _scan_page(1, "fee_receipt")]

        def views(pdf_path, page, scan, budget_left=None):
            return [("ppocr-raw", "pp-a", "ppocr"),
                    ("ppocr-divblur", "pp-b", "ppocr")]

        monkeypatch.setattr(discharge, "_band_views", views)
        monkeypatch.setattr(
            o, "rapid_lines",
            lambda image, idx: [Line(text=t, page_index=idx,
                                     source=Source.OCR, conf=0.8)
                                for t in RECEIPT_GOOD])
        ev = _evidence(fee_status=None)
        assert _run(pages, ev, FakeEngine({})) is None


# ------------------------------------------------ T1: evidence ablation


class TestT1EvidenceAblation:
    def test_untyped_receipt_page_never_fires(self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch, receipt_doc_type=None)
        assert _run(pages, ev, engine) is None

    def test_unrelated_page_in_place_of_receipt_never_fires(
            self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch, receipt_doc_type="registry")
        assert _run(pages, ev, engine) is None

    def test_other_cases_receipt_never_fires(self, flag_on, monkeypatch):
        # A same-type page from ANOTHER case: its own digital footer names
        # the foreign case, so it is not this case's evidence.
        pages, ev, engine = _fee_setup(monkeypatch)
        pages[1].visible_spans = [_span("Packet MIB-000999 / page 2")]
        assert _run(pages, ev, engine) is None

    def test_two_receipt_pages_conflict_never_fires(self, flag_on, monkeypatch):
        second = _scan_page(2, "fee_receipt")
        pages, ev, engine = _fee_setup(monkeypatch, extra_pages=(second,))
        assert _run(pages, ev, engine) is None

    def test_blank_views_never_fire(self, flag_on, monkeypatch):
        pages, ev, _ = _fee_setup(monkeypatch)
        assert _run(pages, ev, FakeEngine({})) is None

    def test_arrival_ablation_no_intake_page_never_fires(
            self, flag_on, monkeypatch):
        pages = [_scan_page(0, "registry"), _scan_page(1, "fee_receipt")]
        _fake_views(monkeypatch, {0: [("raw", "p0")], 1: [("raw", "p1")]})
        engine = FakeEngine({("p0", False): _words(["Arrival Date: 2026-05-04"]),
                             ("p1", False): _words(["Arrival Date: 2026-05-04"])})
        ev = _evidence(arrival_on_intake=False)
        assert _run(pages, ev, engine) is None
        assert ev.arrival_on_intake is False

    def test_solegap_ablation_wrong_page_type_never_fires(
            self, flag_on, monkeypatch):
        pages = [_scan_page(0, "registry")]
        _fake_views(monkeypatch, {0: [("raw", "a"), ("otsu", "b")]})
        engine = FakeEngine({(img, False): _words(["Sponsor ID: SPN-4732"])
                             for img in ("a", "b")})
        ev = _evidence(sponsor_id=None)
        assert _run(pages, ev, engine) is None


# ------------------------------------ T2: packet shape never discharges


class TestT2PacketShapeNeverDischarges:
    def test_shuffled_pages_and_decoys_never_fire_without_evidence(
            self, flag_on, monkeypatch):
        base = [_scan_page(0, "intake"), _scan_page(1, "fee_receipt"),
                _scan_page(2, "biometric"), _scan_page(3, "registry")]
        decoy = _scan_page(4, "registry")
        decoy.visible_spans = [_span("Packet MIB-000999 / page 1")]
        for ordering in ([3, 1, 0, 2], [2, 0, 1, 3], [0, 1, 2, 3]):
            pages = [base[i] for i in ordering] + [decoy, _scan_page(5, None)]
            _fake_views(monkeypatch, {p.index: [("raw", f"junk-{p.index}"),
                                                ("otsu", f"junk2-{p.index}")]
                                      for p in pages})
            engine = FakeEngine({(f"junk-{i}", False): _words(["| ~~ noise ~~ |"])
                                 for i in range(6)})
            for knockout in ("fee_status", "sponsor_id", "visa_class"):
                ev = _evidence(**{knockout: None})
                before = _snapshot(ev)
                assert _run(pages, ev, engine) is None, knockout
                assert _snapshot(ev) == before, knockout
            ev = _evidence(arrival_on_intake=False)
            before = _snapshot(ev)
            assert _run(pages, ev, engine) is None
            assert _snapshot(ev) == before

    def test_evidence_on_the_wrong_page_never_fires(self, flag_on, monkeypatch):
        # A perfect fee row printed on the BIOMETRIC page must not discharge
        # a fee review: the head reads only its own evidence page type.
        pages = [_scan_page(0, "intake"), _scan_page(1, "biometric")]
        _fake_views(monkeypatch, {1: [("raw", "b1"), ("otsu", "b2")]})
        engine = FakeEngine({(img, False): _words(RECEIPT_GOOD)
                             for img in ("b1", "b2")})
        ev = _evidence(fee_status=None)
        assert _run(pages, ev, engine) is None


# ----------------------------- T3/T4: sentinel shapes + affirmative paths


class TestT3SentinelShapes:
    def test_silent_flag_shape_is_never_dischargeable(
            self, flag_on, monkeypatch):
        # The 27-case silent-flag family: everything readable EXCEPT flags.
        # No head targets R12_flags_unread (D-FLAGS killed FINAL) — perfect
        # views of every other page must change nothing.
        pages = [_scan_page(0, "intake"), _scan_page(1, "fee_receipt"),
                 _scan_page(2, "biometric")]
        _fake_views(monkeypatch, {i: [("raw", f"v{i}"), ("otsu", f"w{i}")]
                                  for i in range(3)})
        engine = FakeEngine({(img, False): _words(RECEIPT_GOOD)
                             for img in ("v1", "w1")})
        ev = _evidence(flags_known=False)
        _, path = policy.adjudicate(ev)
        assert path == "R12_flags_unread"
        before = _snapshot(ev)
        assert _run(pages, ev, engine) is None
        assert _snapshot(ev) == before

    def test_unpaid_discharge_lands_denied_not_approved(
            self, flag_on, monkeypatch):
        lines = ["MIB Fee Receipt", "Fee Status: unpaid", "Amount: $0.00"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "unpaid"
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("DENIED", "R3_fee_unpaid")

    def test_stale_arrival_discharge_lands_denied(self, flag_on, monkeypatch):
        pages = [_scan_page(0, "intake")]
        _fake_views(monkeypatch, {0: [("raw", "i1"), ("otsu", "i2")]})
        engine = FakeEngine({(img, False): _words(["Arrival Date: 2025-11-02"])
                             for img in ("i1", "i2")})
        ev = _evidence(arrival_date="2025-11-02", arrival_on_intake=False)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.head == "arrival"
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("DENIED", "R6_stale_arrival")

    def test_arrival_discharge_with_review_flag_stays_review(
            self, flag_on, monkeypatch):
        pages = [_scan_page(0, "intake")]
        _fake_views(monkeypatch, {0: [("raw", "i1"), ("otsu", "i2")]})
        engine = FakeEngine({(img, False): _words(["Arrival Date: 2026-05-04"])
                             for img in ("i1", "i2")})
        ev = _evidence(arrival_on_intake=False,
                       flags={"illegible_biometrics"})
        fired = _run(pages, ev, engine)
        assert fired is not None
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("NEEDS_REVIEW", "R10_review_flags")

    def test_fuzzy_revoked_sponsor_never_fires(self, flag_on, monkeypatch):
        # A revoked-list mint requires a verbatim read on both views: a
        # repaired near-miss must not create an R4 denial (weld posture).
        pages = [_scan_page(0, "attestation")]
        _fake_views(monkeypatch, {0: [("raw", "a1"), ("otsu", "a2")]})
        engine = FakeEngine({(img, False): _words(["Sponsor ID: SPN-27I8"])
                             for img in ("a1", "a2")})
        ev = _evidence(sponsor_id=None)
        assert _run(pages, ev, engine) is None

    def test_verbatim_revoked_sponsor_fires_and_lands_denied(
            self, flag_on, monkeypatch):
        pages = [_scan_page(0, "attestation")]
        _fake_views(monkeypatch, {0: [("raw", "a1"), ("otsu", "a2")]})
        engine = FakeEngine({(img, False): _words(["Sponsor ID: SPN-2718"])
                             for img in ("a1", "a2")})
        ev = _evidence(sponsor_id=None)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "SPN-2718"
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("DENIED", "R4_revoked_sponsor")


class TestT4AffirmativeReviewsUntouched:
    @pytest.mark.parametrize("path", [
        "N0_note_needs_review", "N0_note_conflict", "R8_fee_unknown",
        "R10_review_flags", "N1_reason_needs_review", "FALLBACK_error",
    ])
    def test_affirmative_paths_never_run_heads(self, flag_on, monkeypatch, path):
        pages, ev, engine = _fee_setup(monkeypatch)
        before = _snapshot(ev)
        assert _run(pages, ev, engine, path=path) is None
        assert _snapshot(ev) == before


# --------------------------------------- D-ARRIVAL + T5: residual guard


class TestArrivalHead:
    def _setup(self, monkeypatch, intake_lines, anchor="2026-05-04",
               existing_lines=()):
        pages = [_scan_page(0, "intake", lines=existing_lines),
                 _scan_page(1, "fee_receipt")]
        _fake_views(monkeypatch, {0: [("raw", "i1"), ("otsu", "i2")]})
        engine = FakeEngine({(img, False): _words(intake_lines)
                             for img in ("i1", "i2")})
        ev = _evidence(arrival_date=anchor, arrival_on_intake=False)
        return pages, ev, engine

    def test_fires_when_intake_date_matches_anchor(self, flag_on, monkeypatch):
        pages, ev, engine = self._setup(
            monkeypatch, ["Arrival Date: 2026-05-04"])
        fired = _run(pages, ev, engine)
        assert fired is not None
        assert (fired.head, fired.field) == ("arrival", "arrival_on_intake")
        assert ev.arrival_on_intake is True
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("APPROVED", "R11_default_approve")

    def test_unreadable_token_is_hard_abstain(self, flag_on, monkeypatch):
        pages, ev, engine = self._setup(
            monkeypatch, ["Arrival Date: UNREADABLE"])
        assert _run(pages, ev, engine) is None
        assert ev.arrival_on_intake is False

    def test_garbled_unreadable_is_still_abstain(self, flag_on, monkeypatch):
        pages, ev, engine = self._setup(
            monkeypatch, ["Arrival Date: UNREADAELE"])
        assert _run(pages, ev, engine) is None

    def test_unreadable_in_prior_read_vetoes_a_view_date(
            self, flag_on, monkeypatch):
        # The primary ladder already read the printed UNREADABLE marker; a
        # head view that now hallucinates a date must not override it.
        from mib_pipeline.model import Line
        prior = Line(text="Arrival Date: UNREADABLE", page_index=0,
                     source=Source.OCR, conf=0.8)
        pages, ev, engine = self._setup(
            monkeypatch, ["Arrival Date: 2026-05-04"], existing_lines=(prior,))
        assert _run(pages, ev, engine) is None

    def test_date_mismatching_anchor_never_fires(self, flag_on, monkeypatch):
        pages, ev, engine = self._setup(
            monkeypatch, ["Arrival Date: 2026-03-01"])
        assert _run(pages, ev, engine) is None

    def test_no_anchor_value_never_fires(self, flag_on, monkeypatch):
        pages, ev, engine = self._setup(
            monkeypatch, ["Arrival Date: 2026-05-04"], anchor=None)
        assert _run(pages, ev, engine) is None

    def test_disagreeing_views_never_fire(self, flag_on, monkeypatch):
        pages, ev, _ = self._setup(monkeypatch, [])
        engine = FakeEngine({
            ("i1", False): _words(["Arrival Date: 2026-05-04"]),
            ("i2", False): _words(["Arrival Date: 2026-03-01"]),
        })
        assert _run(pages, ev, engine) is None


# ------------------------------------------------- D-SOLEGAP visa/sponsor


class TestSoleGapHead:
    def test_sponsor_from_attestation_fires(self, flag_on, monkeypatch):
        pages = [_scan_page(0, "attestation")]
        _fake_views(monkeypatch, {0: [("raw", "a1"), ("otsu", "a2")]})
        engine = FakeEngine({(img, False): _words(["Sponsor ID: SPN-4732"])
                             for img in ("a1", "a2")})
        ev = _evidence(sponsor_id=None)
        fired = _run(pages, ev, engine)
        assert fired is not None
        assert (fired.head, fired.field, fired.value) == \
            ("solegap", "sponsor_id", "SPN-4732")
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("APPROVED", "R11_default_approve")

    def test_prose_attestation_sponsor_fires(self, flag_on, monkeypatch):
        prose = ["Sponsor Attestation Letter",
                 "This letter attests that Solul Zamora is expected",
                 "on Earth for research under SPN-4732 and class",
                 "XW-1 compliance."]
        pages = [_scan_page(0, "attestation")]
        _fake_views(monkeypatch, {0: [("raw", "a1"), ("otsu", "a2")]})
        engine = FakeEngine({(img, False): _words(prose)
                             for img in ("a1", "a2")})
        ev = _evidence(sponsor_id=None)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "SPN-4732"

    def test_conflicting_sponsor_pages_never_fire(self, flag_on, monkeypatch):
        pages = [_scan_page(0, "attestation"), _scan_page(1, "intake")]
        _fake_views(monkeypatch, {0: [("raw", "a1"), ("otsu", "a2")],
                                  1: [("raw", "b1"), ("otsu", "b2")]})
        engine = FakeEngine({
            ("a1", False): _words(["Sponsor ID: SPN-4732"]),
            ("a2", False): _words(["Sponsor ID: SPN-4732"]),
            ("b1", False): _words(["Sponsor ID: SPN-9911"]),
            ("b2", False): _words(["Sponsor ID: SPN-9911"]),
        })
        ev = _evidence(sponsor_id=None)
        assert _run(pages, ev, engine) is None

    def test_visa_exact_both_views_fires(self, flag_on, monkeypatch):
        pages = [_scan_page(0, "intake")]
        _fake_views(monkeypatch, {0: [("raw", "i1"), ("otsu", "i2")]})
        engine = FakeEngine({(img, False): _words(["Visa Class: MED-3"])
                             for img in ("i1", "i2")})
        ev = _evidence(visa_class=None)
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "MED-3"
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("APPROVED", "R11_default_approve")

    def test_fuzzy_visa_never_fires(self, flag_on, monkeypatch):
        # Exact token equality for EVERY class (§8.4): "DIP-l" and "XW-l"
        # style near-misses must never mint a visa, in either direction.
        pages = [_scan_page(0, "intake")]
        _fake_views(monkeypatch, {0: [("raw", "i1"), ("otsu", "i2")]})
        engine = FakeEngine({(img, False): _words(["Visa Class: DIP-l"])
                             for img in ("i1", "i2")})
        ev = _evidence(visa_class=None)
        assert _run(pages, ev, engine) is None

    def test_visa_views_disagreeing_on_class_never_fire(
            self, flag_on, monkeypatch):
        pages = [_scan_page(0, "intake")]
        _fake_views(monkeypatch, {0: [("raw", "i1"), ("otsu", "i2")]})
        engine = FakeEngine({
            ("i1", False): _words(["Visa Class: XW-1"]),
            ("i2", False): _words(["Visa Class: XW-2"]),
        })
        ev = _evidence(visa_class=None)
        assert _run(pages, ev, engine) is None


# --------------------------------------- shape-classifier integration (§9)


class TestShapeIntegration:
    def _verdict(self, value, conf=0.8):
        return discharge.ShapeVerdict(value=value, conf=conf,
                                      views=("shape-raw", "shape-divblur"))

    def test_shape_only_fire_approves_through_cascade(
            self, flag_on, monkeypatch):
        pages, ev, _ = _fee_setup(monkeypatch)
        monkeypatch.setattr(discharge, "_shape_verdict",
                            lambda *a, **k: self._verdict("paid"))
        fired = _run(pages, ev, FakeEngine({}))     # OCR reads nothing
        assert fired is not None and fired.value == "paid"
        assert "shape-raw" in fired.views and "shape-divblur" in fired.views
        label, path = policy.adjudicate(ev)
        assert (label, path) == ("APPROVED", "R11_default_approve")

    def test_shape_disagreeing_with_ocr_vetoes_both(
            self, flag_on, monkeypatch):
        # OCR would fire alone (exact unpaid); shape says paid -> abstain.
        lines = ["MIB Fee Receipt", "Fee Status: unpaid"]
        pages, ev, engine = _fee_setup(monkeypatch, receipt_lines=lines)
        monkeypatch.setattr(discharge, "_shape_verdict",
                            lambda *a, **k: self._verdict("paid"))
        assert _run(pages, ev, engine) is None

    def test_shape_agreeing_with_ocr_fires_with_merged_views(
            self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch)
        monkeypatch.setattr(discharge, "_shape_verdict",
                            lambda *a, **k: self._verdict("paid"))
        fired = _run(pages, ev, engine)
        assert fired is not None and fired.value == "paid"
        assert {"raw", "otsu", "shape-raw"} <= set(fired.views)

    def test_feeshape_kill_switch_disables_shape_only(
            self, flag_on, monkeypatch):
        monkeypatch.setenv("MIB_DISCHARGE_FEESHAPE", "0")
        pages, ev, engine = _fee_setup(monkeypatch)
        called = []
        monkeypatch.setattr(discharge, "_shape_verdict",
                            lambda *a, **k: called.append(1))
        fired = _run(pages, ev, engine)             # OCR path still fires
        assert fired is not None and called == []

    def test_shape_verdict_requires_both_variants_both_regions(
            self, flag_on, monkeypatch):
        from mib_pipeline import fee_shape

        crops = {"fee_status": "st", "fee_amount": "am", "waiver_code": "wv"}
        monkeypatch.setattr(
            discharge, "_shape_views",
            lambda *a, **k: [("shape-raw", crops), ("shape-divblur", crops)])
        reads = {
            "st": fee_shape.ShapeRead("paid", 0.9, 0.5),
            "am": fee_shape.ShapeRead("$809.00", 0.9, 0.5),
            "wv": fee_shape.ShapeRead("N/A", 0.9, 0.5),
        }
        monkeypatch.setattr(fee_shape, "classify_status",
                            lambda c: reads.get(c) if c == "st" else None)
        monkeypatch.setattr(fee_shape, "classify_amount",
                            lambda c: reads.get(c) if c == "am" else None)
        monkeypatch.setattr(fee_shape, "classify_waiver",
                            lambda c: reads.get(c) if c == "wv" else None)
        v = discharge._shape_verdict("/x.pdf", _scan_page(1, "fee_receipt"),
                                     None)
        assert v is not None and v.value == "paid"
        # amount abstains on one variant -> whole shape read abstains
        monkeypatch.setattr(fee_shape, "classify_amount", lambda c: None)
        assert discharge._shape_verdict(
            "/x.pdf", _scan_page(1, "fee_receipt"), None) is None

    def test_shape_verdict_unknown_is_no_fire(self, flag_on, monkeypatch):
        from mib_pipeline import fee_shape

        crops = {"fee_status": "st", "fee_amount": "am", "waiver_code": "wv"}
        monkeypatch.setattr(
            discharge, "_shape_views",
            lambda *a, **k: [("shape-raw", crops), ("shape-divblur", crops)])
        # paid claim with $0.00 and no waiver = inconsistent receipt.
        monkeypatch.setattr(fee_shape, "classify_status",
                            lambda c: fee_shape.ShapeRead("paid", 0.9, 0.5))
        monkeypatch.setattr(fee_shape, "classify_amount",
                            lambda c: fee_shape.ShapeRead("$0.00", 0.9, 0.5))
        monkeypatch.setattr(fee_shape, "classify_waiver", lambda c: None)
        assert discharge._shape_verdict(
            "/x.pdf", _scan_page(1, "fee_receipt"), None) is None


# ------------------------------------------------- T6: self-consistency


class TestT6SelfConsistency:
    def test_discharged_row_emits_the_accepted_value_and_rederives(
            self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch)
        fired = _run(pages, ev, engine)
        assert fired is not None
        label, path = policy.adjudicate(ev)
        adjudication, confidence = decision.decide(label, path)
        confidence = min(confidence, discharge.CONF_CAP)
        row = writer.build_row(CASE_ID, ev, adjudication, confidence)
        assert row["fee_status"] == fired.value == "paid"
        assert row["adjudication"] == "APPROVED"
        assert row["confidence"] <= discharge.CONF_CAP
        # the emitted row re-derives its own adjudication under the policy
        relabel, _ = policy.adjudicate(ev)
        assert relabel == row["adjudication"]

    def test_provenance_record_is_complete(self, flag_on, monkeypatch):
        pages, ev, engine = _fee_setup(monkeypatch)
        fired = _run(pages, ev, engine)
        rec = fired.as_dict()
        assert rec["head"] == "fee"
        assert rec["field"] == "fee_status"
        assert rec["value"] == "paid"
        assert rec["page"] == 1
        assert len(rec["views"]) >= 2


# --------------------------------------- pipeline wiring (real PDF, e2e)


def _make_scan_pdf(tmp_path, page_specs, case_id=CASE_ID):
    """Synthetic packet whose pages are full-page raster images (SCAN kind)
    with the genuine vector footer, exercising the real render+OCR path."""
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
RECEIPT_SPEC = [
    "MIB Fee Receipt",
    f"Case ID: {CASE_ID}",
    "Fee Status: paid",
    "Amount: $809.00",
    "Waiver Code: N/A",
]
BIOMETRIC_SPEC = [
    "FORM B-13: Biometric Scan Slip",
    f"Case ID: {CASE_ID}",
    "Applicant: Solul Zamora",
    "Species Match: ORION_GRAYS",
    "Observed flags: none",
]


class TestIntegrationPipeline:
    def test_flag_off_pipeline_is_inert_and_never_calls_discharge(
            self, flag_off, tmp_path, monkeypatch):
        from mib_pipeline import pipeline

        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC, RECEIPT_SPEC,
                                        BIOMETRIC_SPEC])
        calls = []
        real = discharge.run_discharge

        def spy(**kwargs):
            calls.append(1)
            return real(**kwargs)

        monkeypatch.setattr(discharge, "run_discharge", spy)
        row1 = pipeline.process_pdf(pdf)
        row2 = pipeline.process_pdf(pdf)
        assert calls == []
        assert "_discharge" not in row1
        assert row1 == row2

    def test_flag_on_end_to_end_fee_discharge_on_real_pixels(
            self, flag_on, tmp_path, monkeypatch):
        # Muzzle the primary fee resolution so the case presents
        # R8_fee_unread; the head must then recover the fee from its own
        # fresh 576-DPI band read of the real receipt pixels.
        from mib_pipeline import fields as f, pipeline

        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC, RECEIPT_SPEC,
                                        BIOMETRIC_SPEC])
        monkeypatch.setattr(f, "_resolve_fee", lambda candidates: (None, 0.0))
        row = pipeline.process_pdf(pdf)
        assert row["_path"] == "R11_default_approve"
        assert row["adjudication"] == "APPROVED"
        assert row["fee_status"] == "paid"
        assert row["confidence"] <= discharge.CONF_CAP
        assert row["_discharge"]["head"] == "fee"

    def test_flag_off_same_muzzle_stays_review(
            self, flag_off, tmp_path, monkeypatch):
        from mib_pipeline import fields as f, pipeline

        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC, RECEIPT_SPEC,
                                        BIOMETRIC_SPEC])
        monkeypatch.setattr(f, "_resolve_fee", lambda candidates: (None, 0.0))
        row = pipeline.process_pdf(pdf)
        assert row["adjudication"] == "NEEDS_REVIEW"
        assert row["_path"] == "R8_fee_unread"
        assert "_discharge" not in row
