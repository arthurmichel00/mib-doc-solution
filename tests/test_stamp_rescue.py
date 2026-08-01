"""Rescue-only green-stamp adjudication (MIB_STAMP_RESCUE=1, default OFF).

The rescue may promote NEEDS_REVIEW -> APPROVED only for the six
could-not-read policy paths, only when no affirmative deny-relevant read
exists, and only when the calibrated pixel detector finds the green
APPROVED stamp. Every deny path is unreachable by construction (the hook
requires final adjudication == NEEDS_REVIEW); these tests pin the path
allowlist, the veto layer, the flag gate, and the end-to-end pipeline
semantics on synthetic all-digital packets (no OCR engines involved).
"""
from __future__ import annotations

import json

import pymupdf
import pytest

from mib_pipeline import policy, stamp_rescue
from mib_pipeline.fields import CaseEvidence
from mib_pipeline.pipeline import process_pdf

_GREEN = (19 / 255, 137 / 255, 19 / 255)


# --------------------------------------------------------------------------
# fixtures

@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MIB_STAMP_RESCUE", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("MIB_STAMP_RESCUE", raising=False)


@pytest.fixture(autouse=True)
def _diag_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("MIB_STAMP_DIAG_FILE", str(tmp_path / "diag.log"))


def _evidence(**overrides) -> CaseEvidence:
    """Evidence that would reach R11 unless a field is knocked out.

    Pass fld=None to knock a field out (unread); flags/arrival_on_intake
    accepted as keywords.
    """
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
    ev.flags = set(overrides.get("flags", ()))
    ev.flags_known = overrides.get("flags_known", True)
    ev.arrival_on_intake = overrides.get("arrival_on_intake", True)
    return ev


def _adjudicated(ev: CaseEvidence) -> str:
    """Policy path for the fixture — keeps fixtures honest vs the engine."""
    _, path = policy.adjudicate(ev)
    return path


# --------------------------------------------------------------------------
# flag gate

class TestFlag:
    def test_unset_is_disabled(self, flag_off):
        assert not stamp_rescue.enabled()

    def test_zero_is_disabled(self, monkeypatch):
        monkeypatch.setenv("MIB_STAMP_RESCUE", "0")
        assert not stamp_rescue.enabled()

    def test_one_enables(self, flag_on):
        assert stamp_rescue.enabled()


# --------------------------------------------------------------------------
# eligibility: path allowlist

class TestPathAllowlist:
    def test_all_six_unread_paths_are_eligible_when_clean(self):
        knockouts = [
            dict(fee_status=None),                        # R8_fee_unread
            dict(arrival_on_intake=False),                # R9_arrival_not_visible
            dict(flags_known=False),                      # R12_flags_unread
            dict(visa_class=None),                        # R12_visa_unread
            dict(home_world=None),                        # R12_world_unread
            dict(sponsor_id=None),                        # R12_sponsor_unread
        ]
        seen = set()
        for kw in knockouts:
            ev = _evidence(**kw)
            path = _adjudicated(ev)
            seen.add(path)
            assert stamp_rescue.eligible(ev, path), path
        assert seen == set(policy._UNDERDETERMINED_PATHS)

    def test_printed_fee_unknown_is_a_positive_decision_not_eligible(self):
        ev = _evidence(fee_status="unknown")
        path = _adjudicated(ev)
        assert path == "R8_fee_unknown"
        assert not stamp_rescue.eligible(ev, path)

    def test_review_flags_path_is_not_eligible(self):
        ev = _evidence(flags={"sponsor_mismatch"})
        path = _adjudicated(ev)
        assert path == "R10_review_flags"
        assert not stamp_rescue.eligible(ev, path)

    def test_note_conflict_is_not_eligible(self):
        ev = _evidence()
        ev.finding_conflict = True
        path = _adjudicated(ev)
        assert path == "N0_note_conflict"
        assert not stamp_rescue.eligible(ev, path)

    def test_fallback_path_is_not_eligible(self):
        assert not stamp_rescue.eligible(CaseEvidence(), "FALLBACK_error")


# --------------------------------------------------------------------------
# eligibility: affirmative-read vetoes (belt-and-braces: tested directly
# against the function even where policy ordering makes the combination
# unreachable)

class TestVetoes:
    def test_review_only_flag_vetoes_a_reachable_fee_unread_case(self):
        # R8_fee_unread fires BEFORE R10, so this combination is live.
        ev = _evidence(fee_status=None, flags={"sponsor_mismatch"})
        path = _adjudicated(ev)
        assert path == "R8_fee_unread"
        assert not stamp_rescue.eligible(ev, path)

    def test_review_only_flag_vetoes_arrival_not_visible(self):
        ev = _evidence(arrival_on_intake=False, flags={"identity_conflict"})
        path = _adjudicated(ev)
        assert path == "R9_arrival_not_visible"
        assert not stamp_rescue.eligible(ev, path)

    @pytest.mark.parametrize("kw", [
        dict(flags={"memory_tampering"}),
        dict(fee_status="unpaid"),
        dict(fee_status="unknown"),
        dict(home_world="TRAPPIST-1e"),
        dict(visa_class="TRANSIT-7"),
        dict(sponsor_id="SPN-0007"),
        dict(home_world="Wolf-1061c"),
        dict(arrival_date="2025-01-01"),      # stale vs 2026-07-07 epoch
    ])
    def test_each_deny_signal_vetoes_even_on_an_allowlisted_path(self, kw):
        ev = _evidence(**kw)
        assert not stamp_rescue.eligible(ev, "R12_flags_unread")

    def test_dip1_exemption_matches_r4_semantics(self):
        # Revoked sponsor is ignored entirely for DIP-1 (policy R4);
        # the veto layer must not be stricter than the deny rule it guards.
        ev = _evidence(visa_class="DIP-1", sponsor_id="SPN-0007",
                       fee_status=None)
        path = _adjudicated(ev)
        assert path == "R8_fee_unread"
        assert stamp_rescue.eligible(ev, path)


# --------------------------------------------------------------------------
# pipeline integration on synthetic all-digital packets (fast: no OCR)

_INTAKE_ROWS = (
    "FORM I-8090: Extraterrestrial Work Authorization Intake",
    "Case ID: MIB-000123",
    "Applicant: Solul Zamora",
    "Species Code: ORION_GRAYS",
    "Home World: Proxima-b",
    "Visa Class: XW-1",
    "Sponsor ID: SPN-1234",
    "Arrival Date: 2026-05-04",
    "Declared Purpose: research",
)
_BIOMETRIC_ROWS = (
    "FORM B-13: Biometric Scan Slip",
    "Case ID: MIB-000123",
    "Applicant: Solul Zamora",
    "Species Match: ORION_GRAYS",
    "Observed flags: none",
)


def _packet(tmp_path, extra_rows=(), stamp=False,
            flags_row="Observed flags: none"):
    """All-digital two-page packet; fee receipt intentionally absent."""
    doc = pymupdf.open()
    biometric = tuple(flags_row if r.startswith("Observed flags") else r
                      for r in _BIOMETRIC_ROWS)
    for rows in (_INTAKE_ROWS + tuple(extra_rows), biometric):
        page = doc.new_page(width=612, height=792)
        y = 72
        for row in rows:
            page.insert_text((72, y), row, fontsize=11, color=(0, 0, 0))
            y += 18
        if stamp and rows[0].startswith("FORM I-8090"):
            rect = pymupdf.Rect(360, 500, 360 + 163, 500 + 70)
            page.draw_rect(rect, color=_GREEN, width=2.5)
            page.insert_text((rect.x0 + 26, rect.y0 + 45), "APPROVED",
                             fontsize=26, color=_GREEN)
    path = tmp_path / "MIB-000123.pdf"
    doc.save(str(path))
    return str(path)


class TestPipelineRescue:
    def test_fixture_reaches_fee_unread_review(self, tmp_path, flag_off):
        row = process_pdf(_packet(tmp_path))
        assert row["adjudication"] == "NEEDS_REVIEW"
        assert row["_path"] == "R8_fee_unread"

    def test_flag_off_stamped_case_stays_needs_review(self, tmp_path,
                                                      flag_off):
        row = process_pdf(_packet(tmp_path, stamp=True))
        assert row["adjudication"] == "NEEDS_REVIEW"
        assert row["_path"] == "R8_fee_unread"

    def test_flag_on_stamped_eligible_case_is_rescued(self, tmp_path,
                                                      flag_on, capsys):
        row = process_pdf(_packet(tmp_path, stamp=True))
        assert row["adjudication"] == "APPROVED"
        assert row["confidence"] == stamp_rescue.CONF
        assert row["_path"] == stamp_rescue.PATH
        err = capsys.readouterr().err
        assert "stamp detected" in err and "rescued" in err

    def test_flag_on_unstamped_case_stays_needs_review(self, tmp_path,
                                                       flag_on):
        row = process_pdf(_packet(tmp_path))
        assert row["adjudication"] == "NEEDS_REVIEW"
        assert row["_path"] == "R8_fee_unread"

    def test_forged_stamp_on_deny_worthy_packet_never_rescues(self, tmp_path,
                                                              flag_on):
        # The new red-team case: readable unpaid fee + pixel-perfect green
        # stamp. Deny paths dominate: the hook never sees NEEDS_REVIEW.
        row = process_pdf(_packet(
            tmp_path, extra_rows=("Fee Status: unpaid",), stamp=True))
        assert row["adjudication"] == "DENIED"
        assert row["_path"] == "R3_fee_unpaid"

    def test_forged_stamp_with_review_flag_is_vetoed(self, tmp_path, flag_on):
        # Reachable veto: fee unread (allowlisted path) + a read review
        # flag. The stamp must not discharge a positive review signal.
        row = process_pdf(_packet(
            tmp_path, stamp=True,
            flags_row="Observed flags: sponsor_mismatch"))
        assert row["adjudication"] == "NEEDS_REVIEW"

    def test_rescued_row_is_schema_clean(self, tmp_path, flag_on):
        row = dict(process_pdf(_packet(tmp_path, stamp=True)))
        row.pop("_path", None)
        # emitted rows must stay schema-only: no rescue artifacts leak
        assert set(row) == {
            "case_id", "applicant_name", "species_code", "home_world",
            "visa_class", "sponsor_id", "arrival_date", "declared_purpose",
            "risk_flags", "fee_status", "adjudication", "confidence"}
        json.dumps(row)  # serializable exactly like solution.py writes it
