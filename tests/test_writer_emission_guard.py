"""Emission guard: APPROVED rows must be self-consistent under policy.

Only a trusted-note approval can carry a deny-triggering field value (R12
pins every field-based approval to affirmatively clean inputs), and on such
a row the deterministic policy makes the value guaranteed-wrong when the
note is genuine — so the writer swaps it for the established unread
fallback. Adjudications, paths, and confidences must never change; DIP-1
exemptions (revoked sponsor / Wolf-1061c / stale arrival) must be kept.
"""
from __future__ import annotations

from mib_pipeline.fields import CaseEvidence
from mib_pipeline.writer import build_row


def _evidence(flags: set[str] | None = None, **values: str) -> CaseEvidence:
    base = {
        "applicant_name": "Aridane Zavoss",
        "species_code": "ARCTURIAN",
        "home_world": "Proxima-b",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-4560",
        "arrival_date": "2026-04-29",
        "declared_purpose": "research",
        "fee_status": "paid",
    }
    base.update(values)
    ev = CaseEvidence(values=dict(base),
                      known={fld: True for fld in base},
                      conf={fld: 0.9 for fld in base})
    ev.flags = set(flags or ())
    ev.flags_known = True
    return ev


def _approved(ev: CaseEvidence) -> dict:
    return build_row("MIB-000000", ev, "APPROVED", 0.967)


def test_adjudication_and_confidence_pass_through_unchanged():
    row = _approved(_evidence(fee_status="unpaid", visa_class="TRANSIT-7"))
    assert row["adjudication"] == "APPROVED"
    assert row["confidence"] == 0.967


def test_unpaid_fee_suppressed_on_approved():
    row = _approved(_evidence(fee_status="unpaid"))
    assert row["fee_status"] == "paid"


def test_unknown_fee_suppressed_on_approved():
    row = _approved(_evidence(fee_status="unknown"))
    assert row["fee_status"] == "paid"


def test_transit7_visa_suppressed_on_approved():
    row = _approved(_evidence(visa_class="TRANSIT-7"))
    assert row["visa_class"] == "MED-3"


def test_revoked_sponsor_suppressed_on_non_dip_approved():
    row = _approved(_evidence(visa_class="XW-1", sponsor_id="SPN-4040"))
    assert row["sponsor_id"] == "SPN-0000"


def test_transit7_with_revoked_sponsor_cascades():
    # After TRANSIT-7 falls back to MED-3 the row is still non-DIP, so the
    # revoked sponsor must be suppressed in the same pass.
    row = _approved(_evidence(visa_class="TRANSIT-7", sponsor_id="SPN-2718"))
    assert row["visa_class"] == "MED-3"
    assert row["sponsor_id"] == "SPN-0000"


def test_risk_flags_suppressed_on_approved():
    row = _approved(_evidence(flags={"sponsor_mismatch"}))
    assert row["risk_flags"] == "none"


def test_hard_embargo_world_suppressed_even_for_dip1():
    # Includes the upstream-inferred planetary_embargo flag.
    row = _approved(_evidence(flags={"planetary_embargo"},
                              visa_class="DIP-1", home_world="TRAPPIST-1e"))
    assert row["home_world"] == "Luyten-b"
    assert row["risk_flags"] == "none"


def test_soft_embargo_world_and_stale_date_suppressed_on_non_dip():
    row = _approved(_evidence(visa_class="XW-2", home_world="Wolf-1061c",
                              arrival_date="2025-06-15"))
    assert row["home_world"] == "Luyten-b"
    assert row["arrival_date"] == "2026-04-14"


def test_dip1_keeps_revoked_sponsor():
    # R4 keep-list: 17 train approvals carry a genuinely revoked sponsor
    # under DIP-1 (live case MIB-000405 emits SPN-2718 via R11).
    row = _approved(_evidence(visa_class="DIP-1", sponsor_id="SPN-2718"))
    assert row["sponsor_id"] == "SPN-2718"


def test_dip1_keeps_wolf_world_and_stale_arrival():
    row = _approved(_evidence(visa_class="DIP-1", home_world="Wolf-1061c",
                              arrival_date="2025-06-15"))
    assert row["home_world"] == "Wolf-1061c"
    assert row["arrival_date"] == "2025-06-15"


def test_unpaid_fee_suppressed_even_for_dip1():
    # R3 has no DIP-1 exemption (16/16 unpaid DIP-1 are DENIED).
    row = _approved(_evidence(visa_class="DIP-1", fee_status="unpaid"))
    assert row["fee_status"] == "paid"


def test_clean_approved_row_is_untouched():
    ev = _evidence()
    expected = build_row("MIB-000000", _evidence(), "APPROVED", 0.938)
    assert build_row("MIB-000000", ev, "APPROVED", 0.938) == expected
    assert expected["sponsor_id"] == "SPN-4560"
    assert expected["fee_status"] == "paid"


def test_denied_row_keeps_deny_triggering_values():
    ev = _evidence(visa_class="TRANSIT-7", sponsor_id="SPN-4040",
                   fee_status="unpaid", flags={"biohazard_red"})
    row = build_row("MIB-000000", ev, "DENIED", 0.97)
    assert row["visa_class"] == "TRANSIT-7"
    assert row["sponsor_id"] == "SPN-4040"
    assert row["fee_status"] == "unpaid"
    assert row["risk_flags"] == "biohazard_red"


def test_needs_review_row_keeps_values():
    ev = _evidence(fee_status="unknown", flags={"identity_conflict"})
    row = build_row("MIB-000000", ev, "NEEDS_REVIEW", 0.5)
    assert row["fee_status"] == "unknown"
    assert row["risk_flags"] == "identity_conflict"
