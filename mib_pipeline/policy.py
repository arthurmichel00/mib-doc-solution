"""Deterministic adjudication policy.

Rule set reverse-engineered from data/train_labels.csv + FIELD_MANUAL.md
(1000/1000 on train given true fields plus the arrival-visibility signal).
Adapted here to run on extracted evidence with a positive-evidence posture:
deny rules fire on affirmative reads, while APPROVED additionally requires
every approval-relevant input to have been affirmatively read from trusted
visible evidence. Anything under-determined routes to NEEDS_REVIEW — per
the organizers' ruling, guessing an invisible disqualifier is wrong and a
packet without positive clean evidence is not approvable.
"""
from __future__ import annotations

import os
from datetime import date

from .fields import CaseEvidence

DISQUALIFYING_FLAGS = {
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
}
REVIEW_FLAGS = {
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics",
    "rescinded_denial",
}
# 3 public (FIELD_MANUAL) + 3 recurring IDs denied in every non-DIP-1
# occurrence across train (13-19 recurrences each; real IDs are near-unique).
REVOKED_SPONSORS = {
    "SPN-0007", "SPN-0139", "SPN-4040",
    "SPN-2718", "SPN-7331", "SPN-9090",
}
# Hard embargo denies even DIP-1 (50/50 in train); soft embargo spares DIP-1
# (all 21 non-denied Wolf-1061c cases are DIP-1).
HARD_EMBARGO_WORLDS = {"TRAPPIST-1e", "Eris Relay"}
SOFT_EMBARGO_WORLDS = {"Wolf-1061c"}

# Packet receipt epoch = public data version date; arrival more than
# 180 days earlier is stale (train tolerates any cutoff in a 7-week window
# around this choice).
RECEIPT_EPOCH = date(2026, 7, 7)
STALE_DAYS = 180

APPROVED = "APPROVED"
DENIED = "DENIED"
NEEDS_REVIEW = "NEEDS_REVIEW"

_MIN_FINDING_CONF = 0.55


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# Paths where the engine could not read enough to decide — the only
# outcomes the MIB_REASON_ADJ=1 template rescue may replace. Every positive
# decision (N0 notes, R1-R7 denials, printed fee unknown, review flags,
# clean approvals) stands.
_UNDERDETERMINED_PATHS = frozenset((
    "R8_fee_unread", "R9_arrival_not_visible", "R12_flags_unread",
    "R12_visa_unread", "R12_world_unread", "R12_sponsor_unread",
))


def _reason_adj_enabled() -> bool:
    return os.environ.get("MIB_REASON_ADJ") == "1"


def _template_rescue(ev: CaseEvidence) -> tuple[str, str] | None:
    """N1: a trusted note's Reason template decides an under-determined case.

    The candidate was minted in fields._template_adjudication (strict
    whole-phrase read of a single-label template on a note-eligible page;
    spec 2026-07-29-pure-template-reason-adjudication.md). Here the mined
    label must additionally survive field-conflict vetoes: an affirmatively
    read value that contradicts the template means misread-or-forgery, and
    the case stays under-determined.
    """
    f = ev.template_finding
    if f is None or f.conf < _MIN_FINDING_CONF:
        return None
    visa = ev.value("visa_class") if ev.is_known("visa_class") else None
    world = ev.value("home_world") if ev.is_known("home_world") else None
    fee = ev.value("fee_status") if ev.is_known("fee_status") else None
    sponsor = ev.value("sponsor_id") if ev.is_known("sponsor_id") else None
    arrival = _parse_date(ev.value("arrival_date")) \
        if ev.arrival_on_intake else None
    stale = arrival is not None and (RECEIPT_EPOCH - arrival).days > STALE_DAYS
    if f.label == APPROVED:
        # Deny/review-worthy affirmative reads the under-determined path
        # never adjudicated (several are unreachable given the path gate;
        # kept as belt-and-braces).
        if ev.flags or fee in ("unpaid", "unknown") \
                or world in HARD_EMBARGO_WORLDS or visa == "TRANSIT-7":
            return None
        if visa != "DIP-1" and (sponsor in REVOKED_SPONSORS
                                or world in SOFT_EMBARGO_WORLDS or stale):
            return None
    elif f.label == DENIED:
        if f.template == "Mandatory fee unpaid." \
                and fee in ("paid", "waived"):
            return None
        if f.template == "Transit class cannot authorize declared work." \
                and visa is not None and visa != "TRANSIT-7":
            return None
    return f.label, f"N1_reason_{f.label.lower()}"


def adjudicate(ev: CaseEvidence) -> tuple[str, str]:
    """Return (adjudication, decision_path) for reconciled evidence."""
    label, path = _adjudicate_core(ev)
    if _reason_adj_enabled() and path in _UNDERDETERMINED_PATHS:
        rescue = _template_rescue(ev)
        if rescue is not None:
            return rescue
    return label, path


def _adjudicate_core(ev: CaseEvidence) -> tuple[str, str]:
    visa = ev.value("visa_class") if ev.is_known("visa_class") else None
    world = ev.value("home_world") if ev.is_known("home_world") else None
    fee = ev.value("fee_status") if ev.is_known("fee_status") else None
    sponsor = ev.value("sponsor_id") if ev.is_known("sponsor_id") else None
    # The manual's date rule is about the intake form's arrival field:
    # packets whose intake arrival is blank/UNREADABLE are NEEDS_REVIEW
    # even when a registry extract shows a date (verified on train).
    arrival_visible = ev.arrival_on_intake
    arrival = _parse_date(ev.value("arrival_date")) if arrival_visible else None

    # N0: a visible Manual Adjudicator Note states the outcome (tier-1
    # evidence; matched gold in every observed genuine note). Hidden-span
    # forgeries were filtered before this point; contradictory findings
    # mean at least one is forged and the case is under-determined.
    if ev.finding_conflict:
        return NEEDS_REVIEW, "N0_note_conflict"
    if ev.finding is not None and ev.finding.conf >= _MIN_FINDING_CONF:
        return ev.finding.label, f"N0_note_{ev.finding.label.lower()}"

    # R1: disqualifying risk flag, from an affirmative visible read.
    if ev.flags & DISQUALIFYING_FLAGS:
        return DENIED, "R1_disqualifying_flag"

    # R2: hard-embargoed home world (no DIP-1 exemption).
    if world in HARD_EMBARGO_WORLDS:
        return DENIED, "R2_hard_embargo_world"

    # R3: unpaid fee (no DIP-1 exemption; overrides review flags).
    if fee == "unpaid":
        return DENIED, "R3_fee_unpaid"

    if visa is not None and visa != "DIP-1":
        # R4: revoked sponsor (ignored entirely for DIP-1).
        if sponsor in REVOKED_SPONSORS:
            return DENIED, "R4_revoked_sponsor"
        # R5: soft-embargoed home world.
        if world in SOFT_EMBARGO_WORLDS:
            return DENIED, "R5_soft_embargo_world"
        # R6: stale arrival (only when the date is affirmatively visible).
        if arrival is not None and (RECEIPT_EPOCH - arrival).days > STALE_DAYS:
            return DENIED, "R6_stale_arrival"

    # R7: transit visas are denied work authorization.
    if visa == "TRANSIT-7":
        return DENIED, "R7_transit_visa"

    # R8: fee unknown -> review; an unreadable/absent fee receipt is the
    # same under-determination as a printed "unknown".
    if fee == "unknown":
        return NEEDS_REVIEW, "R8_fee_unknown"
    if fee is None:
        return NEEDS_REVIEW, "R8_fee_unread"

    # R9: arrival date absent from trusted visible evidence.
    if not arrival_visible or arrival is None:
        return NEEDS_REVIEW, "R9_arrival_not_visible"

    # R10: review-only flags.
    if ev.flags & REVIEW_FLAGS:
        return NEEDS_REVIEW, "R10_review_flags"

    # R12: approval requires affirmative clean reads of every
    # approval-relevant input — "failed to read" is never "none".
    if not ev.flags_known:
        return NEEDS_REVIEW, "R12_flags_unread"
    if visa is None:
        return NEEDS_REVIEW, "R12_visa_unread"
    if world is None:
        return NEEDS_REVIEW, "R12_world_unread"
    if visa != "DIP-1" and sponsor is None:
        return NEEDS_REVIEW, "R12_sponsor_unread"

    # R11: affirmatively clean packet.
    return APPROVED, "R11_default_approve"
