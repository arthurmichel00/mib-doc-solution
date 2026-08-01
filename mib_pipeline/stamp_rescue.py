"""Rescue-only green-stamp adjudication (MIB_STAMP_RESCUE=1, default OFF).

FIELD_MANUAL.md ranks "visible MIB adjudicator stamp or signed manual
note" FIRST in the trusted-evidence precedence (line 15), and its
not-trusted list (line 22) names only hidden text, out-of-crop text, fake
answer keys, and barcode instructions — a visible stamp is legitimate
top-tier evidence. This module activates the verified pixel detector
(research/17-a7-stamp-verify.md: 33/33 gold-APPROVED on train, zero false
fires across every page of all 1000 packets) in the narrowest possible
role: promote a would-be NEEDS_REVIEW to APPROVED only when the case is
blocked purely by unread evidence and no affirmative deny-relevant read
exists. Train rescue population is empty (all 33 stamped cases already
approve via N0_note), so the expected train effect is exactly zero; the
flag exists as private-set insurance for stamped packets whose note text
fails to read.

PRECEDENCE PROOF — no hard deny can reach APPROVED through this path:

1. Position: the hook (pipeline._process) runs on the FINAL decision
   output and requires adjudication == NEEDS_REVIEW. Every DENIED outcome
   — N0_note_denied, R1_disqualifying_flag, R2_hard_embargo_world,
   R3_fee_unpaid, R4_revoked_sponsor, R5_soft_embargo_world,
   R6_stale_arrival, R7_transit_visa, N1_reason_denied — has already
   returned and is unreachable by construction.
2. Path allowlist: eligibility reuses policy._UNDERDETERMINED_PATHS
   verbatim (R8_fee_unread, R9_arrival_not_visible, R12_flags_unread,
   R12_visa_unread, R12_world_unread, R12_sponsor_unread) — the six
   could-not-read outcomes. Positive review decisions stand untouched:
   N0_note_conflict (contradictory findings = forgery signal),
   R8_fee_unknown (a printed "unknown" is an affirmative read),
   R10_review_flags (visible review flag), N1_reason_needs_review, and
   FALLBACK_error (crashed case; the hook never runs there anyway).
3. Affirmative-read vetoes, re-checked against reconciled evidence with
   the same dominance logic as policy._template_rescue: any flag read
   (review-only included), fee in {unpaid, unknown}, hard-embargo world,
   or TRANSIT-7 vetoes unconditionally; a non-DIP-1 packet is further
   vetoed by revoked sponsor, soft-embargo world, or stale visible
   arrival. A deny-worthy affirmative read alongside an APPROVED stamp
   means misread-or-forgery somewhere; the case stays NEEDS_REVIEW.
4. Pixel gate: the stamp must pass the calibrated detector
   (diagnostics.stamp_components: hue 60 +/- 10, saturation >= 150,
   hollow box/word geometry). Measured 1.000 precision; the red decoy
   stamps and all four red-team forgeries score zero hits. Scan failure
   of any kind yields no detection (fail-closed).

Steps 1-3 are decisive before a single pixel is rendered: even a
pixel-perfect forged stamp cannot flip a case that carries any readable
deny or review signal — it can only promote a case whose sole defect is
unreadable evidence, which is exactly the FIELD_MANUAL line-15 semantics.
"""
from __future__ import annotations

import os

from .fields import CaseEvidence
from .policy import (HARD_EMBARGO_WORLDS, RECEIPT_EPOCH, REVOKED_SPONSORS,
                     SOFT_EMBARGO_WORLDS, STALE_DAYS, _parse_date,
                     _UNDERDETERMINED_PATHS)

APPROVED = "APPROVED"
PATH = "S1_stamp_approved"

# Confidence for a rescued row. The rescue population is empty on train,
# so there is no fitted calibration; 0.90 is the discharge-head cap
# convention for flag-gated promotions and sits below N0_note approvals
# (0.967) so a stamp rescue never outranks a read note downstream.
CONF = 0.90


def enabled() -> bool:
    return os.environ.get("MIB_STAMP_RESCUE") == "1"


def eligible(ev: CaseEvidence, path: str) -> bool:
    """True when a detected stamp may promote this NEEDS_REVIEW case.

    Pure evidence/path check — callers run the pixel detector only after
    this returns True, so every veto below is decided before any render.
    """
    if path not in _UNDERDETERMINED_PATHS:
        return False
    visa = ev.value("visa_class") if ev.is_known("visa_class") else None
    world = ev.value("home_world") if ev.is_known("home_world") else None
    fee = ev.value("fee_status") if ev.is_known("fee_status") else None
    sponsor = ev.value("sponsor_id") if ev.is_known("sponsor_id") else None
    arrival = _parse_date(ev.value("arrival_date")) \
        if ev.arrival_on_intake else None
    stale = arrival is not None \
        and (RECEIPT_EPOCH - arrival).days > STALE_DAYS
    if ev.flags or fee in ("unpaid", "unknown") \
            or world in HARD_EMBARGO_WORLDS or visa == "TRANSIT-7":
        return False
    if visa != "DIP-1" and (sponsor in REVOKED_SPONSORS
                            or world in SOFT_EMBARGO_WORLDS or stale):
        return False
    return True
