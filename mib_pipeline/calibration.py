"""Per-decision-path confidence calibration.

Confidence must be P(chosen adjudication == truth label): the Brier-optimal
emission. Each decision path carries an empirical accuracy plus the split
of its residual error mass over the other two classes; the decision layer
turns these into a posterior.

The v1 numbers are priors from the policy-mining research (per-rule support
on train); fit_calibration replaces them with shrunk out-of-fold estimates.
Paths whose evidence is under-determined are pinned to NEEDS_REVIEW: the
organizers ruled that unrecoverable disqualifiers must not be guessed, so
these paths are never surrendered to train-label expected value (train
labels still say DENIED on ~99 such cases by design).
"""
from __future__ import annotations

from dataclasses import dataclass, field

APPROVED = "APPROVED"
DENIED = "DENIED"
NEEDS_REVIEW = "NEEDS_REVIEW"

CONF_FLOOR = 0.05
CONF_CEIL = 0.97


@dataclass(frozen=True)
class PathStats:
    accuracy: float
    # residual error mass split over the two non-chosen classes
    error_split: dict[str, float] = field(default_factory=dict)
    pinned: str | None = None    # forced adjudication, ignoring train EV


_DEFAULT = PathStats(accuracy=0.5, error_split={DENIED: 0.6, APPROVED: 0.4},
                     pinned=NEEDS_REVIEW)

# Fitted on the full 1000-case IN-CONTAINER serve-OCR dump (A9 refit,
# 2026-07-31, tools/fit_calibration.py on preds_incontainer.jsonl.dev,
# shrinkage m=10 toward global accuracy 0.770). Out-of-fold check with
# seed 8090 / 200 holdout: accuracy 0.805, mean Brier 0.0759.
# Error splits with no observed errors keep conservative priors. Hard-embargo
# worlds surface as R1 (the planetary_embargo flag is inferred upstream).
PATH_STATS: dict[str, PathStats] = {
    # contradictory findings never occur in train (n=0); prior only
    "N0_note_conflict": PathStats(0.5, {DENIED: 0.6, APPROVED: 0.4},
                                  pinned=NEEDS_REVIEW),
    "N0_note_approved": PathStats(0.972, {NEEDS_REVIEW: 0.7, DENIED: 0.3}),
    "N0_note_denied": PathStats(0.981, {NEEDS_REVIEW: 1.0}),
    "N0_note_needs_review": PathStats(0.982, {DENIED: 0.6, APPROVED: 0.4}),
    # Reason-template rescue paths, reachable only under MIB_REASON_ADJ=1
    # (spec 2026-07-29-pure-template-reason-adjudication.md). Priors mirror
    # the N0 note rows minus an OCR-read haircut; basis: template->gold
    # purity 162/162 digital, 0 label-inconsistent accepts on 118k real
    # scan lines. Refit with the in-container calibration pass (A9) if the
    # flag ships.
    "N1_reason_approved": PathStats(0.828, {NEEDS_REVIEW: 0.7, DENIED: 0.3}),
    "N1_reason_denied": PathStats(0.96, {NEEDS_REVIEW: 0.8, APPROVED: 0.2}),
    "N1_reason_needs_review": PathStats(0.828, {DENIED: 0.6, APPROVED: 0.4}),
    "R1_disqualifying_flag": PathStats(0.976, {NEEDS_REVIEW: 0.8, APPROVED: 0.2}),
    "R3_fee_unpaid": PathStats(0.921, {NEEDS_REVIEW: 1.0}),
    "R4_revoked_sponsor": PathStats(0.931, {APPROVED: 0.5, NEEDS_REVIEW: 0.5}),
    "R5_soft_embargo_world": PathStats(0.894, {NEEDS_REVIEW: 0.7, APPROVED: 0.3}),
    "R6_stale_arrival": PathStats(0.892, {NEEDS_REVIEW: 1.0}),
    "R7_transit_visa": PathStats(0.899, {NEEDS_REVIEW: 1.0}),
    "R8_fee_unknown": PathStats(0.944, {DENIED: 1.0}),
    "R8_fee_unread": PathStats(0.400, {APPROVED: 0.73, DENIED: 0.27},
                               pinned=NEEDS_REVIEW),
    "R9_arrival_not_visible": PathStats(0.679, {APPROVED: 0.44, DENIED: 0.56},
                                        pinned=NEEDS_REVIEW),
    "R10_review_flags": PathStats(0.902, {DENIED: 1.0}),
    "R11_default_approve": PathStats(0.949, {DENIED: 1.0}),
    "R12_flags_unread": PathStats(0.283, {APPROVED: 0.79, DENIED: 0.21},
                                  pinned=NEEDS_REVIEW),
    "R12_visa_unread": PathStats(0.648, {APPROVED: 1.0},
                                 pinned=NEEDS_REVIEW),
    "R12_world_unread": PathStats(0.5, {DENIED: 0.6, APPROVED: 0.4},
                                  pinned=NEEDS_REVIEW),
    "R12_sponsor_unread": PathStats(0.648, {APPROVED: 1.0},
                                    pinned=NEEDS_REVIEW),
    "FALLBACK_error": PathStats(0.35, {DENIED: 0.6, APPROVED: 0.4},
                                pinned=NEEDS_REVIEW),
}


def path_stats(path: str) -> PathStats:
    return PATH_STATS.get(path, _DEFAULT)


def clamp_confidence(value: float) -> float:
    return min(CONF_CEIL, max(CONF_FLOOR, value))
