"""Expected-value decision layer over the policy output.

Builds a posterior over {APPROVED, DENIED, NEEDS_REVIEW} from the decision
path's calibrated accuracy and picks the adjudication maximizing expected
classification points under the official payoff matrix:

    E[APPROVED]     = 8 pA - 4 pD + pN
    E[DENIED]       =        8 pD + pN
    E[NEEDS_REVIEW] = 2 (pA + pD) + 8 pN

Under-determined paths are pinned to NEEDS_REVIEW upstream of EV (see
calibration.py). Confidence is the posterior of the chosen class.
"""
from __future__ import annotations

from .calibration import (APPROVED, DENIED, NEEDS_REVIEW, clamp_confidence,
                          path_stats)

_CLASSES = (APPROVED, DENIED, NEEDS_REVIEW)


def posterior(policy_label: str, path: str) -> dict[str, float]:
    stats = path_stats(path)
    probs = {cls: 0.0 for cls in _CLASSES}
    probs[policy_label] = stats.accuracy
    residual = 1.0 - stats.accuracy
    split = {c: s for c, s in stats.error_split.items() if c != policy_label}
    total = sum(split.values())
    for cls, share in split.items():
        probs[cls] += residual * (share / total if total else 0.5)
    return probs


def expected_points(probs: dict[str, float]) -> dict[str, float]:
    pa, pd, pn = probs[APPROVED], probs[DENIED], probs[NEEDS_REVIEW]
    return {
        APPROVED: 8 * pa - 4 * pd + pn,
        DENIED: 8 * pd + pn,
        NEEDS_REVIEW: 2 * (pa + pd) + 8 * pn,
    }


def decide(policy_label: str, path: str) -> tuple[str, float]:
    """Return (final adjudication, calibrated confidence)."""
    stats = path_stats(path)
    probs = posterior(policy_label, path)
    if stats.pinned is not None:
        choice = stats.pinned
    else:
        ev = expected_points(probs)
        choice = max(_CLASSES, key=lambda cls: ev[cls])
    return choice, clamp_confidence(probs[choice])
