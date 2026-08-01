"""tools/rescore.py path->label replay, incl. the N1 template-rescue paths.

Without the N1_ branch, an N1_reason_* row replays as NEEDS_REVIEW and
corrupts any calibration-refit gate run over a MIB_REASON_ADJ=1 dump
(final-rebuild runbook §2e).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mib_pipeline import calibration, decision

_SPEC = importlib.util.spec_from_file_location(
    "rescore", Path(__file__).resolve().parent.parent / "tools" / "rescore.py")
rescore = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rescore)


class TestPolicyLabelFor:
    @pytest.mark.parametrize("path,label", [
        ("N1_reason_approved", "APPROVED"),
        ("N1_reason_denied", "DENIED"),
        ("N1_reason_needs_review", "NEEDS_REVIEW"),
    ])
    def test_n1_paths_carry_their_label(self, path, label):
        assert rescore.policy_label_for(path) == label

    @pytest.mark.parametrize("path,label", [
        ("N0_note_approved", "APPROVED"),
        ("N0_note_conflict", "NEEDS_REVIEW"),
        ("R3_fee_unpaid", "DENIED"),
        ("R11_default_approve", "APPROVED"),
        ("R12_visa_unread", "NEEDS_REVIEW"),
        ("FALLBACK_error", "NEEDS_REVIEW"),
    ])
    def test_existing_paths_unchanged(self, path, label):
        assert rescore.policy_label_for(path) == label

    def test_n1_replay_matches_live_decision_layer(self):
        # An N1 replay must reproduce the live decision layer exactly:
        # same adjudication, confidence = the clamped calibrated path
        # constant (not a frozen literal — the A9 in-container refit
        # re-fits N1 priors whenever the dump observes the path).
        path = "N1_reason_approved"
        adjudication, confidence = decision.decide(
            rescore.policy_label_for(path), path)
        expected = calibration.clamp_confidence(
            calibration.PATH_STATS[path].accuracy)
        assert (adjudication, confidence) == ("APPROVED", pytest.approx(expected))
