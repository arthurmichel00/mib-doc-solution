"""Spec §11 pitch residual refinement (Task 13; spec tests 9 and 10)."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from mib_pipeline import fragment_realign
from mib_pipeline.fragment_realign import (
    FragmentTransform,
    generate_repair_candidates,
)

from test_fragment_realign import _damage_by_known_transforms

# The frozen suite's proven pitch fixture geometry: row pitch 22 (>= the 17px
# refinement floor), three x-strips with inverse_dy (0, 10, -7).
_ROW_CENTERS = (18, 40, 62, 84, 106)
_TRUE_TRANSFORMS = (
    FragmentTransform((0, 60), inverse_dy=0),
    FragmentTransform((60, 120), inverse_dy=10),
    FragmentTransform((120, 180), inverse_dy=-7),
)


def _pitch_page() -> np.ndarray:
    """Continuous text rows (no blank gutters at the strip boundaries) so the
    §10.1 boundary continuity that refinement gates on actually exists."""
    page = np.full((132, 180), 255, dtype=np.uint8)
    for center in _ROW_CENTERS:
        cv2.putText(
            page,
            "AB12CD34EF56GH78IJ90KL12MN34",
            (2, center + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            20,
            1,
            cv2.LINE_8,
        )
        # One-pixel underlines make the boundary continuity peak sharp, so
        # the +-1px neighbor paths fall outside the frozen 0.03 margin.
        cv2.line(page, (0, center + 6), (180, center + 6), 30, 1, cv2.LINE_8)
    return page


def _damaged_page(page: np.ndarray | None = None) -> np.ndarray:
    damaged, _ = _damage_by_known_transforms(
        page if page is not None else _pitch_page(),
        _TRUE_TRANSFORMS,
        partition_axis="x",
    )
    return damaged


def _emitted_pitch_candidate(damaged: np.ndarray):
    for candidate in generate_repair_candidates(damaged, max_fragments=4, top_k=16):
        if (
            candidate.score_terms.get("method_pitch") == 1.0
            and candidate.partition_axis == "x"
            and [f.inverse_dy for f in candidate.fragments] == [0, 10, -7]
        ):
            return _with_consistent_phase(candidate)
    raise AssertionError("fixture did not emit the three-strip pitch candidate")


def _with_consistent_phase(candidate):
    """Store the comb phase that actually fits the candidate's own offsets.

    The legacy family can store a Viterbi phase that its post-hoc
    profile-refined offsets no longer maximize; refinement trusts the stored
    instrument, so the fixture models an emission whose stored phase is
    self-consistent."""
    evidence = fragment_realign._alignment_evidence(
        candidate.source_view,
        fragment_realign.build_content_masks(candidate.source_view),
    )
    pitch = int(candidate.score_terms["row_pitch"])
    repair_size = evidence.shape[0]

    def alignment(phase: int) -> float:
        target = fragment_realign._comb_target(repair_size, pitch, phase)
        return sum(
            fragment_realign._profile_shift_score(
                evidence[:, fragment.interval[0] : fragment.interval[1]].sum(axis=1),
                target,
                fragment.inverse_dy,
            )
            for fragment in candidate.fragments
        )

    best_phase = max(range(pitch), key=alignment)
    return replace(
        candidate,
        score_terms={**candidate.score_terms, "row_phase": float(best_phase)},
    )


def _perturbed(candidate, deltas: tuple[int, ...]):
    transforms = tuple(
        FragmentTransform(
            fragment.interval,
            inverse_dx=fragment.inverse_dx,
            inverse_dy=fragment.inverse_dy + delta,
        )
        for fragment, delta in zip(candidate.fragments, deltas)
    )
    rebuilt = fragment_realign.apply_fragment_transforms(
        candidate.source_view, transforms, partition_axis="x"
    )
    return replace(
        rebuilt,
        region=candidate.region,
        partition_angle_degrees=candidate.partition_angle_degrees,
        score_terms=dict(candidate.score_terms),
    )


@pytest.mark.parametrize("residual", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pitch_refinement_recovers_known_residuals(residual: int) -> None:
    """Spec test 9: every residual 1..8px is recovered exactly, strip
    boundaries unchanged, primary row score not reduced."""
    damaged = _damaged_page()
    real = _emitted_pitch_candidate(damaged)
    perturbed = _perturbed(real, (0, -residual, -residual))
    measurements = fragment_realign._rail_measurements_for_axis(damaged, axis="x")
    assert measurements is not None

    refinement = fragment_realign._refine_pitch_offsets(
        damaged, perturbed, axis="x", measurements=measurements
    )

    assert refinement is not None
    assert refinement.deltas == (0, residual, residual)
    assert [f.interval for f in refinement.refined.fragments] == [
        f.interval for f in real.fragments
    ]
    assert np.array_equal(refinement.refined.reconstruction, real.reconstruction)
    assert refinement.unrefined is perturbed
    assert refinement.runner_up[0] != refinement.deltas
    terms = refinement.refined.score_terms
    assert terms["pitch_refined"] == 1.0
    assert (
        terms["refinement_primary_after"]
        >= terms["refinement_primary_before"] - 0.005
    )


def test_pitch_refinement_rejects_rule_only_improvement() -> None:
    """Spec test 10: a strong rule lattice misregistered by +4 in the middle
    strip pulls boundary continuity toward a nonzero delta, but the text comb
    resists — refinement must be rejected (text evidence prevents a form
    lattice from overpowering displaced content)."""
    page = np.full((132, 180), 255, dtype=np.uint8)
    for center in _ROW_CENTERS:
        # Sparse text away from the strip-boundary windows.
        cv2.putText(page, "AB12", (16, center + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, 20, 1, cv2.LINE_8)
        cv2.putText(page, "CD34", (78, center + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, 20, 1, cv2.LINE_8)
        cv2.putText(page, "EF56", (138, center + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, 20, 1, cv2.LINE_8)
        cv2.line(page, (0, center + 6), (180, center + 6), 25, 2, cv2.LINE_8)
    damaged, _ = _damage_by_known_transforms(
        page, _TRUE_TRANSFORMS, partition_axis="x"
    )
    # Strip 2 was displaced by -10: erase its rules (now at center-4) and
    # redraw them 4px off the shared grid, creating the rule-only pull.
    for center in _ROW_CENTERS:
        damaged[max(0, center - 6) : center - 1, 60:120] = 255
        cv2.line(damaged, (60, center), (119, center), 25, 2, cv2.LINE_8)

    rebuilt = fragment_realign.apply_fragment_transforms(
        damaged, _TRUE_TRANSFORMS, partition_axis="x"
    )
    candidate = replace(
        rebuilt,
        region=(0, 0, 180, 132),
        score_terms={"method_pitch": 1.0, "row_pitch": 22.0, "row_phase": 0.0},
    )
    candidate = _with_consistent_phase(candidate)
    measurements = fragment_realign._rail_measurements_for_axis(damaged, axis="x")
    assert measurements is not None
    # Fixture self-check: the rule pull is real at the first boundary.
    boundary = candidate.fragments[0].interval[1]
    base_relative = (
        candidate.fragments[1].inverse_dy - candidate.fragments[0].inverse_dy
    )
    before = fragment_realign._joint_seam_continuity(
        measurements, boundary, base_relative,
        part_interval=(0, 180), cross_interval=(0, 132),
    )
    pulled = fragment_realign._joint_seam_continuity(
        measurements, boundary, base_relative - 4,
        part_interval=(0, 180), cross_interval=(0, 132),
    )
    assert pulled > before

    refinement = fragment_realign._refine_pitch_offsets(
        damaged, candidate, axis="x", measurements=measurements
    )

    assert refinement is None


def test_pitch_refinement_skips_below_pitch_floor() -> None:
    """Spec 11.2: estimated row pitch below 17px is never refined."""
    damaged = _damaged_page()
    real = _emitted_pitch_candidate(damaged)
    small_pitch = replace(
        real, score_terms={**real.score_terms, "row_pitch": 16.0}
    )
    measurements = fragment_realign._rail_measurements_for_axis(damaged, axis="x")

    assert (
        fragment_realign._refine_pitch_offsets(
            damaged, small_pitch, axis="x", measurements=measurements
        )
        is None
    )
