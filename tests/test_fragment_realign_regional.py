"""Stage A1 harness tests for bounded regional fragment reconstruction.

Pins the spec §7.1 private diagnostics entry point: the public
``generate_page_repair_candidates`` must stay byte-identical to the private
entry in ``all_family_diagnostic`` mode, ``active_default`` must drop only the
bounded two-rail family, and the diagnostics sidecar must stay keyed by the
existing geometry signature.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from mib_pipeline import fragment_realign
from mib_pipeline.fragment_realign import (
    FragmentTransform,
    generate_page_repair_candidates,
)

from test_fragment_realign import (
    _bounded_two_rail_lattice,
    _damage_by_known_transforms,
    _faint_rule_lattice,
)

_PAGE_WHITE = 255
_RAIL_GRAY = 242
_GLYPH_GRAY = 20
_STROKE_GRAY = 90
# Blank border kept free of ink: displacements smaller than this crop only
# white pixels, so an exact repair is byte-equal to the clean page (the
# transform machinery fills uncovered destinations with white).
_CONTENT_MARGIN = 80


def build_lattice_page(
    *,
    length: int = 1600,
    width: int = 1200,
    axis: str = "y",
    rail_period: int = 110,
    glyph_rows: bool = True,
) -> np.ndarray:
    """Clean synthetic form: periodic faint rails crossing the partition axis
    plus asymmetric glyph rows between them.

    For ``axis="y"`` the page is ``(length, width)`` with vertical rails every
    ``rail_period`` columns; ``axis="x"`` returns the exact transpose. Row
    labels and slanted strokes vary deterministically so non-rule continuity
    identifies each seam without periodic aliases.
    """
    if axis not in ("y", "x"):
        raise ValueError("axis must be 'y' or 'x'")
    page = np.full((length, width), _PAGE_WHITE, dtype=np.uint8)
    for x in range(_CONTENT_MARGIN, width - _CONTENT_MARGIN + 1, rail_period):
        cv2.line(
            page,
            (x, _CONTENT_MARGIN // 2),
            (x, length - _CONTENT_MARGIN // 2),
            _RAIL_GRAY,
            2,
            cv2.LINE_8,
        )
    if glyph_rows:
        for index, baseline in enumerate(
            range(_CONTENT_MARGIN + 40, length - _CONTENT_MARGIN, 97)
        ):
            label = f"{chr(65 + index % 23)}{index:02d} {(index * 37) % 89:02d}K"
            cv2.putText(
                page,
                label,
                (_CONTENT_MARGIN + 20 + (index * 53) % 240, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                _GLYPH_GRAY,
                2,
                cv2.LINE_8,
            )
            cv2.line(
                page,
                (width // 2 + (index * 31) % 160, baseline - 12),
                (width // 2 + 150 + (index * 13) % 90, baseline + 10),
                _STROKE_GRAY,
                3,
                cv2.LINE_8,
            )
    if axis == "x":
        return np.ascontiguousarray(page.T)
    return page


def _transforms_for_seams(
    seams: list[tuple[int, int]], *, length: int, axis: str = "y"
) -> tuple[FragmentTransform, ...]:
    """Expected repair transforms for ``seams``: each ``(boundary, offset)``
    displaces everything after ``boundary`` by ``offset`` relative to the
    fragment before it, so fragment shifts are cumulative seam offsets with
    fragment zero as the gauge anchor."""
    boundaries = [boundary for boundary, _ in seams]
    if boundaries != sorted(set(boundaries)) or not all(
        0 < boundary < length for boundary in boundaries
    ):
        raise ValueError("seam boundaries must be strictly increasing and interior")
    edges = [0, *boundaries, length]
    cumulative = [0]
    for _, offset in seams:
        cumulative.append(cumulative[-1] + offset)
    key = "inverse_dx" if axis == "y" else "inverse_dy"
    return tuple(
        FragmentTransform((start, end), **{key: shift})
        for start, end, shift in zip(edges[:-1], edges[1:], cumulative)
    )


def displace_fragments(
    page: np.ndarray, seams: list[tuple[int, int]], *, axis: str = "y"
) -> np.ndarray:
    """Forward-damage ``page`` so that ``_transforms_for_seams(seams)`` is the
    exact repair, with no-wrap semantics from ``apply_fragment_transforms``."""
    length = page.shape[0] if axis == "y" else page.shape[1]
    transforms = _transforms_for_seams(seams, length=length, axis=axis)
    damaged, _ = _damage_by_known_transforms(page, transforms, partition_axis=axis)
    return damaged


def erase_rails_near(
    page: np.ndarray, boundary: int, radius: int, *, axis: str = "y"
) -> np.ndarray:
    """Blank rail pixels (only value ``_RAIL_GRAY``) in the band of
    ``radius`` around ``boundary`` on the partition axis, keeping glyphs."""
    if axis == "x":
        return np.ascontiguousarray(
            erase_rails_near(page.T, boundary, radius, axis="y").T
        )
    result = page.copy()
    band = result[max(0, boundary - radius) : boundary + radius, :]
    band[band == _RAIL_GRAY] = _PAGE_WHITE
    return result


def _damaged_lattice_page() -> np.ndarray:
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, _ = _damage_by_known_transforms(
        _faint_rule_lattice(), transforms, partition_axis="y"
    )
    return damaged


def _damaged_two_rail_page() -> np.ndarray:
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        _bounded_two_rail_lattice(), transforms, partition_axis="y"
    )
    return damaged


def test_public_wrapper_matches_all_family_diagnostic_private_entry() -> None:
    page = _damaged_lattice_page()

    public = generate_page_repair_candidates(page, top_k=16)
    private, _diagnostics = (
        fragment_realign._generate_page_repair_candidates_with_diagnostics(
            page, top_k=16, family_mode="all_family_diagnostic"
        )
    )

    assert public
    assert len(public) == len(private)
    for ours, theirs in zip(public, private):
        assert ours.region == theirs.region
        assert ours.partition_axis == theirs.partition_axis
        assert ours.partition_angle_degrees == theirs.partition_angle_degrees
        assert ours.fragments == theirs.fragments
        assert ours.score_terms == theirs.score_terms
        assert ours.pixel_score_before == theirs.pixel_score_before
        assert ours.pixel_score_after == theirs.pixel_score_after
        assert ours.reconstruction.tobytes() == theirs.reconstruction.tobytes()


def test_active_default_emits_no_bounded_two_rail_candidate() -> None:
    page = _damaged_two_rail_page()

    diagnostic, _ = (
        fragment_realign._generate_page_repair_candidates_with_diagnostics(
            page, top_k=32, family_mode="all_family_diagnostic"
        )
    )
    active, _ = (
        fragment_realign._generate_page_repair_candidates_with_diagnostics(
            page, top_k=32, family_mode="active_default"
        )
    )

    # The fixture must actually exercise the family being filtered out.
    assert any(
        candidate.score_terms.get("method_two_rail_bounded_local") == 1.0
        for candidate in diagnostic
    )
    assert all(
        candidate.score_terms.get("method_two_rail_bounded_local") != 1.0
        for candidate in active
    )


def test_diagnostics_sidecar_is_present_and_signature_keyed() -> None:
    page = _damaged_lattice_page()

    candidates, diagnostics = (
        fragment_realign._generate_page_repair_candidates_with_diagnostics(
            page, top_k=16
        )
    )

    assert isinstance(diagnostics, dict)
    allowed_keys = {
        fragment_realign._geometry_signature_key(candidate)
        for candidate in candidates
    }
    for key, record in diagnostics.items():
        assert isinstance(key, str)
        assert key in allowed_keys
        assert isinstance(record, dict)


def test_family_mode_rejects_unknown_values() -> None:
    page = _damaged_lattice_page()

    with pytest.raises(ValueError):
        fragment_realign._generate_page_repair_candidates_with_diagnostics(
            page, family_mode="everything"
        )


# Captured from generate_page_repair_candidates on the pre-Task-4 code (Stage
# A1, 2026-07-28): the _AnchoredSeam record refactor must reproduce these
# byte-identically. The -33/-104 offsets pin the solver's current alias
# behavior on a uniform rail grid, not ground truth.
_ANCHOR_REGRESSION_SEAMS = [(500, 18), (1000, -30), (1300, 6)]
_ANCHOR_REGRESSION_EXPECTED = [
    (
        "((0, 300, 1200, 700), 'y', (((0, 200), 0, 0), ((200, 400), 18, 0)))",
        {
            "complexity_penalty": 0.015,
            "fragment_count": 2.0,
            "loss_fraction": 0.0075,
            "loss_penalty": 0.0026249999999999997,
            "matched_rail_count": 7.0,
            "matched_rail_span_fraction": 0.1491844203655568,
            "method_rule_anchor": 1.0,
            "non_rule_continuity": 0.72601009408169,
            "non_rule_continuity_after": 0.72601009408169,
            "non_rule_continuity_before": 0.4708112343478187,
            "non_rule_continuity_gain": 0.25519885973387135,
            "non_rule_support": 4122.0,
            "observed_relative_offset": -18.0,
            "periodic_alias_margin": 0.9325824263908049,
            "rail_endpoint_score": 1.0442909425588975,
            "total_gain": 0.23757385973387135,
        },
        (0.4708112343478187, 0.72601009408169),
    ),
    (
        "((0, 800, 1200, 1200), 'y', (((0, 200), 0, 0), ((200, 400), -33, 0)))",
        {
            "complexity_penalty": 0.015,
            "fragment_count": 2.0,
            "loss_fraction": 0.01375,
            "loss_penalty": 0.0048125,
            "matched_rail_count": 8.0,
            "matched_rail_span_fraction": 0.06028868872496557,
            "method_rule_anchor": 1.0,
            "non_rule_continuity": 0.11225294733614334,
            "non_rule_continuity_after": 0.11225294733614334,
            "non_rule_continuity_before": 0.0011053320873980568,
            "non_rule_continuity_gain": 0.11114761524874528,
            "non_rule_support": 3557.0,
            "observed_relative_offset": 33.0,
            "periodic_alias_margin": 0.2795145906487863,
            "rail_endpoint_score": 0.48230950979972453,
            "total_gain": 0.09133511524874528,
        },
        (0.0011053320873980568, 0.11225294733614334),
    ),
    (
        "((0, 1100, 1200, 1500), 'y', (((0, 200), 0, 0), ((200, 400), -104, 0)))",
        {
            "complexity_penalty": 0.015,
            "fragment_count": 2.0,
            "loss_fraction": 0.043333333333333335,
            "loss_penalty": 0.015166666666666667,
            "matched_rail_count": 7.0,
            "matched_rail_span_fraction": 0.05498554997511314,
            "method_rule_anchor": 1.0,
            "non_rule_continuity": 0.08367010571992908,
            "non_rule_continuity_after": 0.08367010571992908,
            "non_rule_continuity_before": 0.010923505832762286,
            "non_rule_continuity_gain": 0.0727465998871668,
            "non_rule_support": 5290.0,
            "observed_relative_offset": 104.0,
            "periodic_alias_margin": 0.01693116425169061,
            "rail_endpoint_score": 0.384898849825792,
            "total_gain": 0.042579933220500125,
        },
        (0.010923505832762286, 0.08367010571992908),
    ),
]


def test_rule_anchor_candidates_match_captured_pre_refactor_values() -> None:
    damaged = displace_fragments(build_lattice_page(), _ANCHOR_REGRESSION_SEAMS)

    candidates = generate_page_repair_candidates(damaged, top_k=8)

    observed = [
        (
            fragment_realign._geometry_signature_key(candidate),
            dict(sorted(candidate.score_terms.items())),
            (float(candidate.pixel_score_before), float(candidate.pixel_score_after)),
        )
        for candidate in candidates
    ]
    assert observed == _ANCHOR_REGRESSION_EXPECTED


def _add_tick_comb(page: np.ndarray, boundary: int) -> np.ndarray:
    """Aperiodic dark tick comb crossing ``boundary``: dense glyph-mask
    content that must reconnect across the cut, with no periodic alias."""
    result = page.copy()
    x = 300
    for step in (5, 7, 4, 8, 6, 5, 9, 4, 6, 7) * 4:
        cv2.line(result, (x, boundary - 9), (x, boundary + 9), _GLYPH_GRAY, 2, cv2.LINE_8)
        x += step
        if x > 450:
            break
    return result


_GLYPH_SEAM_BOUNDARY = 502
_GLYPH_SEAM_OFFSET = 9
_GLYPH_SEAM_REGION = (280, 352, 470, 652)


def _glyph_only_seam_page() -> np.ndarray:
    """One glyph-only seam: rails erased at the boundary, no qualifying rail
    event anywhere (spec test 2 fixture)."""
    page = _add_tick_comb(build_lattice_page(), _GLYPH_SEAM_BOUNDARY)
    page = erase_rails_near(page, _GLYPH_SEAM_BOUNDARY, 60)
    return displace_fragments(page, [(_GLYPH_SEAM_BOUNDARY, _GLYPH_SEAM_OFFSET)])


def test_expand_region_expands_and_clips() -> None:
    expanded = fragment_realign._expand_region(
        (100, 200, 300, 400), rail_length=20, min_fragment=50, page_shape=(1000, 800)
    )
    assert expanded == (50, 150, 350, 450)

    clipped = fragment_realign._expand_region(
        (10, 5, 795, 995), rail_length=30, min_fragment=8, page_shape=(1000, 800)
    )
    assert clipped == (0, 0, 800, 1000)


def _rail_seam_at(boundary: int, centers: tuple[float, float]) -> object:
    seam = fragment_realign._SeamHypothesis(
        boundary=boundary,
        relative_offset=10,
        similarity_before=0.1,
        similarity_after=0.5,
        edge_similarity_before=0.0,
        edge_similarity_after=0.0,
        shared_support=4.0,
        gain=0.4,
        objective=1.0,
    )
    pairs = tuple(
        fragment_realign._MatchedRailPair(
            ending_rail_id=index,
            starting_rail_id=index,
            ending_center=center,
            starting_center=center,
            ending_extent=(0, boundary),
            starting_extent=(boundary, 2 * boundary),
        )
        for index, center in enumerate(centers)
    )
    return fragment_realign._AnchoredSeam(
        seam=seam,
        source="rail_endpoint",
        matched_rail_count=len(pairs),
        non_rule_support=100.0,
        alias_margin=0.5,
        matched_pairs=pairs,
        support_interval=(boundary - 96, boundary + 96),
        objective=1.0,
    )


def test_assign_rail_seams_to_regions_overlap_gate() -> None:
    # Region orthogonal interval (100, 200), length 100. Seam cross spans have
    # length 50 (the smaller interval): 35% of 50 = 17.5.
    region = (100, 0, 200, 500)
    excluded_34 = _rail_seam_at(250, (183.0, 233.0))  # overlap 17 -> excluded
    included_36 = _rail_seam_at(250, (182.0, 232.0))  # overlap 18 -> included
    outside_boundary = _rail_seam_at(600, (150.0, 200.0))

    assigned = fragment_realign._assign_rail_seams_to_regions(
        [excluded_34, included_36, outside_boundary], [region], axis="y"
    )

    assert assigned == {0: [included_36]}


def test_regional_glyph_seams_finds_glyph_only_seam_in_page_coordinates() -> None:
    page = _glyph_only_seam_page()
    measurements = fragment_realign._rail_measurements_for_axis(page, axis="y")
    assert measurements is not None

    seams = fragment_realign._regional_glyph_seams(
        page,
        _GLYPH_SEAM_REGION,
        axis="y",
        min_fragment=measurements.min_fragment,
        measurements=measurements,
    )

    assert len(seams) == 1
    seam = seams[0]
    assert seam.source == "glyph_continuity"
    assert seam.seam.boundary == _GLYPH_SEAM_BOUNDARY
    assert seam.seam.relative_offset == _GLYPH_SEAM_OFFSET
    assert seam.matched_pairs == ()
    assert seam.matched_rail_count == 0
    assert seam.objective == seam.seam.objective
    assert seam.non_rule_support > 0.0
    assert seam.alias_margin > 0.0
    # Non-rule support bounding interval, remapped to page cross coordinates:
    # it must cover the tick comb (x 300..~449, last tick starts at <=450)
    # and stay inside the region.
    x0, _, x1, _ = _GLYPH_SEAM_REGION
    low, high = seam.support_interval
    assert x0 <= low <= 300
    assert 440 <= high <= x1


def test_glyph_only_seam_emits_no_rule_anchor_component() -> None:
    """Spec test 2: a glyph seam cannot initiate a rule-anchor component."""
    page = _glyph_only_seam_page()

    candidates = generate_page_repair_candidates(page, top_k=16)

    assert all(
        candidate.score_terms.get("method_rule_anchor_component") != 1.0
        for candidate in candidates
    )


def _component_fixture_page() -> np.ndarray:
    """Positive-control page: a rail seam at 502 and a rail-erased glyph seam
    at 900, plus a dense text block spanning both so the automatic region
    proposal yields one region containing them."""
    page = build_lattice_page()
    # Column-stable dense field block: fixed x so its profile realigns across
    # both seams after repair (a staggered block would dilute continuity).
    for index, row in enumerate(range(430, 1000, 30)):
        cv2.putText(
            page,
            f"F{index:02d}-{(index * 29) % 97:02d}",
            (260, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _GLYPH_GRAY,
            2,
            cv2.LINE_8,
        )
    # A long field row cut through its middle by the 900 boundary: dense
    # glyph occupancy across the seam band (the detector's 55% support gate)
    # with no rail evidence once the band's rails are erased.
    cv2.putText(
        page,
        "CONTINUATION FIELD ROW WITH MANY GLYPHS 0123456789 ABCDEFGH",
        (120, 906),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _GLYPH_GRAY,
        2,
        cv2.LINE_8,
    )
    page = erase_rails_near(page, 900, 60)
    return displace_fragments(page, [(502, 24), (900, 9)])


def _seam_geometry(candidate) -> tuple[list[int], list[int]]:
    """Page-coordinate seam boundaries and per-seam relative offsets."""
    origin = candidate.region[1] if candidate.partition_axis == "y" else (
        candidate.region[0]
    )
    boundaries = [
        fragment.interval[1] + origin for fragment in candidate.fragments[:-1]
    ]
    shifts = [
        fragment.inverse_dx if candidate.partition_axis == "y" else (
            fragment.inverse_dy
        )
        for fragment in candidate.fragments
    ]
    offsets = [second - first for first, second in zip(shifts, shifts[1:])]
    return boundaries, offsets


def test_rail_anchored_component_completed_by_glyph_seam() -> None:
    """Spec test 1, candidate level: with a qualifying rail seam present, the
    glyph seam completes a component candidate with exact geometry."""
    page = _component_fixture_page()

    candidates = generate_page_repair_candidates(page, top_k=16)

    components = [
        candidate
        for candidate in candidates
        if candidate.score_terms.get("method_rule_anchor_component") == 1.0
    ]
    assert components
    assert all(
        candidate.score_terms.get("method_rule_anchor") == 1.0
        for candidate in components
    )
    geometries = [
        _seam_geometry(candidate)
        for candidate in components
        if candidate.partition_axis == "y"
    ]
    assert ([502, 900], [24, 9]) in geometries


_STRIPE_RAIL_XS = (95, 180, 290, 430, 520, 660, 810, 930, 1080)


def _aperiodic_stripe_page() -> np.ndarray:
    """Spec test 13 fixture: aperiodic rails (no periodic aliases), glyph rows
    around the first seam, and a horizontal stripe band around the second seam
    so its lateral offset is continuity-neutral while its rail-endpoint
    evidence stays excellent."""
    page = np.full((1600, 1200), _PAGE_WHITE, dtype=np.uint8)
    for x in _STRIPE_RAIL_XS:
        cv2.line(page, (x, 40), (x, 1560), _RAIL_GRAY, 2, cv2.LINE_8)
    for index, baseline in enumerate(range(120, 900, 97)):
        label = f"{chr(65 + index % 23)}{index:02d} {(index * 37) % 89:02d}K"
        cv2.putText(
            page,
            label,
            (100 + (index * 53) % 240, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            _GLYPH_GRAY,
            2,
            cv2.LINE_8,
        )
        cv2.line(
            page,
            (600 + (index * 31) % 160, baseline - 12),
            (750 + (index * 13) % 90, baseline + 10),
            _STROKE_GRAY,
            3,
            cv2.LINE_8,
        )
    # Border-to-border one-pixel stripes inside the continuity windows: a
    # lateral shift leaves their profiles identical, so the 1100 seam's joint
    # gain is ~0 while its rail-endpoint evidence stays excellent.  Thin
    # stripes keep this seam's non-rule support below 4x the 500 seam's so
    # the 25% support floor is not the deciding gate.
    for row in range(1068, 1133, 8):
        cv2.line(page, (0, row), (1200, row), _GLYPH_GRAY, 1, cv2.LINE_8)
    return displace_fragments(page, [(500, 24), (1100, 30)])


def test_all_page_chain_rejected_when_one_seam_is_neutral() -> None:
    """Spec test 13: the all-page chain contains one excellent seam and one
    continuity-neutral seam; the corrected joint gate rejects the chain even
    though its best seam is excellent (best-seam inheritance is gone).  The
    excellent seam's independent single-seam candidate survives."""
    page = _aperiodic_stripe_page()

    candidates = generate_page_repair_candidates(page, top_k=16)

    full_page_chains = [
        candidate
        for candidate in candidates
        if candidate.score_terms.get("method_rule_anchor") == 1.0
        and candidate.region == (0, 0, 1200, 1600)
        and len(candidate.fragments) == 3
    ]
    assert full_page_chains == []
    single_seams = [
        _seam_geometry(candidate)
        for candidate in candidates
        if candidate.score_terms.get("method_rule_anchor") == 1.0
        and candidate.partition_axis == "y"
        and len(candidate.fragments) == 2
    ]
    assert ([500], [24]) in single_seams


def _component_node(
    boundary: int,
    offset: int,
    *,
    source: str = "rail_endpoint",
    gain: float = 0.25,
    objective: float = 1.0,
    alias_margin: float = 0.5,
    cross_interval: tuple[float, float] = (80.0, 1120.0),
) -> object:
    seam = fragment_realign._SeamHypothesis(
        boundary=boundary,
        relative_offset=offset,
        similarity_before=0.1,
        similarity_after=0.1 + gain,
        edge_similarity_before=0.0,
        edge_similarity_after=0.0,
        shared_support=4.0,
        gain=gain,
        objective=objective,
    )
    low, high = cross_interval
    if source == "rail_endpoint":
        pairs = tuple(
            fragment_realign._MatchedRailPair(
                ending_rail_id=index,
                starting_rail_id=index,
                ending_center=center,
                starting_center=center,
                ending_extent=(0, boundary),
                starting_extent=(boundary, 2 * boundary),
            )
            for index, center in enumerate((low, high))
        )
        support_interval = (boundary - 96, boundary + 96)
    else:
        pairs = ()
        support_interval = (int(low), int(high))
    return fragment_realign._AnchoredSeam(
        seam=seam,
        source=source,
        matched_rail_count=len(pairs),
        non_rule_support=100.0,
        alias_margin=alias_margin,
        matched_pairs=pairs,
        support_interval=support_interval,
        objective=objective,
    )


def _track(
    center: float, start: int, end: int, *, jog_at: int | None = None, jog: float = 0.0
) -> object:
    return fragment_realign._RailTrack(
        center=center,
        extent=(start, end),
        row_centers=tuple(
            center + (jog if jog_at is not None and row >= jog_at else 0.0)
            for row in range(start, end)
        ),
    )


def _build_components(nodes, *, tracks=(), max_fragments=5, min_fragment=20, bound=96):
    return fragment_realign._build_regional_components(
        list(nodes),
        axis="y",
        max_fragments=max_fragments,
        min_fragment=min_fragment,
        center_tolerance=3,
        rail_tracks=tuple(tracks),
        displacement_bound=bound,
    )


def test_build_regional_components_links_rail_and_glyph_seams() -> None:
    """Spec test 1 at unit level: a rail-anchored path is completed by a glyph
    seam, recovering exact boundaries and offsets, in the frozen retention
    order (descending mean gain, fewer fragments, boundary tuple, offsets)."""
    n1 = _component_node(500, 18, gain=0.5)
    n2 = _component_node(760, 24, gain=0.25)
    n3 = _component_node(
        1100, 9, source="glyph_continuity", gain=0.375, cross_interval=(300.0, 452.0)
    )
    # Rails broken exactly at the seams (and erased around the glyph seam):
    # a segment spanning one fragment interval is NOT an intact-gap witness.
    tracks = [
        _track(center, start, end)
        for center in (100.0, 210.0, 320.0)
        for start, end in ((40, 500), (500, 760), (760, 1040), (1160, 1560))
    ]

    components = _build_components([n3, n1, n2], tracks=tracks)

    assert components == [
        (n1,),
        (n1, n2),
        (n1, n2, n3),
        (n2, n3),
        (n2,),
    ]
    triple = components[2]
    assert [(s.seam.boundary, s.seam.relative_offset) for s in triple] == [
        (500, 18),
        (760, 24),
        (1100, 9),
    ]
    # A glyph seam cannot initiate a component: no glyph-only path.
    assert (n3,) not in components


def test_build_regional_components_blocked_by_intact_rail_gap() -> None:
    """Spec test 6 at unit level: three straight tracks crossing strictly past
    both boundaries certify an intact gap; the nodes stay disjoint."""
    a = _component_node(300, 10)
    b = _component_node(900, -10)
    intact = [_track(center, 100, 1100) for center in (100.0, 210.0, 320.0)]

    blocked = _build_components([a, b], tracks=intact)
    assert blocked == [(a,), (b,)]

    # Two crossing tracks are not an intact gap.
    two_tracks = _build_components([a, b], tracks=intact[:2])
    assert (a, b) in two_tracks

    # A track spanning exactly the inter-node interval is a displaced-fragment
    # rail, not a witness: it does not extend past either boundary.
    fragment_rails = _build_components(
        [a, b], tracks=[_track(center, 300, 900) for center in (100.0, 210.0, 320.0)]
    )
    assert (a, b) in fragment_rails

    # Three crossing tracks whose pooled sampled offsets are <75% within
    # tolerance of one constant do not certify the gap.
    jogged = intact[:2] + [_track(320.0, 100, 1100, jog_at=400, jog=8.0)]
    not_certified = _build_components([a, b], tracks=jogged)
    assert (a, b) in not_certified


def test_build_regional_components_bounded_under_many_page_events() -> None:
    """Spec test 8 at unit level: more events than the fragment cap still emit
    bounded contiguous windows, never an all-page chain."""
    nodes = [
        _component_node(200 + 240 * index, 10, gain=0.6 - 0.05 * index)
        for index in range(6)
    ]

    components = _build_components(nodes, max_fragments=5)

    assert len(components) == 18  # windows of length 1..4 over six nodes
    assert max(len(path) for path in components) == 4
    assert all(len(path) <= 4 for path in components)
    assert tuple(nodes) not in components
    assert tuple(nodes[0:4]) in components
    assert components[0] == (nodes[0],)


def test_build_regional_components_caps_nodes_and_retained_paths() -> None:
    """Node cap: at most 8 rail + 8 glyph nodes by descending objective; path
    cap: at most 32 retained paths; every path carries a rail seam."""
    rails = [
        _component_node(100 + 180 * index, 5 if index % 2 == 0 else -5)
        for index in range(8)
    ]
    glyphs = [
        _component_node(
            190 + 180 * index,
            -5 if index % 2 == 0 else 5,
            source="glyph_continuity",
            cross_interval=(80.0, 1120.0),
        )
        for index in range(8)
    ]
    junk = [
        _component_node(1540, 5, objective=0.1),
        _component_node(1630, -5, objective=0.1),
    ]

    components = _build_components(rails + glyphs + junk, max_fragments=5)

    assert len(components) == 32
    assert all(
        any(seam.source == "rail_endpoint" for seam in path) for path in components
    )
    assert all(len(path) <= 4 for path in components)
    dropped = set()
    for path in components:
        dropped.update(seam.seam.boundary for seam in path)
    assert 1540 not in dropped and 1630 not in dropped


def test_component_edges_require_spacing_and_support_overlap() -> None:
    a = _component_node(300, 10, cross_interval=(150.0, 200.0))
    excluded_34 = _component_node(400, 10, cross_interval=(183.0, 233.0))
    included_36 = _component_node(400, 10, cross_interval=(182.0, 232.0))

    assert (a, excluded_34) not in _build_components([a, excluded_34])
    assert (a, included_36) in _build_components([a, included_36])
    # Same connectable pair, but boundaries closer than min_fragment.
    assert (a, included_36) not in _build_components(
        [a, included_36], min_fragment=120
    )


def test_component_edges_require_alias_margin_for_edges_not_singletons() -> None:
    a = _component_node(300, 10)
    weak = _component_node(400, 10, alias_margin=0.029)
    at_threshold = _component_node(400, 10, alias_margin=0.03)

    weak_components = _build_components([a, weak])
    assert (a, weak) not in weak_components
    assert (a,) in weak_components and (weak,) in weak_components

    assert (a, at_threshold) in _build_components([a, at_threshold])


def test_component_paths_respect_displacement_bound() -> None:
    a = _component_node(300, 60)
    b = _component_node(400, 60)

    bounded = _build_components([a, b], bound=96)
    assert (a, b) not in bounded
    assert (a,) in bounded and (b,) in bounded

    assert (a, b) in _build_components([a, b], bound=130)

    # The very first seam offset is part of the cumulative path.
    oversized = _component_node(300, 100)
    assert _build_components([oversized], bound=96) == []


_JOINT_SCORE_SEAMS = [(500, 18), (1000, -30)]


def _joint_score_page() -> np.ndarray:
    return displace_fragments(build_lattice_page(), _JOINT_SCORE_SEAMS)


def _joint_score(page, component, *, support_floor: float = 0.0):
    measurements = fragment_realign._rail_measurements_for_axis(page, axis="y")
    assert measurements is not None
    return fragment_realign._joint_component_score(
        component,
        gray=page,
        region=(0, 0, page.shape[1], page.shape[0]),
        axis="y",
        measurements=measurements,
        support_floor=support_floor,
    )


def test_joint_component_score_rejects_worsening_seam_and_keeps_subpath() -> None:
    """Spec test 3: two improving seams score; inserting a worsening seam
    (offset at an undamaged boundary) rejects the whole component."""
    page = _joint_score_page()
    good = (_component_node(500, 18), _component_node(1000, -30))
    bad = (good[0], _component_node(760, 15), good[1])

    good_score = _joint_score(page, good)
    bad_score = _joint_score(page, bad)

    assert bad_score is None
    assert good_score is not None
    assert good_score["seam_count"] == 2.0
    assert good_score["fragment_count"] == 3.0
    assert good_score["raw_gain"] == pytest.approx(
        good_score["mean_after"] - good_score["mean_before"]
    )
    assert good_score["complexity_penalty"] == pytest.approx(0.015 * 2)
    assert good_score["loss_penalty"] == pytest.approx(
        0.35 * good_score["loss_fraction"]
    )
    assert good_score["total_gain"] == pytest.approx(
        good_score["raw_gain"]
        - good_score["complexity_penalty"]
        - good_score["loss_penalty"]
    )
    assert good_score["total_gain"] > 0.035


def test_joint_component_score_fails_appended_neutral_alias_seam() -> None:
    """Spec test 4: appending a rail-period alias seam (nearly neutral rule
    continuity, no glyph support) to a correct component fails the per-seam
    gate; the extension can never outrank the base component."""
    page = _joint_score_page()
    base = (_component_node(500, 18), _component_node(1000, -30))
    extended = (*base, _component_node(1300, 110))

    assert _joint_score(page, base) is not None
    assert _joint_score(page, extended) is None


def test_joint_component_score_enforces_rail_support_floor() -> None:
    """A rail seam whose non-rule support falls below the same-axis floor
    fails the frozen standalone predicate (spec 10.2)."""
    page = _joint_score_page()
    component = (_component_node(500, 18), _component_node(1000, -30))

    assert _joint_score(page, component) is not None
    assert _joint_score(page, component, support_floor=1e9) is None


_ENDPOINT_RAIL_XS = tuple(150 + 150 * index for index in range(10))


def _endpoint_rail_page(*, width: int = 2400, height: int = 900) -> np.ndarray:
    """Constructed rail pattern for the endpoint-measurement contract: page
    wide enough that center_tolerance is 6, so a 5px residual stays inside
    tolerance and must be reported exactly."""
    page = np.full((height, width), _PAGE_WHITE, dtype=np.uint8)
    for x in _ENDPOINT_RAIL_XS:
        cv2.line(page, (x, 60), (x, 840), _RAIL_GRAY, 2, cv2.LINE_8)
    return page


def _endpoint_report(page, transforms, *, region=None):
    measurements = fragment_realign._rail_measurements_for_axis(page, axis="y")
    assert measurements is not None
    assert measurements.center_tolerance == 6
    region = region or (0, 0, page.shape[1], page.shape[0])
    x0, y0, x1, y1 = region
    reconstructed = fragment_realign.apply_fragment_transforms(
        page[y0:y1, x0:x1], transforms, partition_axis="y"
    ).reconstruction
    return fragment_realign._measure_endpoint_residuals(
        page,
        reconstructed,
        region=region,
        axis="y",
        transforms=transforms,
        measurements=measurements,
    )


@pytest.mark.parametrize("residual", [0, 1, 2, 5])
def test_endpoint_residuals_report_exact_quarter_pixels(residual: int) -> None:
    page = displace_fragments(_endpoint_rail_page(), [(400, 12)])
    transforms = (
        FragmentTransform((0, 400)),
        FragmentTransform((400, 900), inverse_dx=12 - residual),
    )

    report = _endpoint_report(page, transforms)

    assert len(report.internal) == len(_ENDPOINT_RAIL_XS)
    assert {pair.residual_quarter_px for pair in report.internal} == {4 * residual}
    assert report.crop_edges == ()
    assert report.unmatched_strong_source == 0
    assert report.source_endpoint_absent == 0
    assert report.clipped_strong_rails == 0


def test_endpoint_residuals_absent_endpoint_is_not_zero_residual() -> None:
    """Spec 12.4: an endpoint absent in the source probe is recorded as
    source_endpoint_absent, never as a zero-residual match."""
    page = displace_fragments(_endpoint_rail_page(), [(400, 12)])
    erased_x = _ENDPOINT_RAIL_XS[5] - 12
    page[400:460, erased_x - 2 : erased_x + 4] = _PAGE_WHITE
    transforms = (
        FragmentTransform((0, 400)),
        FragmentTransform((400, 900), inverse_dx=12),
    )

    report = _endpoint_report(page, transforms)

    assert len(report.internal) == len(_ENDPOINT_RAIL_XS) - 1
    assert {pair.residual_quarter_px for pair in report.internal} == {0}
    assert report.source_endpoint_absent == 1
    assert report.unmatched_strong_source == 0


def test_endpoint_residuals_unmatched_strong_endpoint_is_closure_failure() -> None:
    """A strong endpoint whose mate exists in the probe window but beyond
    center_tolerance is a closure failure, not an observable absence."""
    page = displace_fragments(_endpoint_rail_page(), [(400, 12)])
    moved_x = _ENDPOINT_RAIL_XS[5] - 12
    page[400:841, moved_x - 2 : moved_x + 4] = _PAGE_WHITE
    cv2.line(page, (moved_x + 15, 400), (moved_x + 15, 840), _RAIL_GRAY, 2, cv2.LINE_8)
    transforms = (
        FragmentTransform((0, 400)),
        FragmentTransform((400, 900), inverse_dx=12),
    )

    report = _endpoint_report(page, transforms)

    assert len(report.internal) == len(_ENDPOINT_RAIL_XS) - 1
    assert report.unmatched_strong_source == 2
    assert report.source_endpoint_absent == 0


def test_endpoint_residuals_count_clipped_strong_rails() -> None:
    """A strong rail whose transformed position exits the crop is clipped."""
    page = displace_fragments(_endpoint_rail_page(), [(400, -12)])
    cv2.line(page, (8, 400), (8, 840), _RAIL_GRAY, 2, cv2.LINE_8)
    transforms = (
        FragmentTransform((0, 400)),
        FragmentTransform((400, 900), inverse_dx=-12),
    )

    report = _endpoint_report(page, transforms)

    assert len(report.internal) == len(_ENDPOINT_RAIL_XS)
    assert {pair.residual_quarter_px for pair in report.internal} == {0}
    assert report.clipped_strong_rails == 1
    assert report.source_endpoint_absent == 1  # the extra rail has no top mate


def test_endpoint_residuals_measure_crop_edges_against_untouched_pixels() -> None:
    """Spec 12.2 step 7: crop-edge residuals compare the transformed inside
    probe with the unchanged outside probe.  A gauge shifting the whole
    region leaves exact in-tolerance jumps at both edges; a jump beyond
    center_tolerance leaves the edge endpoints unmatched instead of paired."""
    page = displace_fragments(_endpoint_rail_page(), [(400, 12)])
    region = (0, 100, 2400, 800)
    gauged = (
        FragmentTransform((0, 300), inverse_dx=-6),
        FragmentTransform((300, 700), inverse_dx=6),
    )

    report = _endpoint_report(page, gauged, region=region)

    assert len(report.internal) == len(_ENDPOINT_RAIL_XS)
    assert {pair.residual_quarter_px for pair in report.internal} == {0}
    top = [pair for pair in report.crop_edges if pair.boundary == 100]
    bottom = [pair for pair in report.crop_edges if pair.boundary == 800]
    assert len(top) == len(_ENDPOINT_RAIL_XS)
    assert len(bottom) == len(_ENDPOINT_RAIL_XS)
    assert {pair.residual_quarter_px for pair in top} == {4 * 6}
    assert {pair.residual_quarter_px for pair in bottom} == {4 * 6}
    assert report.unmatched_strong_source == 0
    assert report.clipped_strong_rails == 0

    beyond_tolerance = (
        FragmentTransform((0, 300), inverse_dx=3),
        FragmentTransform((300, 700), inverse_dx=15),
    )

    report_far = _endpoint_report(page, beyond_tolerance, region=region)

    top_far = [pair for pair in report_far.crop_edges if pair.boundary == 100]
    assert {pair.residual_quarter_px for pair in top_far} == {4 * 3}
    assert [pair for pair in report_far.crop_edges if pair.boundary == 800] == []
    assert report_far.unmatched_strong_source == 2 * len(_ENDPOINT_RAIL_XS)


def test_select_gauge_prefers_minimum_loss_anchor() -> None:
    """Spec test 5: with unequal fragment lengths where fragment 0 is NOT the
    minimum-loss anchor, the selected gauge translates the short fragment
    instead, without changing internal seam scores; every gauge records
    predicted and measured loss."""
    page = displace_fragments(build_lattice_page(), [(500, 24)])
    measurements = fragment_realign._rail_measurements_for_axis(page, axis="y")
    assert measurements is not None
    component = (_component_node(500, 24),)
    region = (0, 0, 1200, 1600)

    selection = fragment_realign._select_gauge(
        page, component, region=region, axis="y", measurements=measurements
    )

    assert selection.gauge_offset == 24
    assert selection.anchor_fragment_index == 1
    assert selection.anchor_fragment_index != 0
    assert len(selection.objective_tuple) == 7
    assert all(isinstance(term, int) for term in selection.objective_tuple)
    # Every distinct cumulative offset evaluated, with predicted and
    # measured loss recorded per gauge.
    assert len(selection.evaluated) == 2
    by_gauge = {record["gauge_offset"]: record for record in selection.evaluated}
    assert by_gauge[0.0]["predicted_loss_pixels"] == 24 * 1100
    assert by_gauge[0.0]["measured_loss_pixels"] == 24 * 1100
    assert by_gauge[24.0]["predicted_loss_pixels"] == 24 * 500
    assert by_gauge[24.0]["measured_loss_pixels"] == 24 * 500
    # The winning gauge reconstructs with the short fragment translated.
    shifts = [fragment.inverse_dx for fragment in selection.candidate.fragments]
    assert shifts == [-24, 0]
    # Gauge choice cannot change the relative internal seam scores.
    anchored_zero = fragment_realign._joint_component_score(
        component, gray=page, region=region, axis="y", measurements=measurements
    )
    gauged = fragment_realign._joint_component_score(
        component,
        gray=page,
        region=region,
        axis="y",
        measurements=measurements,
        candidate=selection.candidate,
    )
    assert anchored_zero is not None and gauged is not None
    assert gauged["mean_before"] == anchored_zero["mean_before"]
    assert gauged["mean_after"] == anchored_zero["mean_after"]


def _family_candidate(family: str, gain: float, *, region: tuple[int, int, int, int]):
    from dataclasses import replace

    seed = fragment_realign.apply_fragment_transforms(
        np.full((48, 64), 255, dtype=np.uint8),
        (FragmentTransform((0, 48)),),
        partition_axis="y",
    )
    terms: dict[str, float] = {"total_gain": gain, "non_rule_support": 100.0}
    if family == "method_rule_anchor_component":
        terms["method_rule_anchor"] = 1.0
        terms["method_rule_anchor_component"] = 1.0
    elif family != "method_seam":
        terms[family] = 1.0
    return replace(seed, region=region, score_terms=terms)


def _family_fixture() -> dict[str, list]:
    gains = {
        "method_rule_anchor_component": (0.5, 0.2),
        "method_rule_anchor": (0.9, 0.8),
        "method_pitch": (0.4, 0.3),
        "method_seam": (0.7, 0.1),
        "method_two_rail_bounded_local": (0.6, 0.05),
    }
    fixture: dict[str, list] = {}
    region_index = 0
    for family, family_gains in gains.items():
        members = []
        for gain in family_gains:
            region_index += 1
            members.append(
                _family_candidate(family, gain, region=(region_index, 0, 64, 48))
            )
        fixture[family] = members
    return fixture


def _shuffled(fixture: dict[str, list]) -> list:
    members = [candidate for family in fixture.values() for candidate in family]
    return [members[index] for index in (7, 2, 9, 0, 5, 3, 8, 1, 6, 4)]


def test_retain_by_family_round_robin_under_top_k_eight() -> None:
    """Spec test 11: every viable family's first candidate is retained before
    any family's second, in the fixed family order, with exact
    family_rank/retention_round/retention_slot values."""
    fixture = _family_fixture()

    retention = fragment_realign._retain_by_family(
        _shuffled(fixture), top_k=8, family_mode="all_family_diagnostic"
    )

    expected = [
        (fixture["method_rule_anchor_component"][0], "method_rule_anchor_component", 1, 1, 1),
        (fixture["method_rule_anchor"][0], "method_rule_anchor", 1, 1, 2),
        (fixture["method_pitch"][0], "method_pitch", 1, 1, 3),
        (fixture["method_seam"][0], "method_seam", 1, 1, 4),
        (fixture["method_two_rail_bounded_local"][0], "method_two_rail_bounded_local", 1, 1, 5),
        (fixture["method_rule_anchor_component"][1], "method_rule_anchor_component", 2, 2, 6),
        (fixture["method_rule_anchor"][1], "method_rule_anchor", 2, 2, 7),
        (fixture["method_pitch"][1], "method_pitch", 2, 2, 8),
    ]
    assert [
        (record[0].region, *record[1:]) for record in retention
    ] == [(candidate.region, *rest) for candidate, *rest in expected]


def test_retain_by_family_small_top_k_takes_fixed_family_order() -> None:
    """Spec test 11: with top_k below the viable-family count, exactly the
    first top_k families in the FIXED order are represented — even though the
    seam and two-rail candidates outscore the pitch candidate."""
    fixture = _family_fixture()

    retention = fragment_realign._retain_by_family(
        _shuffled(fixture), top_k=3, family_mode="all_family_diagnostic"
    )

    assert [(record[1], record[2], record[3], record[4]) for record in retention] == [
        ("method_rule_anchor_component", 1, 1, 1),
        ("method_rule_anchor", 1, 1, 2),
        ("method_pitch", 1, 1, 3),
    ]


def test_retain_by_family_active_default_drops_two_rail_family() -> None:
    fixture = _family_fixture()

    retention = fragment_realign._retain_by_family(
        _shuffled(fixture), top_k=8, family_mode="active_default"
    )

    families = [record[1] for record in retention]
    assert "method_two_rail_bounded_local" not in families
    assert families == [
        "method_rule_anchor_component",
        "method_rule_anchor",
        "method_pitch",
        "method_seam",
        "method_rule_anchor_component",
        "method_rule_anchor",
        "method_pitch",
        "method_seam",
    ]


def test_retain_by_family_retention_invariance() -> None:
    """Spec test 11 / §8.5: rescaling only one family's retention-layer
    total_gain cannot change the family round-robin, remove another family's
    representative, or change which families survive top_k; only
    within-family order may move."""
    from dataclasses import replace

    fixture = _family_fixture()
    base = fragment_realign._retain_by_family(
        _shuffled(fixture), top_k=8, family_mode="all_family_diagnostic"
    )

    def rescaled(transform):
        adjusted = dict(fixture)
        adjusted["method_pitch"] = [
            replace(
                candidate,
                score_terms={
                    **candidate.score_terms,
                    "total_gain": transform(candidate.score_terms["total_gain"]),
                },
            )
            for candidate in fixture["method_pitch"]
        ]
        return fragment_realign._retain_by_family(
            _shuffled(adjusted), top_k=8, family_mode="all_family_diagnostic"
        )

    monotone = rescaled(lambda gain: 3.0 * gain + 0.5)
    assert [(record[0].region, *record[1:]) for record in monotone] == [
        (record[0].region, *record[1:]) for record in base
    ]

    reversed_order = rescaled(lambda gain: 1.0 - gain)
    assert [record[1:] for record in reversed_order] == [record[1:] for record in base]
    for base_record, new_record in zip(base, reversed_order):
        if base_record[1] != "method_pitch":
            assert new_record[0].region == base_record[0].region
    pitch_regions_base = {r[0].region for r in base if r[1] == "method_pitch"}
    pitch_regions_new = {r[0].region for r in reversed_order if r[1] == "method_pitch"}
    assert pitch_regions_new == pitch_regions_base


def test_evaluator_emits_schema_v2_retention_fields(monkeypatch) -> None:
    """Spec test 11 tail: schema-v2 candidates carry string solver_family and
    positive one-based ints; pre-feature (schema v1) candidates omit all four
    fields."""
    from tools import evaluate_fragment_realign

    fixture = _family_fixture()
    retention = fragment_realign._retain_by_family(
        _shuffled(fixture), top_k=8, family_mode="all_family_diagnostic"
    )
    retained = sorted(
        (record[0] for record in retention),
        key=lambda candidate: -candidate.score_terms["total_gain"],
    )

    records = evaluate_fragment_realign._candidate_retention_records(
        retained, top_k=8, family_mode="all_family_diagnostic"
    )

    assert len(records) == len(retained)
    for record in records:
        assert isinstance(record["solver_family"], str)
        for key in ("family_rank", "retention_round", "retention_slot"):
            assert isinstance(record[key], int) and record[key] >= 1
    assert sorted(record["retention_slot"] for record in records) == list(range(1, 9))

    monkeypatch.setattr(fragment_realign, "_SCHEMA_VERSION", 1)
    legacy = evaluate_fragment_realign._candidate_retention_records(
        retained, top_k=8, family_mode="all_family_diagnostic"
    )
    assert legacy == [None] * len(retained)


_PASSING_CLOSURE = {
    "internal_matched": 6,
    "internal_median_quarter_px": 2,
    "internal_max_quarter_px": 6,
    "unmatched_strong_source": 0,
    "source_endpoint_absent": 0,
    "clipped_strong_rails": 0,
    "interior_uncovered_pixels": 0,
    "crop_edges": [
        {"boundary": 300, "matched_pairs": 3, "max_jump_quarter_px": 4,
         "max_worsening_quarter_px": 2},
        {"boundary": 900, "matched_pairs": 2, "max_jump_quarter_px": 6,
         "max_worsening_quarter_px": 4},
    ],
}


def _geo_record(
    *,
    rank: int,
    family: str,
    gain: float,
    region: tuple[int, int, int, int],
    fragments: list[tuple[tuple[int, int], int]],
    axis: str = "y",
    closure: dict | None = None,
    field_recovery_status: str = "underdetermined",
    overlap_pixels: int = 0,
    uncovered_pixels: int = 0,
    cropped_source_pixels: int = 0,
) -> dict:
    terms: dict[str, float] = {"total_gain": gain}
    if family == "method_rule_anchor_component":
        terms["method_rule_anchor"] = 1.0
        terms["method_rule_anchor_component"] = 1.0
    elif family != "method_seam":
        terms[family] = 1.0
    return {
        "rank": rank,
        "solver_family": family,
        "region": list(region),
        "partition_axis": axis,
        "fragments": [
            {
                "interval": list(interval),
                "inverse_dx": shift if axis == "y" else 0,
                "inverse_dy": shift if axis == "x" else 0,
            }
            for interval, shift in fragments
        ],
        "score_terms": terms,
        "overlap_pixels": overlap_pixels,
        "uncovered_pixels": uncovered_pixels,
        "cropped_source_pixels": cropped_source_pixels,
        "closure": dict(closure) if closure is not None else dict(_PASSING_CLOSURE),
        "field_recovery_status": field_recovery_status,
    }


def test_regional_outcome_underdetermined_on_family_margin() -> None:
    """Spec 6: a same-family overlapping distinct transform within the frozen
    0.03 margin breaks uniqueness; at or beyond the margin it does not."""
    from tools import evaluate_fragment_realign

    primary = _geo_record(
        rank=1, family="method_rule_anchor", gain=0.50,
        region=(0, 0, 1200, 1600), fragments=[((0, 500), 0), ((500, 1600), 24)],
    )
    near_tied = _geo_record(
        rank=2, family="method_rule_anchor", gain=0.48,
        region=(0, 0, 1200, 1600), fragments=[((0, 500), 0), ((500, 1600), 134)],
    )

    outcome, reason = evaluate_fragment_realign._regional_outcome(
        primary, [primary, near_tied]
    )
    assert outcome == "underdetermined"
    assert "margin" in reason

    clear = dict(near_tied, score_terms={**near_tied["score_terms"], "total_gain": 0.47})
    outcome, _ = evaluate_fragment_realign._regional_outcome(primary, [primary, clear])
    assert outcome != "underdetermined"

    # A near-tied re-crop of the SAME displacement is not a distinct
    # transform: the mapping agrees everywhere on shared support.
    recrop = _geo_record(
        rank=2, family="method_rule_anchor", gain=0.49,
        region=(0, 300, 1200, 900), fragments=[((0, 200), 0), ((200, 600), 24)],
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        primary, [primary, recrop]
    )
    assert outcome != "underdetermined"

    # Arthur's 2026-07-29 ruling (specs/2026-07-29-s6-passing-candidates-
    # ruling.md): a closure-FAILING near-tied alternative cannot veto.
    failing_vetoer = _geo_record(
        rank=2, family="method_rule_anchor", gain=0.48,
        region=(0, 0, 1200, 1600), fragments=[((0, 500), 0), ((500, 1600), 134)],
        closure={**_PASSING_CLOSURE, "unmatched_strong_source": 1},
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        primary, [primary, failing_vetoer]
    )
    assert outcome != "underdetermined"


def test_regional_outcome_underdetermined_on_cross_family_conflict() -> None:
    """Spec 6: passing candidates from different families that map a shared
    source pixel more than 2 pixels apart force underdetermined; agreement
    within 2 pixels is compatible coexistence."""
    from tools import evaluate_fragment_realign

    primary = _geo_record(
        rank=1, family="method_rule_anchor", gain=0.50,
        region=(0, 0, 1200, 1600), fragments=[((0, 500), 0), ((500, 1600), 24)],
    )
    conflicting = _geo_record(
        rank=2, family="method_pitch", gain=0.30,
        region=(0, 400, 1200, 900), fragments=[((0, 100), 0), ((100, 500), 21)],
    )

    outcome, reason = evaluate_fragment_realign._regional_outcome(
        primary, [primary, conflicting]
    )
    assert outcome == "underdetermined"
    assert "famil" in reason

    compatible = _geo_record(
        rank=2, family="method_pitch", gain=0.30,
        region=(0, 400, 1200, 900), fragments=[((0, 100), 0), ((100, 500), 23)],
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        primary, [primary, compatible]
    )
    assert outcome != "underdetermined"

    # Arthur's 2026-07-29 ruling: a closure-FAILING conflicting alternative
    # cannot veto; a closure-passing one (above) still does.
    failing_vetoer = dict(
        conflicting,
        closure={**_PASSING_CLOSURE, "clipped_strong_rails": 1},
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        primary, [primary, failing_vetoer]
    )
    assert outcome != "underdetermined"


@pytest.mark.parametrize(
    "mutation",
    [
        {"internal_median_quarter_px": 5},
        {"internal_max_quarter_px": 9},
        {"unmatched_strong_source": 1},
        {"clipped_strong_rails": 1},
        {"interior_uncovered_pixels": 3},
        {
            "crop_edges": [
                {"boundary": 300, "matched_pairs": 1, "max_jump_quarter_px": 4,
                 "max_worsening_quarter_px": 2}
            ]
        },
        {
            "crop_edges": [
                {"boundary": 300, "matched_pairs": 3, "max_jump_quarter_px": 9,
                 "max_worsening_quarter_px": 2}
            ]
        },
        {
            "crop_edges": [
                {"boundary": 300, "matched_pairs": 3, "max_jump_quarter_px": 8,
                 "max_worsening_quarter_px": 5}
            ]
        },
    ],
)
def test_regional_outcome_region_repair_requires_every_closure_gate(
    mutation: dict,
) -> None:
    """Spec 12.2/6.1: all gates pass -> region_repair; any single failure
    (median >1px, max >2px, closure failure, clipped rail, interior uncovered,
    <2 strong continuing tracks, edge jump >2px, continuous-rail worsening
    >1px) prevents it."""
    from tools import evaluate_fragment_realign

    passing = _geo_record(
        rank=1, family="method_rule_anchor_component", gain=0.50,
        region=(0, 200, 1200, 1000), fragments=[((0, 300), 0), ((300, 800), 24)],
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(passing, [passing])
    assert outcome == "region_repair"

    failing = _geo_record(
        rank=1, family="method_rule_anchor_component", gain=0.50,
        region=(0, 200, 1200, 1000), fragments=[((0, 300), 0), ((300, 800), 24)],
        closure={**_PASSING_CLOSURE, **mutation},
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(failing, [failing])
    assert outcome == "geometry_only"


def test_regional_outcome_region_repair_requires_zero_destination_overlap() -> None:
    from tools import evaluate_fragment_realign

    overlapping = _geo_record(
        rank=1, family="method_rule_anchor_component", gain=0.50,
        region=(0, 200, 1200, 1000), fragments=[((0, 300), 0), ((300, 800), 24)],
        overlap_pixels=5,
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        overlapping, [overlapping]
    )
    assert outcome == "geometry_only"


def test_regional_outcome_salvage_gated_by_seven_percent_loss() -> None:
    """Spec 12.3: a pixel-supported field with closure failure classifies as
    partial_field_salvage only while BOTH loss fractions stay at or under 7%
    and no interior uncovered component exists."""
    from tools import evaluate_fragment_realign

    region = (0, 0, 1000, 1000)  # area 1_000_000
    base = dict(
        rank=1, family="method_rule_anchor", gain=0.50, region=region,
        fragments=[((0, 500), 0), ((500, 1000), 24)],
        field_recovery_status="pixel_supported",
        closure={**_PASSING_CLOSURE, "clipped_strong_rails": 1},  # closure fails
    )

    at_ceiling = _geo_record(
        **base, uncovered_pixels=70_000, cropped_source_pixels=70_000
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(at_ceiling, [at_ceiling])
    assert outcome == "partial_field_salvage"

    over_destination = _geo_record(
        **base, uncovered_pixels=70_001, cropped_source_pixels=0
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        over_destination, [over_destination]
    )
    assert outcome == "geometry_only"

    over_source = _geo_record(
        **base, uncovered_pixels=0, cropped_source_pixels=70_001
    )
    outcome, _ = evaluate_fragment_realign._regional_outcome(
        over_source, [over_source]
    )
    assert outcome == "geometry_only"

    interior = _geo_record(**base)
    interior["closure"]["interior_uncovered_pixels"] = 4
    outcome, _ = evaluate_fragment_realign._regional_outcome(interior, [interior])
    assert outcome == "geometry_only"


def test_compose_compatible_partial_candidates() -> None:
    """Spec test 15: two compatible partial mappings (agreement on shared
    support, no provenance conflict) compose into one explicit reconstruction
    equal to the full repair; a conflicting pair refuses composition."""
    clean = build_lattice_page()
    damaged = displace_fragments(clean, [(500, 18), (1000, -30)])

    def partial(region_rows, fragments):
        y0, y1 = region_rows
        transforms = tuple(
            FragmentTransform(interval, inverse_dx=shift)
            for interval, shift in fragments
        )
        from dataclasses import replace

        return replace(
            fragment_realign.apply_fragment_transforms(
                damaged[y0:y1, :], transforms, partition_axis="y"
            ),
            region=(0, y0, 1200, y1),
        )

    first = partial((0, 760), [((0, 500), 0), ((500, 760), 18)])
    second = partial((600, 1600), [((0, 400), 18), ((400, 1000), -12)])

    composed = fragment_realign._compose_candidates(damaged, first, second)
    assert composed is not None
    assert composed.region == (0, 0, 1200, 1600)
    assert np.array_equal(composed.reconstruction, clean)

    conflicting = partial((600, 1600), [((0, 400), 21), ((400, 1000), -12)])
    assert fragment_realign._compose_candidates(damaged, first, conflicting) is None


class _SilentOcrEngine:
    """Deterministic empty OCR: geometry-only classification paths."""

    def words(self, image, sparse: bool = False):
        return []


def _assembled_run(
    tmp_path,
    page,
    *,
    run_id: str = "t10",
    source_pdf: str = "SYNTHETIC-000000.pdf",
    page_index: int = 0,
):
    from tools import evaluate_fragment_realign

    candidates, diagnostics = (
        fragment_realign._generate_page_repair_candidates_with_diagnostics(
            page, max_fragments=5, top_k=8, family_mode="all_family_diagnostic"
        )
    )
    serialized = evaluate_fragment_realign._serialized_view_candidates(
        tmp_path,
        run_id,
        page,
        candidates,
        top_k=8,
        family_mode="all_family_diagnostic",
    )
    return {
        "record_id": run_id,
        "source_pdf": source_pdf,
        "page_index": page_index,
        "rotation_k_ccw": 0,
        "geometry_candidate_count": len(serialized),
        "geometry_gate": "triggered" if serialized else "abstained: none",
        "candidates": serialized,
        "candidate_diagnostics": diagnostics,
    }


def test_report_preserves_records_and_never_claims_full_page(tmp_path) -> None:
    """Spec test 12: serialization preserves per-seam records, gauge
    diagnostics, provenance, closure, and classification — and no report
    field contains a full-page repair claim."""
    import json as json_module
    import re as re_module

    from tools import evaluate_fragment_realign

    run = _assembled_run(tmp_path, _component_fixture_page())
    evaluate_fragment_realign._evaluate_ocr(
        [run], tmp_path, engine=_SilentOcrEngine()
    )

    payload = json_module.loads(evaluate_fragment_realign._canonical_json(run))
    components = [
        candidate
        for candidate in payload["candidates"]
        if candidate["score_terms"].get("method_rule_anchor_component") == 1.0
    ]
    assert components
    for candidate in components:
        assert candidate["closure"]["crop_edges"]
        assert candidate["provenance_contract"]["reciprocal_visible_mapping"]
        assert isinstance(candidate["retention_slot"], int)
        signature_diagnostics = payload["candidate_diagnostics"]
        matching = [
            record
            for record in signature_diagnostics.values()
            if record.get("per_seam")
        ]
        assert matching
        assert any("gauge" in record for record in matching)
    assert payload["regional_reconstruction_outcome"] in (
        "underdetermined",
        "region_repair",
        "partial_field_salvage",
        "geometry_only",
    )
    assert payload["full_page_geometry_status"] in (
        "unverified",
        "pixel_unsupported",
    )
    claim = re_module.compile(
        r"full[- ]?page (geometry |alignment )?"
        r"(repair(ed)?|verified|proven|complete)|entire page repaired",
        re_module.IGNORECASE,
    )

    def walk(value):
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            assert not claim.search(value), value

    walk(payload)


def test_selected_candidate_schema_v2_fields(tmp_path, monkeypatch) -> None:
    """Spec 13: schema-v2 run records carry selected_candidate_retention_slot
    and selected_candidate_family_rank (null when nothing is selected); the
    legacy selected_candidate_rank keeps its meaning; Stage A1 records omit
    the new fields."""
    from tools import evaluate_fragment_realign

    run = _assembled_run(tmp_path, _component_fixture_page())
    evaluate_fragment_realign._evaluate_ocr(
        [run], tmp_path, engine=_SilentOcrEngine()
    )
    assert isinstance(run["selected_candidate_rank"], int)
    primary = next(
        candidate
        for candidate in run["candidates"]
        if candidate["rank"] == run["selected_candidate_rank"]
    )
    assert run["selected_candidate_retention_slot"] == primary["retention_slot"]
    assert run["selected_candidate_family_rank"] == primary["family_rank"]

    empty_run = {
        "record_id": "empty",
        "page_index": 0,
        "rotation_k_ccw": 0,
        "geometry_gate": "abstained: geometry generator returned no candidate",
        "candidates": [],
        "candidate_diagnostics": {},
    }
    evaluate_fragment_realign._evaluate_ocr(
        [empty_run], tmp_path, engine=_SilentOcrEngine()
    )
    assert empty_run["selected_candidate_rank"] is None
    assert empty_run["selected_candidate_retention_slot"] is None
    assert empty_run["selected_candidate_family_rank"] is None
    assert empty_run["regional_reconstruction_outcome"] is None

    monkeypatch.setattr(fragment_realign, "_SCHEMA_VERSION", 1)
    legacy_run = _assembled_run(tmp_path, _component_fixture_page(), run_id="a1")
    evaluate_fragment_realign._evaluate_ocr(
        [legacy_run], tmp_path, engine=_SilentOcrEngine()
    )
    assert "selected_candidate_retention_slot" not in legacy_run
    assert "selected_candidate_family_rank" not in legacy_run
    assert "regional_reconstruction_outcome" not in legacy_run


def test_annotation_isolation_leaves_geometry_byte_identical(
    tmp_path, monkeypatch
) -> None:
    """Spec test 14: removing AND mutating every annotation target (including
    'Nexvara Zarix') leaves geometry signatures, ranks, retention slots,
    transforms, provenance hashes, and the frozen geometry digest
    byte-identical."""
    from tools import evaluate_fragment_realign

    page = _component_fixture_page()

    def frozen_digest(state_dir):
        # A real annotation key (178 page 6 carries the Nexvara Zarix target)
        # so the annotation join genuinely exercises the mutated values.
        run = _assembled_run(
            state_dir,
            page,
            run_id="iso",
            source_pdf="MIB-000178.pdf",
            page_index=5,
        )
        payload = [
            {
                key: value
                for key, value in run.items()
                if key not in {"geometry_elapsed_seconds", "candidate_diagnostics"}
            }
        ]
        digest = evaluate_fragment_realign._sha256_bytes(
            evaluate_fragment_realign._canonical_json(payload).encode("utf-8")
        )
        evaluate_fragment_realign._evaluate_ocr(
            [run], state_dir, engine=_SilentOcrEngine()
        )
        evaluate_fragment_realign._join_reporting_annotations([run])
        return digest, [
            (
                candidate["rank"],
                candidate.get("retention_slot"),
                candidate["fragments"],
                candidate["asset_sha256"],
            )
            for candidate in run["candidates"]
        ]

    baseline = frozen_digest(tmp_path / "baseline")

    monkeypatch.setattr(evaluate_fragment_realign, "REVIEWED_ANNOTATIONS", {})
    removed = frozen_digest(tmp_path / "removed")

    mutated_annotations = {}
    for key, annotation in _ORIGINAL_ANNOTATIONS.items():
        entry = dict(annotation)
        if "expected_fields" in entry:
            entry["expected_fields"] = {
                field: f"MUTATED-{value}"
                for field, value in entry["expected_fields"].items()
            }
        mutated_annotations[key] = entry
    assert any(
        "Nexvara Zarix" in str(annotation.get("expected_fields", {}))
        for annotation in _ORIGINAL_ANNOTATIONS.values()
    )
    monkeypatch.setattr(
        evaluate_fragment_realign, "REVIEWED_ANNOTATIONS", mutated_annotations
    )
    mutated = frozen_digest(tmp_path / "mutated")

    assert baseline == removed == mutated


from tools import evaluate_fragment_realign as _evaluator_module

_ORIGINAL_ANNOTATIONS = dict(_evaluator_module.REVIEWED_ANNOTATIONS)


def test_transpose_equivalence_of_candidates_and_outcome(tmp_path) -> None:
    """Spec test 7: the component fixture and its transpose produce
    transpose-equivalent regions, transforms, reconstructions, masks, and
    bit-identical score terms; the evaluator outcome matches."""
    from tools import evaluate_fragment_realign

    page = _component_fixture_page()
    transposed = np.ascontiguousarray(page.T)

    original, _ = fragment_realign._generate_page_repair_candidates_with_diagnostics(
        page, max_fragments=5, top_k=8, family_mode="all_family_diagnostic"
    )
    swapped, _ = fragment_realign._generate_page_repair_candidates_with_diagnostics(
        transposed, max_fragments=5, top_k=8, family_mode="all_family_diagnostic"
    )

    def transposed_signature(candidate):
        x0, y0, x1, y1 = candidate.region
        return (
            (y0, x0, y1, x1),
            "x" if candidate.partition_axis == "y" else "y",
            tuple(
                (fragment.interval, fragment.inverse_dy, fragment.inverse_dx)
                for fragment in candidate.fragments
            ),
        )

    # The frozen legacy pitch family is row-oriented by design (it measures
    # text-row combs; orientation coverage comes from the evaluator's
    # four-rotation sweep — the 802 sentinel pitch lives at 90CW).  Spec
    # test 7 governs the regional feature: every NON-pitch candidate must
    # transpose exactly.
    original = [
        candidate
        for candidate in original
        if candidate.score_terms.get("method_pitch") != 1.0
    ]
    swapped = [
        candidate
        for candidate in swapped
        if candidate.score_terms.get("method_pitch") != 1.0
    ]
    by_expected = {
        transposed_signature(candidate): candidate for candidate in original
    }
    assert len(swapped) == len(original)
    for candidate in swapped:
        key = (
            tuple(candidate.region),
            candidate.partition_axis,
            tuple(
                (fragment.interval, fragment.inverse_dx, fragment.inverse_dy)
                for fragment in candidate.fragments
            ),
        )
        match = by_expected.get(key)
        assert match is not None, f"no transposed twin for {key}"
        assert candidate.score_terms == match.score_terms
        assert np.array_equal(candidate.reconstruction, match.reconstruction.T)
        assert np.array_equal(candidate.overlap_mask, match.overlap_mask.T)
        assert np.array_equal(candidate.uncovered_mask, match.uncovered_mask.T)
        assert np.array_equal(
            candidate.cropped_source_mask, match.cropped_source_mask.T
        )

    def classified_outcome(state_dir, image, members, run_id):
        serialized = evaluate_fragment_realign._serialized_view_candidates(
            state_dir,
            run_id,
            image,
            members,
            top_k=8,
            family_mode="all_family_diagnostic",
        )
        run = {
            "record_id": run_id,
            "source_pdf": "SYNTHETIC-000000.pdf",
            "page_index": 0,
            "rotation_k_ccw": 0,
            "geometry_gate": "triggered",
            "candidates": serialized,
            "candidate_diagnostics": {},
        }
        evaluate_fragment_realign._evaluate_ocr(
            [run], state_dir, engine=_SilentOcrEngine()
        )
        return run["regional_reconstruction_outcome"]

    assert classified_outcome(
        tmp_path / "orig", page, original, "orig"
    ) == classified_outcome(tmp_path / "swap", transposed, swapped, "swap")


def test_displaced_lattice_single_seam_repairs_to_byte_equality() -> None:
    """Pins the fixture builders to real solver semantics: a page damaged by
    ``displace_fragments`` must be repaired by the existing single-seam family
    back to byte-equality with the clean page."""
    clean = build_lattice_page()
    seams = [(760, 24)]
    damaged = displace_fragments(clean, seams)
    expected = _transforms_for_seams(seams, length=1600, axis="y")

    candidates = generate_page_repair_candidates(damaged, top_k=16)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "y"
        and candidate.region == (0, 0, 1200, 1600)
        and candidate.fragments == expected
    )
    assert np.array_equal(recovered.reconstruction, clean)
