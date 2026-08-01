from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from mib_pipeline import fragment_realign as fragment_realign_module
from mib_pipeline.fragment_realign import (
    FragmentTransform,
    apply_fragment_transforms,
    build_content_masks,
    estimate_periods,
    generate_page_repair_candidates,
    generate_repair_candidates,
    propose_candidate_regions,
)


def _field_image(height: int = 72, width: int = 96) -> np.ndarray:
    """A deterministic, asymmetric grayscale fixture with dark and gray ink."""
    image = np.full((height, width), 255, dtype=np.uint8)
    cv2.putText(image, "A7", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_8)
    cv2.putText(image, "Q2", (32, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1, cv2.LINE_8)
    cv2.rectangle(image, (61, 8), (84, 26), 148, thickness=-1)
    cv2.circle(image, (74, 52), 8, 78, thickness=2)
    return image


def _faint_rule_lattice(*, with_text: bool = True) -> np.ndarray:
    """Wide form whose faint rails cross two literal horizontal cut rows."""
    image = np.full((360, 640), 255, dtype=np.uint8)
    for x in (45, 220, 400, 595):
        cv2.line(image, (x, 18), (x, 341), 242, 2, cv2.LINE_8)
    if with_text:
        cv2.putText(
            image,
            "Sponsor ID: SPN-1680",
            (70, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            20,
            2,
            cv2.LINE_8,
        )
        cv2.putText(
            image,
            "Visa Class: XW-1",
            (70, 226),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            20,
            2,
            cv2.LINE_8,
        )
    return image


def _bounded_two_rail_lattice(*, with_text: bool = True) -> np.ndarray:
    """A two-rail field with ink that must reconnect across both seams."""
    image = np.full((220, 900), 255, dtype=np.uint8)
    for x in (160, 740):
        cv2.line(image, (x, 10), (x, 210), 242, 2, cv2.LINE_8)
    if with_text:
        cv2.putText(
            image,
            "ALPHA 23",
            (260, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            20,
            2,
            cv2.LINE_8,
        )
        cv2.putText(
            image,
            "BETA 58",
            (310, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            20,
            2,
            cv2.LINE_8,
        )
    return image


def _periodic_two_rail_lattice() -> np.ndarray:
    """A displaced two-rail field whose repeated marks make offsets ambiguous."""
    image = _bounded_two_rail_lattice(with_text=False)
    for y in (62, 128, 146):
        for x in range(246, 660, 24):
            cv2.rectangle(image, (x, y - 5), (x + 8, y + 5), 20, thickness=-1)
    return image


def _assert_no_two_rail_bounded_candidate(image: np.ndarray) -> None:
    for variant in (image, image.T):
        candidates = generate_page_repair_candidates(variant, top_k=32)
        assert all(
            candidate.score_terms.get("method_two_rail_bounded_local") != 1.0
            for candidate in candidates
        )


def _assert_final_bounded_canvas_identity(
    image: np.ndarray, candidate: object
) -> None:
    """Final fallback maps only its inverse-deskewed bounded band."""
    terms = candidate.score_terms
    x0 = int(terms["bounded_window_x0"])
    y0 = int(terms["bounded_window_y0"])
    x1 = int(terms["bounded_window_x1"])
    y1 = int(terms["bounded_window_y1"])
    local_upper = int(terms["local_upper_boundary"])
    local_lower = int(terms["local_lower_boundary"])
    local_height, local_width = y1 - y0, x1 - x0

    rectified_band = np.zeros((local_height, local_width), dtype=np.uint8)
    rectified_band[local_upper:local_lower, :] = 255
    inverse_matrix = cv2.getRotationMatrix2D(
        (local_width / 2.0, local_height / 2.0),
        -candidate.partition_angle_degrees,
        1.0,
    )
    local_band = (
        cv2.warpAffine(
            rectified_band,
            inverse_matrix,
            (local_width, local_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    canonical = image if candidate.partition_axis == "y" else image.T
    canonical_band = np.zeros(canonical.shape, dtype=bool)
    canonical_band[y0:y1, x0:x1] = local_band
    band = (
        canonical_band
        if candidate.partition_axis == "y"
        else canonical_band.T
    )

    height, width = image.shape
    identity = np.arange(height * width, dtype=np.int64).reshape(image.shape)
    outside = ~band
    assert np.array_equal(
        candidate.source_to_destination_map[outside],
        identity[outside],
    )
    assert np.array_equal(
        candidate.destination_to_source_map[outside],
        identity[outside],
    )
    assert np.array_equal(candidate.reconstruction[outside], image[outside])
    assert np.array_equal(candidate.source_view, image)
    visible = candidate.destination_to_source_map >= 0
    assert np.array_equal(
        candidate.reconstruction[visible],
        image.ravel()[candidate.destination_to_source_map[visible]],
    )

    final_loss = candidate.overlap_mask | candidate.uncovered_mask
    canonical_loss = (
        final_loss
        if candidate.partition_axis == "y"
        else final_loss.T
    )
    actual_page_loss = float(final_loss.sum() / final_loss.size)
    actual_local_loss = float(
        canonical_loss[y0:y1, x0:x1].sum()
        / max(local_height * local_width, 1)
    )
    assert terms["page_loss_fraction"] == actual_page_loss
    assert terms["local_window_loss_fraction"] == actual_local_loss
    assert actual_page_loss <= 0.02


def _eight_band_rule_lattice() -> np.ndarray:
    image = np.full((360, 640), 255, dtype=np.uint8)
    for x in (45, 220, 400, 595):
        cv2.line(image, (x, 12), (x, 347), 242, 2, cv2.LINE_8)
    for index, boundary in enumerate(range(45, 360, 45), start=1):
        cv2.putText(
            image,
            f"R{index} AB12",
            (75, boundary + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            20,
            1,
            cv2.LINE_8,
        )
    return image


def _periodic_alias_lattice(*, with_glyphs: bool = True) -> np.ndarray:
    """Eight bands whose periodic rails alone prefer the wrong pitch aliases."""
    image = np.full((400, 760), 255, dtype=np.uint8)
    for index, (start, end) in enumerate(
        zip(range(0, 400, 50), range(50, 401, 50))
    ):
        # Alternating the finite rail extent makes +/- one pitch as plausible
        # as the intact grid.  Only non-rule continuity identifies the page.
        rail_start = 40 if index % 2 == 0 else 96
        rail_end = 658 if index % 2 == 0 else 714
        for x in range(rail_start, rail_end, 56):
            cv2.line(image, (x, start + 2), (x, end - 2), 242, 2, cv2.LINE_8)
    if with_glyphs:
        for index, (boundary, label) in enumerate(
            zip(
                range(50, 400, 50),
                ("AX7Q", "B92M", "C4TZ", "D81K", "E3VR", "F76P", "G5WN"),
            )
        ):
            cv2.putText(
                image,
                label,
                (150 + (index * 37) % 180, boundary + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                20,
                2,
                cv2.LINE_8,
            )
            cv2.line(
                image,
                (500 - (index * 19) % 120, boundary - 12),
                (650 - (index * 7) % 80, boundary + 12),
                90,
                3,
                cv2.LINE_8,
            )
    return image


def _periodic_alias_transforms() -> tuple[FragmentTransform, ...]:
    offsets = (0, 17, -13, 22, -18, 11, 25, -7)
    return tuple(
        FragmentTransform((start, end), inverse_dy=offset)
        for start, end, offset in zip(
            range(0, 400, 50), range(50, 401, 50), offsets
        )
    )


def _damage_by_known_transforms(
    original: np.ndarray,
    transforms: tuple[FragmentTransform, ...],
    *,
    partition_axis: str,
) -> tuple[np.ndarray, object]:
    """Forward synthetic damage: the public inverse transform reverses it."""
    forward = tuple(
        FragmentTransform(
            interval=item.interval,
            inverse_dx=-item.inverse_dx,
            inverse_dy=-item.inverse_dy,
        )
        for item in transforms
    )
    result = apply_fragment_transforms(
        original,
        forward,
        partition_axis=partition_axis,
    )
    return result.reconstruction, result


def _assert_round_trip(
    original: np.ndarray, forward_candidate: object, repaired_candidate: object
) -> None:
    """Every source pixel that survived damage must return to its source index."""
    forward_map = forward_candidate.source_to_destination_map.ravel()
    repair_map = repaired_candidate.source_to_destination_map.ravel()
    retained_source = np.flatnonzero(forward_map >= 0)
    damaged_indices = forward_map[retained_source]
    final_indices = repair_map[damaged_indices]
    assert np.array_equal(final_indices, retained_source)
    assert np.array_equal(
        repaired_candidate.reconstruction.ravel()[final_indices],
        original.ravel()[retained_source],
    )


def test_exactly_restores_horizontal_bands_with_positive_sideways_shift() -> None:
    original = _field_image()
    transforms = (
        FragmentTransform((0, 36), inverse_dx=0),
        FragmentTransform((36, 72), inverse_dx=9),
    )
    damaged, forward = _damage_by_known_transforms(original, transforms, partition_axis="y")

    repaired = apply_fragment_transforms(damaged, transforms, partition_axis="y")

    _assert_round_trip(original, forward, repaired)
    assert repaired.fragments == transforms


def test_exactly_restores_horizontal_bands_with_negative_sideways_shift() -> None:
    original = _field_image()
    transforms = (
        FragmentTransform((0, 24), inverse_dx=0),
        FragmentTransform((24, 48), inverse_dx=-7),
        FragmentTransform((48, 72), inverse_dx=5),
    )
    damaged, forward = _damage_by_known_transforms(original, transforms, partition_axis="y")

    repaired = apply_fragment_transforms(damaged, transforms, partition_axis="y")

    _assert_round_trip(original, forward, repaired)


def test_exactly_restores_vertical_strips_with_vertical_shifts() -> None:
    original = _field_image(height=88, width=96)
    transforms = (
        FragmentTransform((0, 32), inverse_dy=0),
        FragmentTransform((32, 64), inverse_dy=11),
        FragmentTransform((64, 96), inverse_dy=-8),
    )
    damaged, forward = _damage_by_known_transforms(original, transforms, partition_axis="x")

    repaired = apply_fragment_transforms(damaged, transforms, partition_axis="x")

    _assert_round_trip(original, forward, repaired)


def test_apply_is_transpose_equivalent_for_the_canonical_solver() -> None:
    original = _field_image(height=80, width=100)
    horizontal = (
        FragmentTransform((0, 25), inverse_dx=0),
        FragmentTransform((25, 50), inverse_dx=6),
        FragmentTransform((50, 80), inverse_dx=-4),
    )

    direct = apply_fragment_transforms(original, horizontal, partition_axis="y")
    transposed = apply_fragment_transforms(
        original.T,
        tuple(
            FragmentTransform(item.interval, inverse_dy=item.inverse_dx)
            for item in horizontal
        ),
        partition_axis="x",
    )

    assert np.array_equal(direct.reconstruction.T, transposed.reconstruction)


def test_provenance_records_overlap_uncovered_and_cropped_source_pixels() -> None:
    source = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)
    repaired = apply_fragment_transforms(
        source,
        (
            FragmentTransform((0, 4), inverse_dx=3, inverse_dy=2),
            FragmentTransform((4, 8), inverse_dx=-2, inverse_dy=-1),
        ),
        partition_axis="y",
    )

    assert repaired.overlap_mask.any()
    assert repaired.uncovered_mask.any()
    assert repaired.cropped_source_mask.any()
    assert repaired.source_to_destination_map.shape == source.shape
    assert repaired.source_to_destination_map.dtype.kind in "iu"


def test_content_masks_keep_gray_patch_and_stamp_edges_when_rules_are_discounted() -> None:
    image = np.full((80, 120), 255, dtype=np.uint8)
    cv2.line(image, (0, 14), (119, 14), 0, 1)
    cv2.line(image, (18, 0), (18, 79), 0, 1)
    cv2.rectangle(image, (50, 26), (85, 55), 165, thickness=-1)
    cv2.circle(image, (92, 46), 11, 90, thickness=2)

    masks = build_content_masks(image)

    assert masks.rule_mask[14, 60]
    assert not masks.glyph_mask[14, 60]
    assert masks.structure_mask[40, 50]
    assert masks.structure_mask[46, 81]


def test_estimate_periods_finds_repeated_spacing_without_fixed_pitch() -> None:
    mask = np.zeros((96, 48), dtype=bool)
    for y in (8, 29, 50, 71):
        mask[y : y + 3, 8:40] = True

    periods = estimate_periods(mask, axis="y")

    assert 21 in periods


def test_generator_finds_vertical_shift_candidate_from_gray_anchor_fragments() -> None:
    original = np.full((80, 96), 255, dtype=np.uint8)
    cv2.rectangle(original, (8, 8), (31, 68), 150, thickness=-1)
    cv2.rectangle(original, (32, 8), (63, 68), 112, thickness=-1)
    cv2.rectangle(original, (64, 8), (88, 68), 174, thickness=-1)
    transforms = (
        FragmentTransform((0, 32), inverse_dy=0),
        FragmentTransform((32, 64), inverse_dy=8),
        FragmentTransform((64, 96), inverse_dy=-6),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_repair_candidates(damaged, max_fragments=4, top_k=8)

    exact = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "x" and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, exact)


def test_generator_exactly_repairs_split_stamp_without_ocr_or_angle_metadata() -> None:
    original = np.full((84, 120), 255, dtype=np.uint8)
    cv2.putText(original, "STAMP", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 50, 2, cv2.LINE_8)
    cv2.line(original, (0, 22), (119, 28), 135, 3)
    cv2.rectangle(original, (40, 10), (76, 75), 170, thickness=-1)
    transforms = (
        FragmentTransform((0, 42), inverse_dx=0),
        FragmentTransform((42, 84), inverse_dx=7),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    candidates = generate_repair_candidates(damaged, max_fragments=3, top_k=8)

    exact = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "y" and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, exact)


def test_generator_abstains_for_clean_and_skew_only_images() -> None:
    clean = _field_image()
    matrix = cv2.getRotationMatrix2D((clean.shape[1] / 2, clean.shape[0] / 2), 1.0, 1.0)
    skewed = cv2.warpAffine(clean, matrix, (clean.shape[1], clean.shape[0]), borderValue=255)

    assert generate_repair_candidates(clean) == []
    assert generate_repair_candidates(skewed) == []


def test_row_pitch_solver_recovers_offsets_across_blank_vertical_seams() -> None:
    original = np.full((132, 180), 255, dtype=np.uint8)
    for center in (18, 40, 62, 84, 106):
        for start, end in ((5, 55), (65, 115), (125, 175)):
            cv2.putText(
                original,
                "AB12",
                (start, center + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                20,
                1,
                cv2.LINE_8,
            )
    transforms = (
        FragmentTransform((0, 60), inverse_dy=0),
        FragmentTransform((60, 120), inverse_dy=10),
        FragmentTransform((120, 180), inverse_dy=-7),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_repair_candidates(damaged, max_fragments=4, top_k=16)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "x"
        and [item.inverse_dy for item in candidate.fragments] == [0, 10, -7]
        and abs(candidate.fragments[0].interval[1] - 60) <= 8
        and abs(candidate.fragments[1].interval[1] - 120) <= 8
    )
    del forward
    supported_ink = original < 250
    assert np.array_equal(
        recovered.reconstruction[supported_ink], original[supported_ink]
    )


def test_page_region_proposal_is_generic_and_page_wrapper_preserves_coordinates() -> None:
    page = np.full((260, 340), 255, dtype=np.uint8)
    field = _field_image(height=88, width=150)
    page[70:158, 90:240] = field

    regions = propose_candidate_regions(page)
    candidates = generate_page_repair_candidates(page)

    assert any(x0 <= 90 and y0 <= 70 and x1 >= 240 and y1 >= 158 for x0, y0, x1, y1 in regions)
    assert candidates == []


def test_page_wrapper_uses_faint_rule_lattice_to_repair_horizontal_bands() -> None:
    original = _faint_rule_lattice()
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    candidates = generate_page_repair_candidates(damaged, top_k=16)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "y"
        and candidate.region == (0, 0, 640, 360)
        and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, recovered)


def test_rule_lattice_repair_is_transpose_equivalent() -> None:
    original = _faint_rule_lattice().T
    transforms = (
        FragmentTransform((0, 110), inverse_dy=0),
        FragmentTransform((110, 220), inverse_dy=21),
        FragmentTransform((220, 360), inverse_dy=18),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_page_repair_candidates(damaged, top_k=16)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "x"
        and candidate.region == (0, 0, 360, 640)
        and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, recovered)


def test_bounded_two_rail_fallback_recovers_only_the_closed_middle_band() -> None:
    original = _bounded_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    candidates = generate_page_repair_candidates(damaged, top_k=32)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.score_terms.get("method_two_rail_bounded_local") == 1.0
        and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, recovered)
    assert recovered.score_terms["page_loss_fraction"] <= 0.02


def test_bounded_two_rail_fallback_is_transpose_equivalent() -> None:
    original = _bounded_two_rail_lattice().T
    transforms = (
        FragmentTransform((0, 72), inverse_dy=0),
        FragmentTransform((72, 138), inverse_dy=48),
        FragmentTransform((138, 220), inverse_dy=0),
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_page_repair_candidates(damaged, top_k=32)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.score_terms.get("method_two_rail_bounded_local") == 1.0
        and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, recovered)
    assert recovered.score_terms["page_loss_fraction"] <= 0.02


def test_bounded_two_rail_nonzero_hypothesis_composes_rectified_provenance(
    monkeypatch: object,
) -> None:
    """A selected local angle must be embodied in pixels and provenance."""
    original = _bounded_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    def injected_hypothesis(
        measurements: object, seed: object
    ) -> list[object]:
        inverse_offset = 48 if seed.boundary < 100 else -48
        return [
            fragment_realign_module._TwoRailBoundaryHypothesis(
                boundary=seed.boundary,
                inverse_offset=inverse_offset,
                angle=1.0,
                angle_margin=0.25,
                continuity_before=0.1,
                continuity_after=0.7,
                continuity_gain=0.6,
                literal_gain=0.6,
                non_rule_gain=0.6,
                offset_margin=0.25,
                support_bin_count=3,
                support_bin_span=2,
                matched_span=580.0,
                matched_span_threshold=230.0,
                matched_fraction=1.0,
                pair_residual=0.0,
                maximum_inverse_offset=220,
                local_window_loss_fraction=0.01,
                boundary_stability=1.0,
                offset_stability=0.0,
                angle_competitor_count=1,
                angle_rejected_competitor_count=0,
                x0=seed.x0,
                x1=seed.x1,
                y0=seed.y0,
                y1=seed.y1,
            )
        ]

    monkeypatch.setattr(
        fragment_realign_module,
        "_stable_two_rail_hypotheses",
        injected_hypothesis,
    )

    recovered = next(
        candidate
        for candidate in fragment_realign_module._bounded_two_rail_candidates_for_axis(
            damaged, axis="y"
        )
        if candidate.fragments == transforms
    )

    assert recovered.partition_angle_degrees == 1.0
    _assert_final_bounded_canvas_identity(damaged, recovered)


def test_bounded_two_rail_public_solver_recovers_rotated_lattice_and_transpose() -> None:
    original = _bounded_two_rail_lattice()
    cv2.putText(
        original,
        "Z9",
        (590, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        20,
        2,
        cv2.LINE_8,
    )
    cv2.putText(
        original,
        "K4",
        (590, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        20,
        2,
        cv2.LINE_8,
    )
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )
    height, width = damaged.shape
    skew_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), 1.0, 1.0
    )
    skewed = cv2.warpAffine(
        damaged,
        skew_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    transposed_transforms = tuple(
        FragmentTransform(
            fragment.interval,
            inverse_dy=fragment.inverse_dx,
        )
        for fragment in transforms
    )

    for image, axis, expected_transforms in (
        (skewed, "y", transforms),
        (skewed.T, "x", transposed_transforms),
    ):
        recovered = next(
            candidate
            for candidate in generate_page_repair_candidates(image, top_k=32)
            if candidate.score_terms.get("method_two_rail_bounded_local") == 1.0
            and abs(candidate.partition_angle_degrees) >= 0.5
        )
        assert recovered.partition_axis == axis
        assert recovered.fragments == expected_transforms

        _assert_final_bounded_canvas_identity(image, recovered)


def test_bounded_two_rail_score_terms_use_each_actual_evaluation_window() -> None:
    original = _bounded_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    recovered = next(
        candidate
        for candidate in generate_page_repair_candidates(damaged, top_k=32)
        if candidate.score_terms.get("method_two_rail_bounded_local") == 1.0
        and candidate.fragments == transforms
    )
    terms = recovered.score_terms
    _assert_final_bounded_canvas_identity(damaged, recovered)
    assert terms["upper_matched_span_threshold"] == 0.35 * 681.0
    assert terms["lower_matched_span_threshold"] == 0.35 * 681.0
    assert terms["upper_maximum_inverse_offset"] == 681 // 3
    assert terms["lower_maximum_inverse_offset"] == 681 // 3


def test_bounded_angle_margin_rejects_half_degree_tie_and_missing_competitor(
    monkeypatch: object,
) -> None:
    original = _bounded_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )
    measurements = fragment_realign_module._rail_measurements_for_axis(
        damaged, axis="y"
    )
    seed = fragment_realign_module._two_rail_windows(measurements)[0]

    def hypothesis(angle: float, score: float) -> object:
        return fragment_realign_module._TwoRailBoundaryHypothesis(
            boundary=seed.boundary,
            inverse_offset=48,
            angle=angle,
            angle_margin=0.0,
            continuity_before=0.1,
            continuity_after=score,
            continuity_gain=score - 0.1,
            literal_gain=0.5,
            non_rule_gain=0.5,
            offset_margin=0.2,
            support_bin_count=3,
            support_bin_span=2,
            matched_span=580.0,
            matched_span_threshold=230.0,
            matched_fraction=1.0,
            pair_residual=0.0,
            maximum_inverse_offset=220,
            local_window_loss_fraction=0.01,
            boundary_stability=0.0,
            offset_stability=0.0,
            angle_competitor_count=0,
            angle_rejected_competitor_count=0,
            x0=seed.x0,
            x1=seed.x1,
            y0=seed.y0,
            y1=seed.y1,
        )

    monkeypatch.setattr(
        fragment_realign_module,
        "_near_axis_hough_modes",
        lambda gray, *, rail_length: [0.5],
    )

    def tied_evaluation(
        canonical: np.ndarray,
        event: object,
        *,
        angle: float,
        enforce_score_gates: bool = True,
    ) -> object:
        del canonical, event, enforce_score_gates
        score = 0.70 if angle <= 0.25 else 0.68
        return hypothesis(angle, score)

    monkeypatch.setattr(
        fragment_realign_module,
        "_evaluate_two_rail_boundary",
        tied_evaluation,
    )
    assert (
        fragment_realign_module._stable_two_rail_hypotheses(
            measurements, seed
        )
        == []
    )

    def missing_evaluation(
        canonical: np.ndarray,
        event: object,
        *,
        angle: float,
        enforce_score_gates: bool = True,
    ) -> object:
        del canonical, event, enforce_score_gates
        if angle == 0.5:
            return None
        return hypothesis(angle, 0.70)

    monkeypatch.setattr(
        fragment_realign_module,
        "_evaluate_two_rail_boundary",
        missing_evaluation,
    )
    assert (
        fragment_realign_module._stable_two_rail_hypotheses(
            measurements, seed
        )
        == []
    )


def test_bounded_two_rail_fallback_abstains_without_glyph_support() -> None:
    rules_only = _bounded_two_rail_lattice(with_text=False)
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        rules_only, transforms, partition_axis="y"
    )

    _assert_no_two_rail_bounded_candidate(damaged)


def test_bounded_two_rail_fallback_abstains_for_open_excursion() -> None:
    original = _bounded_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 220), inverse_dx=48),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    _assert_no_two_rail_bounded_candidate(damaged)


def test_bounded_two_rail_fallback_abstains_for_nonreciprocal_exit() -> None:
    original = _bounded_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=19),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    _assert_no_two_rail_bounded_candidate(damaged)


def test_bounded_two_rail_fallback_abstains_for_clean_common_skew() -> None:
    clean = _bounded_two_rail_lattice()
    matrix = cv2.getRotationMatrix2D(
        (clean.shape[1] / 2, clean.shape[0] / 2), 2.0, 1.0
    )
    skewed = cv2.warpAffine(
        clean, matrix, (clean.shape[1], clean.shape[0]), borderValue=255
    )

    _assert_no_two_rail_bounded_candidate(skewed)


def test_bounded_two_rail_fallback_abstains_for_gray_paste_endpoints() -> None:
    pasted = _bounded_two_rail_lattice()
    cv2.rectangle(pasted, (310, 70), (590, 140), 225, thickness=-1)

    _assert_no_two_rail_bounded_candidate(pasted)


def test_bounded_two_rail_fallback_abstains_for_near_tied_periodic_modes() -> None:
    original = _periodic_two_rail_lattice()
    transforms = (
        FragmentTransform((0, 72), inverse_dx=0),
        FragmentTransform((72, 138), inverse_dx=48),
        FragmentTransform((138, 220), inverse_dx=0),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    _assert_no_two_rail_bounded_candidate(damaged)


def test_strong_detector_preserves_rule_anchor_signature() -> None:
    original = _faint_rule_lattice()
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    candidates = generate_page_repair_candidates(damaged, top_k=32)

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "y"
        and candidate.region == (0, 0, 640, 360)
        and candidate.fragments == transforms
    )
    assert recovered.score_terms["method_rule_anchor"] == 1.0
    assert recovered.score_terms["matched_rail_count"] == 3.0
    assert recovered.score_terms["fragment_count"] == 3.0
    assert recovered.score_terms.get("method_two_rail_bounded_local") != 1.0


def test_rule_anchor_scores_improvement_over_the_unshifted_seam() -> None:
    original = _faint_rule_lattice()
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    recovered = next(
        candidate
        for candidate in generate_page_repair_candidates(damaged, top_k=32)
        if candidate.partition_axis == "y"
        and candidate.region == (0, 0, 640, 360)
        and candidate.fragments == transforms
    )

    assert 0.0 < recovered.pixel_score_before < recovered.pixel_score_after
    assert recovered.score_terms["non_rule_continuity_before"] == (
        recovered.pixel_score_before
    )
    assert recovered.score_terms["non_rule_continuity_after"] == (
        recovered.pixel_score_after
    )
    expected_gain = (
        recovered.pixel_score_after
        - recovered.pixel_score_before
        - 0.015 * (len(recovered.fragments) - 1)
        - 0.35 * recovered.score_terms["loss_fraction"]
    )
    assert np.isclose(recovered.score_terms["total_gain"], expected_gain)


def test_page_top_k_retains_a_viable_pitch_family(
    monkeypatch: object,
) -> None:
    page = np.full((48, 64), 255, dtype=np.uint8)
    seed = apply_fragment_transforms(
        page,
        (FragmentTransform((0, 48)),),
        partition_axis="y",
    )

    def candidate(method: str, gain: float) -> object:
        return replace(
            seed,
            pixel_score_before=0.2,
            pixel_score_after=0.2 + gain,
            score_terms={
                f"method_{method}": 1.0,
                "non_rule_support": 100.0,
                "total_gain": gain,
            },
        )

    rule_candidates = [
        candidate("rule_anchor", gain)
        for gain in (0.9, 0.8, 0.7)
    ]
    pitch_candidate = candidate("pitch", 0.4)
    monkeypatch.setattr(
        fragment_realign_module,
        "_rule_anchor_candidates",
        lambda _image, *, max_fragments: rule_candidates,
    )
    monkeypatch.setattr(
        fragment_realign_module,
        "propose_candidate_regions",
        lambda _image, *, max_regions: [(0, 0, 64, 48)],
    )
    monkeypatch.setattr(
        fragment_realign_module,
        "generate_repair_candidates",
        lambda _crop, *, max_fragments, top_k: [pitch_candidate],
    )

    retained = generate_page_repair_candidates(page, top_k=2)

    assert len(retained) == 2
    assert retained[0].score_terms["method_rule_anchor"] == 1.0
    assert retained[1].score_terms["method_pitch"] == 1.0


def test_page_rejects_rule_anchor_with_weak_relative_glyph_support(
    monkeypatch: object,
) -> None:
    page = np.full((48, 64), 255, dtype=np.uint8)
    seed = apply_fragment_transforms(
        page,
        (FragmentTransform((0, 48)),),
        partition_axis="y",
    )
    weak = replace(
        seed,
        pixel_score_before=0.05,
        pixel_score_after=0.95,
        score_terms={
            "method_rule_anchor": 1.0,
            "non_rule_support": 10.0,
            "total_gain": 0.9,
        },
    )
    supported = replace(
        seed,
        pixel_score_before=0.2,
        pixel_score_after=0.7,
        score_terms={
            "method_rule_anchor": 1.0,
            "non_rule_support": 100.0,
            "total_gain": 0.5,
        },
    )
    monkeypatch.setattr(
        fragment_realign_module,
        "_rule_anchor_candidates",
        lambda _image, *, max_fragments: [weak, supported],
    )
    monkeypatch.setattr(
        fragment_realign_module,
        "propose_candidate_regions",
        lambda _image, *, max_regions: [],
    )

    retained = generate_page_repair_candidates(page, top_k=8)

    assert retained == [supported]


def test_rule_lattice_supports_eight_transposed_fragments() -> None:
    original = _eight_band_rule_lattice().T
    offsets = (0, 9, -7, 14, -12, 6, 18, -3)
    edges = tuple(range(0, 361, 45))
    transforms = tuple(
        FragmentTransform((start, end), inverse_dy=offset)
        for start, end, offset in zip(edges, edges[1:], offsets)
    )
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_page_repair_candidates(
        damaged, max_fragments=8, top_k=16
    )

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "x"
        and candidate.region == (0, 0, 360, 640)
        and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, recovered)


def test_periodic_rule_aliases_use_non_rule_continuity_for_exact_eight_strip_chain() -> None:
    original = _periodic_alias_lattice().T
    transforms = _periodic_alias_transforms()
    damaged, forward = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_page_repair_candidates(
        damaged, max_fragments=8, top_k=32
    )

    recovered = next(
        candidate
        for candidate in candidates
        if candidate.partition_axis == "x"
        and candidate.region == (0, 0, 400, 760)
        and candidate.fragments == transforms
    )
    _assert_round_trip(original, forward, recovered)


def test_periodic_rules_without_non_rule_alias_evidence_abstain() -> None:
    original = _periodic_alias_lattice(with_glyphs=False).T
    transforms = _periodic_alias_transforms()
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="x"
    )

    candidates = generate_page_repair_candidates(
        damaged, max_fragments=8, top_k=32
    )

    assert all(
        candidate.score_terms.get("method_rule_anchor") != 1.0
        for candidate in candidates
    )


def test_rule_anchor_keeps_local_repairs_when_a_full_chain_is_available() -> None:
    original = _faint_rule_lattice()
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    candidates = generate_page_repair_candidates(
        damaged, max_fragments=5, top_k=16
    )

    assert any(
        candidate.score_terms.get("method_rule_anchor") == 1.0
        and candidate.region == (0, 0, 640, 360)
        and candidate.fragments == transforms
        for candidate in candidates
    )
    assert any(
        candidate.score_terms.get("method_rule_anchor") == 1.0
        and candidate.region != (0, 0, 640, 360)
        and len(candidate.fragments) == 2
        for candidate in candidates
    )


def test_rule_lattice_ignores_continuous_vertical_distractors() -> None:
    original = _faint_rule_lattice()
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )
    for x in range(12, damaged.shape[1], 17):
        cv2.line(damaged, (x, 0), (x, damaged.shape[0] - 1), 238, 1, cv2.LINE_8)

    candidates = generate_page_repair_candidates(damaged, top_k=16)

    assert any(
        candidate.partition_axis == "y"
        and candidate.region == (0, 0, 640, 360)
        and candidate.fragments == transforms
        for candidate in candidates
    )


def test_many_rule_seams_emit_local_two_band_candidates() -> None:
    original = np.full((360, 640), 255, dtype=np.uint8)
    for x in (45, 220, 400, 595):
        cv2.line(original, (x, 8), (x, 351), 242, 2, cv2.LINE_8)
    for index, boundary in enumerate((60, 120, 180, 240, 300), start=1):
        cv2.putText(
            original,
            f"FIELD {index} AB12",
            (72, boundary + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            20,
            1,
            cv2.LINE_8,
        )
    offsets = (0, 12, -9, 20, -15, 8)
    edges = (0, 60, 120, 180, 240, 300, 360)
    transforms = tuple(
        FragmentTransform((start, end), inverse_dx=offset)
        for start, end, offset in zip(edges, edges[1:], offsets)
    )
    damaged, _ = _damage_by_known_transforms(
        original, transforms, partition_axis="y"
    )

    candidates = generate_page_repair_candidates(
        damaged, max_fragments=5, top_k=32
    )

    assert any(
        candidate.partition_axis == "y"
        and candidate.region[1] < 180 < candidate.region[3]
        and len(candidate.fragments) == 2
        and candidate.fragments[1].inverse_dx
        - candidate.fragments[0].inverse_dx
        == 29
        for candidate in candidates
    )


def test_rule_anchor_abstains_for_smooth_one_pixel_drift() -> None:
    clean = _faint_rule_lattice()
    smooth_drift = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=1),
        FragmentTransform((220, 360), inverse_dx=2),
    )
    damaged, _ = _damage_by_known_transforms(
        clean, smooth_drift, partition_axis="y"
    )

    candidates = generate_page_repair_candidates(damaged, top_k=16)

    assert all(
        candidate.score_terms.get("method_rule_anchor") != 1.0
        for candidate in candidates
    )


def test_displaced_rule_lattice_without_glyph_support_abstains() -> None:
    rules_only = _faint_rule_lattice(with_text=False)
    transforms = (
        FragmentTransform((0, 110), inverse_dx=0),
        FragmentTransform((110, 220), inverse_dx=21),
        FragmentTransform((220, 360), inverse_dx=18),
    )
    damaged, _ = _damage_by_known_transforms(
        rules_only, transforms, partition_axis="y"
    )

    assert generate_page_repair_candidates(damaged, top_k=16) == []


def test_repeated_clean_rows_with_common_skew_do_not_become_strip_steps() -> None:
    clean = np.full((180, 240), 255, dtype=np.uint8)
    for y in (25, 55, 85, 115, 145):
        cv2.putText(
            clean,
            "Label        Value",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            20,
            1,
            cv2.LINE_8,
        )
    matrix = cv2.getRotationMatrix2D(
        (clean.shape[1] / 2, clean.shape[0] / 2), 1.5, 1.0
    )
    skewed = cv2.warpAffine(clean, matrix, (clean.shape[1], clean.shape[0]), borderValue=255)

    assert generate_repair_candidates(skewed, top_k=16) == []


def test_shallow_footer_cluster_touching_page_edge_is_not_a_local_region() -> None:
    page = np.full((1200, 900), 255, dtype=np.uint8)
    cv2.line(page, (40, 35), (520, 35), 20, 4)
    cv2.putText(
        page,
        "Synthetic challenge document",
        (70, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        20,
        2,
        cv2.LINE_8,
    )

    assert propose_candidate_regions(page) == []
