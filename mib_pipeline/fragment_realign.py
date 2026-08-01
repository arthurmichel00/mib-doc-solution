"""Geometry-only reconstruction of locally displaced document fragments.

The module deliberately has no OCR, parser, policy, case-id, or pipeline
imports.  It accepts one grayscale region and returns inspectable geometry
candidates.  A caller may later OCR the unchanged and reconstructed views with
identical settings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Literal

import cv2
import numpy as np


Axis = Literal["x", "y"]


@dataclass(frozen=True)
class FragmentTransform:
    """Inverse translation for one contiguous interval of the source image."""

    interval: tuple[int, int]
    inverse_dx: int = 0
    inverse_dy: int = 0

    def __post_init__(self) -> None:
        start, end = self.interval
        if start < 0 or end <= start:
            raise ValueError("fragment interval must be a non-empty non-negative range")


@dataclass(frozen=True)
class ContentMasks:
    """Separate glyph, long-rule, and non-text structural evidence."""

    glyph_mask: np.ndarray
    rule_mask: np.ndarray
    structure_mask: np.ndarray


@dataclass(frozen=True)
class RepairCandidate:
    """One reconstruction plus enough provenance to audit every moved pixel.

    ``destination_to_source_map`` is destination-shaped and stores flattened
    source indices. ``source_to_destination_map`` is source-shaped and stores
    flattened destination indices. Negative entries mean no visible mapping.
    """

    rotation_k_ccw: int
    rotation_human: str
    region: tuple[int, int, int, int]
    partition_axis: Axis
    partition_angle_degrees: float
    fragments: tuple[FragmentTransform, ...]
    source_to_destination_map: np.ndarray
    destination_to_source_map: np.ndarray
    overlap_mask: np.ndarray
    uncovered_mask: np.ndarray
    cropped_source_mask: np.ndarray
    pixel_score_before: float
    pixel_score_after: float
    score_terms: dict[str, float]
    source_view: np.ndarray
    reconstruction: np.ndarray


@dataclass(frozen=True)
class _SeamHypothesis:
    boundary: int
    relative_offset: int
    similarity_before: float
    similarity_after: float
    edge_similarity_before: float
    edge_similarity_after: float
    shared_support: float
    gain: float
    objective: float


@dataclass(frozen=True)
class _MatchedRailPair:
    """One matched rail endpoint pair across a seam; id -1 when a traced
    center cannot be attributed to a measured rail within tolerance."""

    ending_rail_id: int
    starting_rail_id: int
    ending_center: float
    starting_center: float
    ending_extent: tuple[int, int]
    starting_extent: tuple[int, int]


@dataclass(frozen=True)
class _AnchoredSeam:
    """Structured rule-anchor seam evidence (replaces the anonymous 6-tuples).

    ``objective`` carries the legacy tuple item [0] so selection and ordering
    stay byte-identical; ``alias_margin`` carries legacy item [5], which on
    the refinement path historically held the continuity value. The seam's
    ``similarity_after`` is the legacy item [4] continuity."""

    seam: _SeamHypothesis
    source: str  # "rail_endpoint" | "glyph_continuity"
    matched_rail_count: int
    non_rule_support: float
    alias_margin: float
    matched_pairs: tuple[_MatchedRailPair, ...]
    support_interval: tuple[int, int]
    objective: float


@dataclass(frozen=True)
class _RailTrack:
    """One traced thin-rail track on the canonical partition axis.

    ``row_centers`` holds the per-row traced center over ``extent`` (start
    inclusive, end exclusive); NaN marks a row where no run was traceable."""

    center: float
    extent: tuple[int, int]
    row_centers: tuple[float, ...]


@dataclass(frozen=True)
class _EndpointResidual:
    """One matched endpoint pair under a proposed transform (spec 12.2).

    Centers are source canonical coordinates; the residual is the absolute
    post-transform center difference in integer quarter-pixels."""

    boundary: int
    before_rail_id: int
    after_rail_id: int
    before_center: float
    after_center: float
    residual_quarter_px: int


@dataclass(frozen=True)
class _EndpointReport:
    """Spec 12.2 endpoint measurement for one proposed reconstruction."""

    internal: tuple[_EndpointResidual, ...]
    crop_edges: tuple[_EndpointResidual, ...]
    unmatched_strong_source: int
    source_endpoint_absent: int
    clipped_strong_rails: int


@dataclass(frozen=True)
class _PitchRefinement:
    """Spec 11: residual-refined pitch candidate plus the exposed original."""

    refined: RepairCandidate
    unrefined: RepairCandidate
    deltas: tuple[int, ...]
    refinement_score: float
    runner_up: tuple[tuple[int, ...], float]


@dataclass(frozen=True)
class _GaugeSelection:
    """Spec 9.2 loss-minimizing common-translation choice for one component."""

    gauge_offset: int
    anchor_fragment_index: int
    objective_tuple: tuple[int, int, int, int, int, int, int]
    evaluated: tuple[dict[str, float], ...]
    candidate: RepairCandidate


@dataclass(frozen=True)
class _RailMeasurements:
    """Canonical faint-rule measurements shared by the rail solvers."""

    canonical: np.ndarray
    local_contrast: np.ndarray
    rail_mask: np.ndarray
    rails: tuple[tuple[float, int, int], ...]
    rail_margin: np.ndarray
    non_rule: np.ndarray
    rail_length: int
    max_rail_width: int
    min_fragment: int
    endpoint_tolerance: int
    center_tolerance: int


@dataclass(frozen=True)
class _TwoRailWindow:
    """One endpoint-rich event and its bounded canonical evaluation window."""

    boundary: int
    x0: int
    y0: int
    x1: int
    y1: int
    angle_hint: float = 0.0


@dataclass(frozen=True)
class _TwoRailBoundaryHypothesis:
    """A stable two-rail explanation for one side of a closed excursion."""

    boundary: int
    inverse_offset: int
    angle: float
    angle_margin: float
    continuity_before: float
    continuity_after: float
    continuity_gain: float
    literal_gain: float
    non_rule_gain: float
    offset_margin: float
    support_bin_count: int
    support_bin_span: int
    matched_span: float
    matched_span_threshold: float
    matched_fraction: float
    pair_residual: float
    maximum_inverse_offset: int
    local_window_loss_fraction: float
    boundary_stability: float
    offset_stability: float
    angle_competitor_count: int
    angle_rejected_competitor_count: int
    x0: int
    x1: int
    y0: int
    y1: int


def _validate_gray(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray)
    if image.ndim != 2:
        raise ValueError("fragment realignment accepts one grayscale image")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.integer):
            raise ValueError("grayscale image must use an integer dtype")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _rail_measurements_for_axis(
    gray: np.ndarray, *, axis: Axis
) -> _RailMeasurements | None:
    """Collect the existing faint-rule channels on one canonical axis."""
    canonical = gray if axis == "y" else gray.T
    height, width = canonical.shape
    if min(height, width) < 32:
        return None

    background = cv2.GaussianBlur(canonical, (0, 0), 9)
    local_contrast = cv2.subtract(background, canonical)
    faint_ink = (local_contrast >= 3).astype(np.uint8) * 255
    # Keep this shorter than the smallest supported strip.  Otherwise an
    # oriented opening erases the rail segments at the first and last seams.
    rail_length = max(12, min(48, height // 16))
    vertical = cv2.morphologyEx(
        faint_ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, rail_length)),
    )
    # Remove only the columns that remain rail-like for essentially the entire
    # page.  They are form borders/distractors, never endpoint evidence.  Doing
    # this before component extraction also separates a distractor that touches
    # one edge of a three-pixel displaced rail.
    persistent_columns = (vertical > 0).sum(axis=0) >= int(round(height * 0.85))
    vertical[:, persistent_columns] = 0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    # Adjacent three-pixel rails can touch after a small displacement, yielding
    # a six-pixel component.  It is still a thin rail, not a broad text blob.
    max_rail_width = max(6, min(14, width // 120))
    min_rail_height = max(12, min(48, height // 64))
    rails: list[tuple[float, int, int]] = []
    for x, y, rail_width, rail_height, area in stats[1:component_count]:
        density = float(area) / max(float(rail_width * rail_height), 1.0)
        if (
            rail_width <= max_rail_width
            and rail_height >= min_rail_height
            and density >= 0.45
        ):
            rails.append(
                (float(x + (rail_width - 1) / 2.0), int(y), int(y + rail_height))
            )

    rail_margin = cv2.dilate(
        vertical, np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    # The rule detector intentionally accepts faint local contrast.  The
    # independent support gate must not: otherwise shifted rail ends become
    # their own apparent glyph evidence.  Dark ink and strong edges retain
    # real labels, stamps, and boxes without treating the faint lattice as
    # confirmation.
    dark_glyph = canonical < 180
    strong_edges = cv2.dilate(
        (cv2.Canny(canonical, 50, 120) > 0).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    non_rule = (dark_glyph | strong_edges) & ~rail_margin
    return _RailMeasurements(
        canonical=canonical,
        local_contrast=local_contrast,
        rail_mask=vertical,
        rails=tuple(rails),
        rail_margin=rail_margin,
        non_rule=non_rule,
        rail_length=rail_length,
        max_rail_width=max_rail_width,
        min_fragment=max(8, height // 80),
        endpoint_tolerance=max(4, min(10, height // 200)),
        center_tolerance=max(2, min(6, width // 400)),
    )


def build_content_masks(gray: np.ndarray) -> ContentMasks:
    """Build complementary masks without leaking long rules into evidence."""
    image = _validate_gray(gray)
    if image.size == 0:
        empty = np.zeros(image.shape, dtype=bool)
        return ContentMasks(empty, empty, empty)

    background = cv2.GaussianBlur(image, (0, 0), 9)
    normalized = cv2.divide(image, background, scale=255)
    local_ink = normalized < 225
    absolute_ink = image < 215
    ink = local_ink | absolute_ink
    ink_u8 = ink.astype(np.uint8) * 255
    # Rule extraction must be conservative.  If it starts from every gray
    # foreground pixel, the long edge of a pasted gray patch is mislabeled as
    # a form rule and the most useful alignment anchor disappears.
    rule_ink_u8 = (image < 80).astype(np.uint8) * 255

    height, width = image.shape
    h_len = max(12, min(100, width // 3))
    v_len = max(12, min(100, height // 3))
    horizontal = cv2.morphologyEx(
        rule_ink_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)),
    )
    vertical = cv2.morphologyEx(
        rule_ink_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)),
    )
    rule = cv2.bitwise_or(horizontal, vertical) > 0
    glyph = ink & ~cv2.dilate(rule.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # Gray pasted rectangles, stamps, and watermark fragments contribute their
    # edges.  Long form rules are explicitly removed from this channel too.
    edges = cv2.Canny(image, 18, 70) > 0
    structure = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    structure &= ~cv2.dilate(rule.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return ContentMasks(glyph, rule, structure)


def estimate_periods(mask: np.ndarray, *, axis: Axis = "y") -> tuple[int, ...]:
    """Return prominent non-zero autocorrelation periods for a binary mask."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("period estimation accepts a two-dimensional mask")
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    profile = values.sum(axis=1 if axis == "y" else 0).astype(np.float64)
    if profile.size < 8 or not np.any(profile):
        return ()
    centered = profile - profile.mean()
    correlation = np.correlate(centered, centered, mode="full")[profile.size - 1 :]
    limit = profile.size // 2
    if limit < 3 or correlation[0] <= 0:
        return ()
    peaks: list[tuple[float, int]] = []
    for offset in range(2, limit):
        score = float(correlation[offset])
        if score > 0 and score >= correlation[offset - 1] and score >= correlation[offset + 1]:
            peaks.append((score, offset))
    if not peaks:
        return ()
    strongest = max(score for score, _ in peaks)
    return tuple(
        offset
        for score, offset in sorted(peaks, key=lambda item: (-item[0], item[1]))
        if score >= strongest * 0.25
    )


def _validate_partition(
    transforms: tuple[FragmentTransform, ...], partition_axis: Axis, shape: tuple[int, int]
) -> None:
    if partition_axis not in ("x", "y"):
        raise ValueError("partition_axis must be 'x' or 'y'")
    limit = shape[1] if partition_axis == "x" else shape[0]
    if not transforms:
        raise ValueError("at least one fragment is required")
    expected = 0
    for fragment in transforms:
        start, end = fragment.interval
        if start != expected:
            raise ValueError("fragment intervals must form a disjoint contiguous partition")
        if end > limit:
            raise ValueError("fragment interval exceeds the selected axis")
        expected = end
    if expected != limit:
        raise ValueError("fragment intervals must cover the selected axis")


def apply_fragment_transforms(
    gray: np.ndarray,
    transforms: tuple[FragmentTransform, ...] | list[FragmentTransform],
    *,
    partition_axis: Axis,
) -> RepairCandidate:
    """Apply explicit integer translations without interpolation or wrapping."""
    source = _validate_gray(gray)
    fragments = tuple(transforms)
    _validate_partition(fragments, partition_axis, source.shape)
    height, width = source.shape
    reconstruction = np.full(source.shape, 255, dtype=np.uint8)
    destination_to_source = np.full(source.shape, -1, dtype=np.int64)
    source_to_destination = np.full(source.shape, -1, dtype=np.int64)
    write_count = np.zeros(source.shape, dtype=np.uint16)

    for fragment in fragments:
        start, end = fragment.interval
        if partition_axis == "y":
            sy0, sy1, sx0, sx1 = start, end, 0, width
        else:
            sy0, sy1, sx0, sx1 = 0, height, start, end
        dy, dx = fragment.inverse_dy, fragment.inverse_dx
        dy0, dy1, dx0, dx1 = sy0 + dy, sy1 + dy, sx0 + dx, sx1 + dx
        cy0, cy1 = max(0, dy0), min(height, dy1)
        cx0, cx1 = max(0, dx0), min(width, dx1)
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        ky0, kx0 = sy0 + (cy0 - dy0), sx0 + (cx0 - dx0)
        ky1, kx1 = ky0 + (cy1 - cy0), kx0 + (cx1 - cx0)
        src = source[ky0:ky1, kx0:kx1]
        dst_slice = (slice(cy0, cy1), slice(cx0, cx1))
        src_indices = (
            np.arange(ky0, ky1, dtype=np.int64)[:, None] * width
            + np.arange(kx0, kx1, dtype=np.int64)[None, :]
        )
        dst_indices = (
            np.arange(cy0, cy1, dtype=np.int64)[:, None] * width
            + np.arange(cx0, cx1, dtype=np.int64)[None, :]
        )
        free = destination_to_source[dst_slice] < 0
        dst_values = reconstruction[dst_slice]
        dst_values[free] = src[free]
        reconstruction[dst_slice] = dst_values
        dst_provenance = destination_to_source[dst_slice]
        dst_provenance[free] = src_indices[free]
        destination_to_source[dst_slice] = dst_provenance
        source_to_destination.flat[src_indices[free]] = dst_indices[free]
        write_count[dst_slice] += 1

    overlap = write_count > 1
    uncovered = write_count == 0
    cropped = source_to_destination < 0
    return RepairCandidate(
        rotation_k_ccw=0,
        rotation_human="0",
        region=(0, 0, width, height),
        partition_axis=partition_axis,
        partition_angle_degrees=0.0,
        fragments=fragments,
        source_to_destination_map=source_to_destination,
        destination_to_source_map=destination_to_source,
        overlap_mask=overlap,
        uncovered_mask=uncovered,
        cropped_source_mask=cropped,
        pixel_score_before=0.0,
        pixel_score_after=0.0,
        score_terms={},
        source_view=source,
        reconstruction=reconstruction,
    )


def _dominant_text_angle(gray: np.ndarray) -> float:
    """Estimate a conservative common angle without connecting step edges.

    A generous Hough line gap can join the stepped edges created by the very
    damage we are trying to repair and falsely call the staircase a page
    slant.  Short, gap-free edge segments instead recover the local direction.
    """
    source = _validate_gray(gray)
    image = cv2.Canny(source, 10, 50)
    height, width = source.shape
    lines = cv2.HoughLinesP(
        image,
        1,
        np.pi / 720,
        threshold=max(7, min(height, width) // 24),
        minLineLength=max(8, min(height, width) // 12),
        maxLineGap=1,
    )
    if lines is None:
        return 0.0
    bins: dict[float, float] = {}
    length_cap = max(12.0, min(height, width) / 4.0)
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        if abs(angle) <= 10:
            rounded = round(angle * 2.0) / 2.0
            length = min(float(np.hypot(x1 - x0, y1 - y0)), length_cap)
            bins[rounded] = bins.get(rounded, 0.0) + length
    if not bins:
        return 0.0
    angle, support = max(bins.items(), key=lambda item: (item[1], -abs(item[0])))
    zero_support = sum(value for key, value in bins.items() if abs(key) < 0.75)
    if abs(angle) < 0.75 or support < max(16.0, zero_support * 1.10):
        return 0.0
    return float(angle)


def _rectify_with_provenance(gray: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Deskew with nearest-neighbor sampling and retain original source ids."""
    height, width = gray.shape
    identity = np.arange(height * width, dtype=np.int32).reshape(height, width)
    if angle == 0.0:
        return gray, identity
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rectified = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    provenance = cv2.warpAffine(
        identity,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1,
    )
    return rectified, provenance.astype(np.int64)


def _compose_original_provenance(
    candidate: RepairCandidate, rectified_to_original: np.ndarray, original_shape: tuple[int, int]
) -> RepairCandidate:
    rect_src = candidate.destination_to_source_map
    destination_to_original = np.full(rect_src.shape, -1, dtype=np.int64)
    valid = rect_src >= 0
    destination_to_original[valid] = rectified_to_original.flat[rect_src[valid]]
    source_to_destination = np.full(original_shape, -1, dtype=np.int64)
    flat_dest = np.arange(destination_to_original.size, dtype=np.int64).reshape(
        destination_to_original.shape
    )
    visible = destination_to_original >= 0
    # First visible destination wins if nearest-neighbor deskew duplicated a
    # source pixel; the overlap remains visible in the destination mask.
    for src, dst in zip(destination_to_original[visible], flat_dest[visible]):
        if source_to_destination.flat[int(src)] < 0:
            source_to_destination.flat[int(src)] = int(dst)
    return replace(
        candidate,
        source_to_destination_map=source_to_destination,
        destination_to_source_map=destination_to_original,
        cropped_source_mask=source_to_destination < 0,
    )


def _alignment_evidence(gray: np.ndarray, masks: ContentMasks) -> np.ndarray:
    """Combine text, patch interiors, and patch edges while discounting rules."""
    del gray  # Kept in the signature to make channel construction explicit.
    return np.maximum(
        masks.glyph_mask.astype(np.float32),
        masks.structure_mask.astype(np.float32) * 0.45,
    )


def _shifted_soft_iou(
    left: np.ndarray, right: np.ndarray, offset: int
) -> tuple[float, float]:
    """Soft intersection-over-union after moving ``right`` by ``offset``."""
    size = left.size
    if abs(offset) >= size:
        return 0.0, 0.0
    if offset >= 0:
        left_view = left[offset:]
        right_view = right[: size - offset]
    else:
        left_view = left[: size + offset]
        right_view = right[-offset:]
    union = float(np.maximum(left_view, right_view).sum())
    if union <= 1e-8:
        return 0.0, 0.0
    intersection = float(np.minimum(left_view, right_view).sum())
    return intersection / union, intersection


def _shifted_profile(profile: np.ndarray, offset: int) -> np.ndarray:
    """No-wrap shift of a 1-D ink profile (frozen rule-anchor semantics)."""
    shifted = np.zeros_like(profile)
    if offset >= 0:
        if offset < profile.size:
            shifted[offset:] = profile[: profile.size - offset]
    elif -offset < profile.size:
        shifted[: profile.size + offset] = profile[-offset:]
    return shifted


def _profile_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return float(left @ right) / denominator


def _seam_hypotheses(
    gray: np.ndarray, evidence: np.ndarray, axis: Axis
) -> list[_SeamHypothesis]:
    """Find discontinuities whose two sides align under a non-zero shift."""
    partition_size = evidence.shape[1] if axis == "x" else evidence.shape[0]
    repair_size = evidence.shape[0] if axis == "x" else evidence.shape[1]
    min_width = max(4, partition_size // 24)
    span = 3
    max_offset = min(
        max(16, int(round(repair_size * 0.04))),
        max(8, int(repair_size * 0.20)),
        96,
    )
    min_offset = max(2, min(8, int(np.ceil(repair_size * 0.02))))
    gradient_axis = 0 if axis == "x" else 1
    edge = np.abs(np.diff(gray.astype(np.int16), axis=gradient_axis)) > 10
    if gradient_axis == 0:
        edge = np.pad(edge, ((0, 1), (0, 0)))
    else:
        edge = np.pad(edge, ((0, 0), (0, 1)))
    edge = edge.astype(np.float32)
    candidates: list[_SeamHypothesis] = []
    for boundary in range(min_width, partition_size - min_width + 1):
        if boundary < span or partition_size - boundary < span:
            continue
        if axis == "x":
            left = evidence[:, boundary - span : boundary].max(axis=1)
            right = evidence[:, boundary : boundary + span].max(axis=1)
            left_edge = edge[:, boundary - span : boundary].max(axis=1)
            right_edge = edge[:, boundary : boundary + span].max(axis=1)
        else:
            left = evidence[boundary - span : boundary, :].max(axis=0)
            right = evidence[boundary : boundary + span, :].max(axis=0)
            left_edge = edge[boundary - span : boundary, :].max(axis=0)
            right_edge = edge[boundary : boundary + span, :].max(axis=0)
        before, _ = _shifted_soft_iou(left, right, 0)
        edge_before, _ = _shifted_soft_iou(left_edge, right_edge, 0)
        best: tuple[float, float, float, float, int] | None = None
        for offset in range(-max_offset, max_offset + 1):
            if abs(offset) < min_offset:
                continue
            similarity, intersection = _shifted_soft_iou(left, right, offset)
            edge_similarity, edge_intersection = _shifted_soft_iou(
                left_edge, right_edge, offset
            )
            occupancy_gain = similarity - before
            edge_gain = edge_similarity - edge_before
            objective = (
                occupancy_gain
                + edge_gain
                - 0.35 * abs(offset) / max(float(repair_size), 1.0)
            )
            item = (
                objective,
                similarity,
                edge_similarity,
                intersection + edge_intersection,
                offset,
            )
            if best is None or item[:4] > best[:4]:
                best = item
        if best is None:
            continue
        objective, after, edge_after, support, offset = best
        gain = after - before
        edge_gain = edge_after - edge_before
        occupancy_support = float(np.maximum(left, right).sum()) / max(
            float(repair_size), 1.0
        )
        edge_support = float(np.maximum(left_edge, right_edge).sum()) / max(
            float(repair_size), 1.0
        )
        if (
            occupancy_support < 0.55
            or after < 0.82
            or gain < 0.06
            or edge_support < 0.02
            or edge_after < 0.35
            or edge_gain < 0.08
        ):
            continue
        candidates.append(
            _SeamHypothesis(
                boundary=boundary,
                relative_offset=offset,
                similarity_before=before,
                similarity_after=after,
                edge_similarity_before=edge_before,
                edge_similarity_after=edge_after,
                shared_support=support,
                gain=gain + edge_gain,
                objective=objective,
            )
        )

    # Non-maximum suppression makes nearby pixels one boundary hypothesis.
    selected: list[_SeamHypothesis] = []
    for item in sorted(
        candidates,
        key=lambda value: (
            -value.objective,
            value.boundary,
        ),
    ):
        if all(abs(item.boundary - prior.boundary) > 2 for prior in selected):
            selected.append(item)
        if len(selected) >= 8:
            break
    return selected


def _transforms_from_seams(
    seams: tuple[_SeamHypothesis, ...], partition_size: int, axis: Axis, anchor: int
) -> tuple[FragmentTransform, ...]:
    ordered = tuple(sorted(seams, key=lambda item: item.boundary))
    offsets = [0]
    for seam in ordered:
        offsets.append(offsets[-1] + seam.relative_offset)
    gauge = offsets[anchor]
    offsets = [value - gauge for value in offsets]
    edges = (0, *(item.boundary for item in ordered), partition_size)
    return tuple(
        FragmentTransform(
            (start, end),
            inverse_dx=offset if axis == "y" else 0,
            inverse_dy=offset if axis == "x" else 0,
        )
        for (start, end), offset in zip(zip(edges, edges[1:]), offsets)
    )


def _solve_axis(
    rectified: np.ndarray,
    evidence: np.ndarray,
    *,
    axis: Axis,
    max_fragments: int,
) -> list[RepairCandidate]:
    """Bounded piecewise-constant solver over candidate seam state changes."""
    seams = _seam_hypotheses(rectified, evidence, axis)
    if not seams:
        return []
    partition_size = rectified.shape[1] if axis == "x" else rectified.shape[0]
    candidates: list[RepairCandidate] = []
    # Selecting one through four state changes is the small, explicit dynamic
    # program for version 1.  The state path is constant between chosen seams.
    for seam_count in range(1, min(max_fragments - 1, len(seams)) + 1):
        for chosen in combinations(seams, seam_count):
            ordered = tuple(sorted(chosen, key=lambda item: item.boundary))
            edges = (0, *(item.boundary for item in ordered), partition_size)
            if min(end - start for start, end in zip(edges, edges[1:])) < max(
                4, partition_size // 24
            ):
                continue
            before = float(
                sum(
                    item.similarity_before + item.edge_similarity_before
                    for item in ordered
                )
            )
            after = float(
                sum(
                    item.similarity_after + item.edge_similarity_after
                    for item in ordered
                )
            )
            raw_gain = after - before
            support = float(sum(item.shared_support for item in ordered))
            transforms = _transforms_from_seams(ordered, partition_size, axis, anchor=0)
            candidate = apply_fragment_transforms(
                rectified, transforms, partition_axis=axis
            )
            loss_fraction = float(
                (candidate.overlap_mask.sum() + candidate.uncovered_mask.sum())
                / max(candidate.reconstruction.size, 1)
            )
            complexity = 0.015 * (len(transforms) - 1)
            loss_penalty = 0.35 * loss_fraction
            total_gain = raw_gain - complexity - loss_penalty
            if total_gain <= 0.035:
                continue
            candidates.append(
                replace(
                    candidate,
                    pixel_score_before=before,
                    pixel_score_after=after,
                    score_terms={
                        "boundary_similarity_before": before,
                        "boundary_similarity_after": after,
                        "boundary_similarity_gain": raw_gain,
                        "shared_support": support,
                        "complexity_penalty": complexity,
                        "loss_fraction": loss_fraction,
                        "loss_penalty": loss_penalty,
                        "total_gain": total_gain,
                        "fragment_count": float(len(transforms)),
                    },
                )
            )
    return candidates


def _comb_target(length: int, pitch: int, phase: int) -> np.ndarray:
    target = np.zeros(length, dtype=np.float32)
    half_width = max(2, int(round(pitch * 0.22)))
    for center in range(phase, length, pitch):
        target[max(0, center - half_width) : min(length, center + half_width + 1)] = 1.0
    return target


def _profile_shift_score(profile: np.ndarray, target: np.ndarray, offset: int) -> float:
    total = max(float(profile.sum()), 1.0)
    size = profile.size
    if offset >= 0:
        if offset >= size:
            return 0.0
        return float(profile[: size - offset] @ target[offset:]) / total
    if -offset >= size:
        return 0.0
    return float(profile[-offset:] @ target[: size + offset]) / total


def _viterbi_piecewise_offsets(
    scores: np.ndarray, states: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Choose a piecewise-constant offset path across narrow content bins."""
    bin_count, state_count = scores.shape
    zero_state = int(np.argmin(np.abs(states)))
    cost = np.full((bin_count, state_count), np.inf, dtype=np.float64)
    back = np.zeros((bin_count, state_count), dtype=np.int16)
    cost[0] = -scores[0] + 0.004 * np.abs(states)
    cost[0] += 0.06 * np.abs(states)
    cost[0, zero_state] -= 1.0
    transition = 0.018 * np.abs(states[:, None] - states[None, :])
    transition += 0.12 * (states[:, None] != states[None, :])
    for index in range(1, bin_count):
        alternatives = cost[index - 1][:, None] + transition
        back[index] = np.argmin(alternatives, axis=0)
        ink_weight = min(1.0, float(scores[index].max()) * 2.0)
        data_cost = -scores[index] * (1.5 + ink_weight)
        data_cost += 0.004 * np.abs(states)
        cost[index] = (
            alternatives[back[index], np.arange(state_count)] + data_cost
        )
    path = np.zeros(bin_count, dtype=np.int16)
    path[-1] = int(np.argmin(cost[-1]))
    for index in range(bin_count - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    chosen = states[path]
    informative = scores.max(axis=1) >= 0.15
    if not np.any(informative):
        return chosen, 0.0, 0.0
    before = float(scores[informative, zero_state].mean())
    after = float(scores[np.arange(bin_count), path][informative].mean())
    return chosen, before, after


def _refine_profile_offsets(
    evidence: np.ndarray,
    edges: tuple[int, ...],
    offsets: tuple[int, ...],
    max_offset: int,
) -> tuple[int, ...]:
    """Refine row-comb offsets using whole-fragment projection correlation."""
    profiles = [
        evidence[:, start:end].sum(axis=1).astype(np.float32)
        for start, end in zip(edges, edges[1:])
    ]
    anchor = int(np.argmax([profile.sum() for profile in profiles]))
    anchor_profile = profiles[anchor]
    refined: list[int] = []
    for profile, approximate in zip(profiles, offsets):
        ranked: list[tuple[float, int, int]] = []
        for offset in range(-max_offset, max_offset + 1):
            similarity, _ = _shifted_soft_iou(anchor_profile, profile, offset)
            ranked.append((similarity, -abs(offset - approximate), -abs(offset)))
        best = max(ranked)
        refined.append(-best[2] if best[2] else 0)
        # The second sort key selects the closest periodic alternative, but
        # recover the signed offset from the actual best-scoring entries.
        tied = [
            offset
            for offset in range(-max_offset, max_offset + 1)
            if abs(
                _shifted_soft_iou(anchor_profile, profile, offset)[0] - best[0]
            )
            <= 1e-6
        ]
        refined[-1] = min(
            tied, key=lambda value: (abs(value - approximate), abs(value), value)
        )
    gauge = refined[0]
    return tuple(value - gauge for value in refined)


def _comb_row_support(
    profile: np.ndarray, target: np.ndarray, offset: int, pitch: int
) -> int:
    """Count distinct periodic rows containing meaningful fragment evidence."""
    size = profile.size
    shifted = np.zeros_like(profile)
    if offset >= 0 and offset < size:
        shifted[offset:] = profile[: size - offset]
    elif offset < 0 and -offset < size:
        shifted[: size + offset] = profile[-offset:]
    band_energy = shifted * target
    if band_energy.sum() <= 0:
        return 0
    occupied = band_energy > max(0.5, float(band_energy.max()) * 0.08)
    merge = max(1, pitch // 3)
    occupied = cv2.dilate(
        occupied.astype(np.uint8).reshape(-1, 1),
        np.ones((merge, 1), dtype=np.uint8),
    ).ravel()
    count = 0
    active = False
    for value in occupied:
        if value and not active:
            count += 1
        active = bool(value)
    return count


def _pitch_paths(
    canonical_gray: np.ndarray,
    canonical_evidence: np.ndarray,
    max_fragments: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], dict[str, float]]]:
    """Find blank-seam repairs by aligning narrow bins to an estimated row comb."""
    repair_size, partition_size = canonical_gray.shape
    binary = canonical_evidence >= 0.5
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    textlike_components = 0
    textlike_area = 0
    region_area = float(max(canonical_gray.size, 1))
    for index in range(1, component_count):
        _, _, width, height, area = [int(value) for value in component_stats[index]]
        if (
            2 <= area <= region_area * 0.005
            and 1 <= width <= partition_size * 0.15
            and 2 <= height <= repair_size * 0.10
            and width / max(height, 1) < 5.0
        ):
            textlike_components += 1
            textlike_area += area
    minimum_components = 25 if canonical_gray.size >= 50_000 else 10
    textlike_fraction = textlike_area / max(float(binary.sum()), 1.0)
    if textlike_components < minimum_components or textlike_fraction < 0.25:
        return []
    periods = [
        value
        for value in estimate_periods(binary, axis="y")
        if 8 <= value <= repair_size // 3
    ]
    pitches: list[int] = []
    for period in periods:
        if any(abs(period - multiple * prior) <= 2 for prior in pitches for multiple in (1, 2, 3)):
            continue
        pitches.append(period)
        if len(pitches) >= 2:
            break
    paths: list[tuple[tuple[int, ...], tuple[int, ...], dict[str, float]]] = []
    projection = canonical_evidence.sum(axis=1)
    for pitch in pitches:
        bin_width = max(2, pitch // 4)
        max_offset = min(96, pitch * 3)
        states = np.arange(-max_offset, max_offset + 1, dtype=np.int16)
        # Multiple phases are retained because displaced strips can make the
        # most-populated phase different from the intact row phase.
        phase_step = 1 if pitch <= 36 else 2
        for phase in range(0, pitch, phase_step):
            target = _comb_target(repair_size, pitch, phase)
            if float(projection @ target) <= 0:
                continue
            profiles = [
                canonical_evidence[:, start : min(start + bin_width, partition_size)].sum(
                    axis=1
                )
                for start in range(0, partition_size, bin_width)
            ]
            scores = np.asarray(
                [
                    [_profile_shift_score(profile, target, int(offset)) for offset in states]
                    for profile in profiles
                ],
                dtype=np.float32,
            )
            chosen, before, after = _viterbi_piecewise_offsets(scores, states)
            if after - before < 0.08:
                continue
            runs: list[tuple[int, int, int]] = []
            run_start = 0
            for index in range(1, chosen.size + 1):
                if index == chosen.size or chosen[index] != chosen[run_start]:
                    runs.append(
                        (
                            run_start * bin_width,
                            min(index * bin_width, partition_size),
                            int(chosen[run_start]),
                        )
                    )
                    run_start = index
            # Blank leading/trailing bins inherit their nearest content run
            # instead of becoming meaningless one-bin fragments.
            while len(runs) > 1 and runs[0][1] - runs[0][0] <= bin_width:
                _, end, _ = runs.pop(0)
                start, next_end, offset = runs[0]
                runs[0] = (0, next_end, offset)
            while len(runs) > 1 and runs[-1][1] - runs[-1][0] <= bin_width:
                start, _, _ = runs.pop()
                prior_start, _, offset = runs[-1]
                runs[-1] = (prior_start, partition_size, offset)
            if not 2 <= len(runs) <= max_fragments:
                continue
            gauge = runs[0][2]
            offsets = tuple(offset - gauge for _, _, offset in runs)
            edges = (0, *(end for _, end, _ in runs[:-1]), partition_size)
            offsets = _refine_profile_offsets(
                canonical_evidence, edges, offsets, max_offset
            )
            if max(abs(value) for value in offsets) < 2:
                continue
            profiles = [
                canonical_evidence[:, start:end].sum(axis=1)
                for start, end in zip(edges, edges[1:])
            ]
            supports = [
                _comb_row_support(profile, target, offset, pitch)
                for profile, offset in zip(profiles, offsets)
            ]
            if min(supports) < 3:
                continue
            paths.append(
                (
                    edges,
                    offsets,
                    {
                        "row_pitch": float(pitch),
                        "row_phase": float(phase),
                        "row_comb_before": before,
                        "row_comb_after": after,
                        "row_comb_gain": after - before,
                        "minimum_row_support": float(min(supports)),
                        "fragment_count": float(len(runs)),
                    },
                )
            )
    return paths


def _solve_pitch_axis(
    gray: np.ndarray,
    evidence: np.ndarray,
    *,
    axis: Axis,
    max_fragments: int,
) -> list[RepairCandidate]:
    canonical_gray = gray if axis == "x" else gray.T
    canonical_evidence = evidence if axis == "x" else evidence.T
    candidates: list[RepairCandidate] = []
    for edges, offsets, terms in _pitch_paths(
        canonical_gray, canonical_evidence, max_fragments
    ):
        transforms = tuple(
            FragmentTransform(
                (start, end),
                inverse_dx=offset if axis == "y" else 0,
                inverse_dy=offset if axis == "x" else 0,
            )
            for (start, end), offset in zip(zip(edges, edges[1:]), offsets)
        )
        candidate = apply_fragment_transforms(gray, transforms, partition_axis=axis)
        loss_fraction = float(
            (candidate.overlap_mask.sum() + candidate.uncovered_mask.sum())
            / max(candidate.reconstruction.size, 1)
        )
        complexity = 0.015 * (len(transforms) - 1)
        loss_penalty = 0.35 * loss_fraction
        total_gain = terms["row_comb_gain"] - complexity - loss_penalty
        if total_gain <= 0.035:
            continue
        candidates.append(
            replace(
                candidate,
                pixel_score_before=terms["row_comb_before"],
                pixel_score_after=terms["row_comb_after"],
                score_terms={
                    **terms,
                    "method_pitch": 1.0,
                    "complexity_penalty": complexity,
                    "loss_fraction": loss_fraction,
                    "loss_penalty": loss_penalty,
                    "total_gain": total_gain,
                },
            )
        )
    return candidates


def _looks_like_residual_skew(candidate: RepairCandidate, angle: float) -> bool:
    """Reject a smooth offset ramp already explained by common page skew."""
    if abs(angle) < 0.75 or len(candidate.fragments) < 2:
        return False
    offsets = np.asarray(
        [
            fragment.inverse_dx
            if candidate.partition_axis == "y"
            else fragment.inverse_dy
            for fragment in candidate.fragments
        ],
        dtype=np.float64,
    )
    centers = np.asarray(
        [
            (fragment.interval[0] + fragment.interval[1]) / 2.0
            for fragment in candidate.fragments
        ],
        dtype=np.float64,
    )
    offset_range = float(np.ptp(offsets))
    partition_size = float(candidate.source_view.shape[0 if candidate.partition_axis == "y" else 1])
    explained_range = abs(np.tan(np.radians(angle))) * partition_size
    if offset_range > max(4.0, explained_range * 1.5 + 2.0):
        return False
    slope, intercept = np.polyfit(centers, offsets, 1)
    residual = float(np.max(np.abs(offsets - (slope * centers + intercept))))
    differences = np.diff(offsets)
    monotonic = bool(np.all(differences >= -1.0) or np.all(differences <= 1.0))
    return monotonic and residual <= max(2.0, offset_range * 0.25)


def generate_repair_candidates(
    gray: np.ndarray, *, max_fragments: int = 5, top_k: int = 8
) -> list[RepairCandidate]:
    """Return ranked, distinct, geometry-only candidates or abstain with ``[]``."""
    image = _validate_gray(gray)
    if not 2 <= max_fragments <= 12:
        raise ValueError("max_fragments must be between 2 and 12")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    # The legacy evidence solvers enumerate seam subsets.  Keep their search
    # bounded even though the public contract admits larger rule-anchor paths.
    legacy_max_fragments = min(max_fragments, 5)
    initial_masks = build_content_masks(image)
    initial_evidence = _alignment_evidence(image, initial_masks)
    angle = _dominant_text_angle(image)
    identity = np.arange(image.size, dtype=np.int64).reshape(image.shape)
    seam_raw = _solve_axis(
        image, initial_evidence, axis="y", max_fragments=legacy_max_fragments
    ) + _solve_axis(
        image, initial_evidence, axis="x", max_fragments=legacy_max_fragments
    )

    pitch_view, pitch_to_original = _rectify_with_provenance(image, angle)
    pitch_masks = build_content_masks(pitch_view)
    pitch_evidence = _alignment_evidence(pitch_view, pitch_masks)
    pitch_raw = _solve_pitch_axis(
        pitch_view, pitch_evidence, axis="y", max_fragments=legacy_max_fragments
    ) + _solve_pitch_axis(
        pitch_view, pitch_evidence, axis="x", max_fragments=legacy_max_fragments
    )
    pitch_raw = [
        candidate
        for candidate in pitch_raw
        if not _looks_like_residual_skew(candidate, angle)
    ]

    composed: list[RepairCandidate] = []
    for candidate, view_to_original, source_view in (
        *((candidate, identity, image) for candidate in seam_raw),
        *((candidate, pitch_to_original, pitch_view) for candidate in pitch_raw),
    ):
        candidate = _compose_original_provenance(
            candidate, view_to_original, image.shape
        )
        candidate = replace(
            candidate,
            partition_angle_degrees=angle,
            source_view=source_view,
            region=(0, 0, source_view.shape[1], source_view.shape[0]),
        )
        composed.append(candidate)

    def signature(candidate: RepairCandidate) -> tuple:
        return (
            candidate.partition_axis,
            tuple(
                (
                    item.interval,
                    item.inverse_dx,
                    item.inverse_dy,
                )
                for item in candidate.fragments
            ),
        )

    distinct: dict[tuple, RepairCandidate] = {}
    for candidate in composed:
        key = signature(candidate)
        if key not in distinct or candidate.score_terms["total_gain"] > distinct[key].score_terms[
            "total_gain"
        ]:
            distinct[key] = candidate
    ordered = sorted(
        distinct.values(),
        key=lambda item: (
            -item.score_terms["total_gain"],
            item.partition_axis,
            tuple((f.interval, f.inverse_dx, f.inverse_dy) for f in item.fragments),
        ),
    )
    return ordered[:top_k]


def _two_rail_windows(
    measurements: _RailMeasurements, *, max_windows: int = 8
) -> list[_TwoRailWindow]:
    """Seed bounded windows with exactly two ending and starting rail traces."""
    rails = measurements.rails
    if len(rails) < 6:
        return []
    height, width = measurements.canonical.shape
    tolerance = measurements.endpoint_tolerance
    starts = sorted((start, index) for index, (_, start, _) in enumerate(rails))
    ends = sorted((end, index) for index, (_, _, end) in enumerate(rails))
    boundary_seeds: set[int] = set()
    start_index = 0
    for end, _ in ends:
        while (
            start_index < len(starts)
            and starts[start_index][0] < end - tolerance
        ):
            start_index += 1
        index = start_index
        while index < len(starts) and starts[index][0] <= end + tolerance:
            boundary_seeds.add(int(round((end + starts[index][0]) / 2.0)))
            index += 1

    windows: list[_TwoRailWindow] = []

    def append_window(
        boundary: int, centers: list[float], *, angle_hint: float
    ) -> None:
        padding = max(
            2 * measurements.rail_length,
            4 * measurements.center_tolerance,
        )
        x0 = max(0, int(np.floor(min(centers))) - padding)
        x1 = min(width, int(np.ceil(max(centers))) + padding + 1)
        y0 = max(0, boundary - 4 * measurements.rail_length)
        y1 = min(height, boundary + 4 * measurements.rail_length)
        if (
            x1 - x0 < 2 * measurements.rail_length
            or y1 - y0 < 2 * measurements.rail_length
        ):
            return
        candidate = _TwoRailWindow(
            boundary,
            x0,
            y0,
            x1,
            y1,
            angle_hint,
        )
        if all(
            (
                prior.boundary,
                prior.x0,
                prior.x1,
                prior.angle_hint,
            )
            != (
                candidate.boundary,
                candidate.x0,
                candidate.x1,
                candidate.angle_hint,
            )
            for prior in windows
        ):
            windows.append(candidate)

    for boundary in sorted(boundary_seeds):
        ending = tuple(
            index for end, index in ends if abs(end - boundary) <= tolerance
        )
        starting = tuple(
            index for start, index in starts if abs(start - boundary) <= tolerance
        )
        if len(ending) != 2 or len(starting) != 2:
            continue
        centers = [
            *(rails[index][0] for index in ending),
            *(rails[index][0] for index in starting),
        ]
        append_window(boundary, centers, angle_hint=0.0)

    # A genuinely deskewable event is slanted before local rectification: the
    # two rail pairs meet at different rows, but each ending trace still meets
    # one starting trace with a common horizontal displacement.  Fit that
    # two-point endpoint line within the same +/-4 degree hypothesis bound.
    trace_rails = tuple(
        sorted(
            (
                rail
                for rail in rails
                if rail[2] - rail[1] >= 2 * measurements.rail_length
            ),
            key=lambda rail: (
                -(rail[2] - rail[1]),
                rail[1],
                rail[0],
            ),
        )[:64]
    )
    # Form endpoint correspondences before considering pairs of rails.  The
    # retained list is deliberately bounded here, so adversarial pages with
    # many short vertical components cannot turn the pair search into O(R^4).
    endpoint_correspondences = sorted(
        (
            (
                abs(ending[2] - starting[1]),
                -min(
                    ending[2] - ending[1],
                    starting[2] - starting[1],
                ),
                ending_index,
                starting_index,
            )
            for ending_index, ending in enumerate(trace_rails)
            for starting_index, starting in enumerate(trace_rails)
            if (
                abs(ending[2] - starting[1]) <= tolerance
                and 8
                <= abs(starting[0] - ending[0])
                <= width // 3
            )
        ),
        key=lambda item: item,
    )[:64]
    for left_match, right_match in combinations(
        endpoint_correspondences, 2
    ):
        _, _, left_ending_index, left_starting_index = left_match
        _, _, right_ending_index, right_starting_index = right_match
        if (
            left_ending_index == right_ending_index
            or left_starting_index == right_starting_index
        ):
            continue
        matched_pairs = sorted(
            (
                (
                    trace_rails[left_ending_index],
                    trace_rails[left_starting_index],
                ),
                (
                    trace_rails[right_ending_index],
                    trace_rails[right_starting_index],
                ),
            ),
            key=lambda pair: 0.5 * (pair[0][0] + pair[1][0]),
        )
        ending = [pair[0] for pair in matched_pairs]
        starting = [pair[1] for pair in matched_pairs]
        event_rows = [
            0.5 * (ending[index][2] + starting[index][1])
            for index in range(2)
        ]
        observed_offsets = [
            starting[index][0] - ending[index][0]
            for index in range(2)
        ]
        if (
            max(observed_offsets) - min(observed_offsets)
            > measurements.center_tolerance
        ):
            continue
        event_centers = [
            0.5 * (ending[index][0] + starting[index][0])
            for index in range(2)
        ]
        event_span = event_centers[1] - event_centers[0]
        if event_span <= 0:
            continue
        angle_hint = float(
            np.degrees(
                np.arctan2(
                    event_rows[1] - event_rows[0],
                    event_span,
                )
            )
        )
        if not 0.125 <= abs(angle_hint) <= 4.0:
            continue
        intercept = event_rows[0] - np.tan(
            np.radians(angle_hint)
        ) * event_centers[0]
        ending_on_event = [
            rail
            for rail in trace_rails
            if abs(
                rail[2]
                - (
                    intercept
                    + np.tan(np.radians(angle_hint)) * rail[0]
                )
            )
            <= tolerance
        ]
        starting_on_event = [
            rail
            for rail in trace_rails
            if abs(
                rail[1]
                - (
                    intercept
                    + np.tan(np.radians(angle_hint)) * rail[0]
                )
            )
            <= tolerance
        ]
        if len(ending_on_event) != 2 or len(starting_on_event) != 2:
            continue
        boundary = int(round(float(np.mean(event_rows))))
        append_window(
            boundary,
            [
                *(rail[0] for rail in ending),
                *(rail[0] for rail in starting),
            ],
            angle_hint=round(angle_hint * 4.0) / 4.0,
        )

    return sorted(
        windows,
        key=lambda item: (
            item.boundary,
            abs(item.angle_hint),
            item.x0,
            item.x1,
        ),
    )[:max_windows]


def _near_axis_hough_modes(gray: np.ndarray, *, rail_length: int) -> list[float]:
    """Return the two strongest distinct near-vertical deskew modes."""
    height, width = gray.shape
    edges = cv2.Canny(gray, 10, 50)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=max(7, min(height, width) // 24),
        minLineLength=max(8, rail_length),
        maxLineGap=1,
    )
    if lines is None:
        return []
    bins: dict[float, float] = {}
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        deviation = angle - 90.0 if angle >= 0.0 else angle + 90.0
        if abs(deviation) > 4.0:
            continue
        mode = round(deviation * 4.0) / 4.0
        length = float(np.hypot(x1 - x0, y1 - y0))
        bins[mode] = bins.get(mode, 0.0) + length
    ordered = sorted(
        bins.items(),
        key=lambda item: (-item[1], abs(item[0]), item[0]),
    )
    modes: list[float] = []
    for mode, _ in ordered:
        if abs(mode) < 0.125:
            continue
        if all(abs(mode - prior) > 0.25 for prior in modes):
            modes.append(float(mode))
        if len(modes) >= 2:
            break
    return modes


def _shift_profile(profile: np.ndarray, offset: int) -> np.ndarray:
    shifted = np.zeros_like(profile)
    if offset >= 0:
        if offset < profile.size:
            shifted[offset:] = profile[: profile.size - offset]
    elif -offset < profile.size:
        shifted[: profile.size + offset] = profile[-offset:]
    return shifted


def _profile_cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return float(left @ right) / denominator


def _map_local_rectified_boundary_to_full(
    seed: _TwoRailWindow,
    *,
    local_boundary: int,
    angle: float,
    canonical_shape: tuple[int, int],
) -> int:
    """Map a locally rectified cut to the same full-canonical deskew frame."""
    local_height = seed.y1 - seed.y0
    local_width = seed.x1 - seed.x0
    local_matrix = cv2.getRotationMatrix2D(
        (local_width / 2.0, local_height / 2.0), angle, 1.0
    )
    local_inverse = cv2.invertAffineTransform(local_matrix)
    local_source = local_inverse @ np.asarray(
        [local_width / 2.0, float(local_boundary), 1.0],
        dtype=np.float64,
    )
    canonical_source = np.asarray(
        [
            local_source[0] + seed.x0,
            local_source[1] + seed.y0,
            1.0,
        ],
        dtype=np.float64,
    )
    height, width = canonical_shape
    full_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), angle, 1.0
    )
    full_destination = full_matrix @ canonical_source
    return int(round(float(full_destination[1])))


def _evaluate_two_rail_boundary(
    canonical: np.ndarray,
    seed: _TwoRailWindow,
    *,
    angle: float,
    enforce_score_gates: bool = True,
) -> _TwoRailBoundaryHypothesis | None:
    """Evaluate one angle using only the seed's bounded source pixels."""
    window = canonical[seed.y0 : seed.y1, seed.x0 : seed.x1]
    rectified, rectified_to_window = _rectify_with_provenance(window, angle)
    measurements = _rail_measurements_for_axis(rectified, axis="y")
    if measurements is None:
        return None
    expected_boundary = seed.boundary - seed.y0
    # The shared collector deliberately admits short vertical glyph strokes for
    # the strong detector.  Within this bounded window, a rail trace must span
    # at least two opening lengths; this keeps those glyph strokes from
    # inflating the exact two-ending/two-starting count.
    trace_rails = tuple(
        rail
        for rail in measurements.rails
        if rail[2] - rail[1] >= 2 * measurements.rail_length
    )
    starts = sorted(
        (start, index) for index, (_, start, _) in enumerate(trace_rails)
    )
    ends = sorted(
        (end, index) for index, (_, _, end) in enumerate(trace_rails)
    )
    local_boundaries = {
        int(round((end + start) / 2.0))
        for end, _ in ends
        for start, _ in starts
        if abs(end - start) <= measurements.endpoint_tolerance
    }
    nearby = [
        boundary
        for boundary in local_boundaries
        if abs(boundary - expected_boundary)
        <= measurements.endpoint_tolerance + 2
    ]
    if not nearby:
        return None
    event_boundary = min(
        nearby,
        key=lambda boundary: (
            abs(boundary - expected_boundary),
            boundary,
        ),
    )

    tolerance = measurements.endpoint_tolerance
    ending = sorted(
        trace_rails[index][0]
        for end, index in ends
        if abs(end - event_boundary) <= tolerance
    )
    starting = sorted(
        trace_rails[index][0]
        for start, index in starts
        if abs(start - event_boundary) <= tolerance
    )
    if len(ending) != 2 or len(starting) != 2:
        return None
    observed_offsets = np.asarray(starting) - np.asarray(ending)
    observed_offset = int(
        np.copysign(
            np.floor(abs(float(observed_offsets.mean())) + 0.5),
            float(observed_offsets.mean()),
        )
    )
    pair_residual = float(
        np.max(np.abs(observed_offsets - float(observed_offset)))
    )
    matched_span = float(ending[1] - ending[0])
    matched_fraction = 1.0
    inverse_offset = -observed_offset
    max_offset = min(320, rectified.shape[1] // 3)
    if not (
        matched_span >= 0.35 * rectified.shape[1]
        and pair_residual <= measurements.center_tolerance
        and 8 <= abs(inverse_offset) <= max_offset
    ):
        return None

    full_ink = measurements.local_contrast.astype(np.float32) / 255.0
    full_ink[measurements.local_contrast < 3] = 0.0
    non_rule_ink = full_ink.copy()
    non_rule_ink[measurements.rail_margin] = 0.0
    non_rule_depth = max(16, min(32, 2 * measurements.rail_length))

    def continuity(
        boundary: int, offset: int, *, support_bins: bool = False
    ) -> tuple[float, float, float, float, float, int, int]:
        if not (
            non_rule_depth <= boundary <= rectified.shape[0] - non_rule_depth
        ):
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
        literal_left = full_ink[boundary - 1]
        literal_right = full_ink[boundary]
        non_rule_left = non_rule_ink[
            boundary - non_rule_depth : boundary
        ].mean(axis=0)
        non_rule_right = non_rule_ink[
            boundary : boundary + non_rule_depth
        ].mean(axis=0)
        literal_before = _profile_cosine_score(literal_left, literal_right)
        literal_after = _profile_cosine_score(
            literal_left, _shift_profile(literal_right, offset)
        )
        non_rule_before = _profile_cosine_score(
            non_rule_left, non_rule_right
        )
        shifted_non_rule = _shift_profile(non_rule_right, offset)
        non_rule_after = _profile_cosine_score(
            non_rule_left, shifted_non_rule
        )
        before = 0.5 * (literal_before + non_rule_before)
        after = 0.5 * (literal_after + non_rule_after)

        if not support_bins:
            return (
                before,
                after,
                literal_after - literal_before,
                non_rule_after - non_rule_before,
                after - before,
                0,
                0,
            )
        bin_width = max(
            2 * measurements.rail_length,
            rectified.shape[1] // 32,
        )
        support_bin_indexes: list[int] = []
        for index, start in enumerate(
            range(0, rectified.shape[1], bin_width)
        ):
            end = min(rectified.shape[1], start + bin_width)
            before_bin = _profile_cosine_score(
                non_rule_left[start:end],
                non_rule_right[start:end],
            )
            after_bin = _profile_cosine_score(
                non_rule_left[start:end],
                shifted_non_rule[start:end],
            )
            if after_bin - before_bin >= 0.03:
                support_bin_indexes.append(index)
        support_span = (
            max(support_bin_indexes) - min(support_bin_indexes)
            if len(support_bin_indexes) >= 2
            else 0
        )
        return (
            before,
            after,
            literal_after - literal_before,
            non_rule_after - non_rule_before,
            after - before,
            len(support_bin_indexes),
            support_span,
        )

    boundary_options = range(
        max(non_rule_depth, event_boundary - 3),
        min(rectified.shape[0] - non_rule_depth, event_boundary + 3) + 1,
    )
    if enforce_score_gates:
        allowed_offsets = [
            offset
            for offset in range(-max_offset, max_offset + 1)
            if abs(offset) >= 8
        ]
    else:
        allowed_offsets = list(
            range(
                inverse_offset - measurements.center_tolerance,
                inverse_offset + measurements.center_tolerance + 1,
            )
        )
    scored: list[
        tuple[
            float,
            int,
            int,
            tuple[float, float, float, float, float, int, int],
        ]
    ] = []
    for boundary in boundary_options:
        for offset in allowed_offsets:
            terms = continuity(boundary, offset)
            scored.append((terms[1], boundary, offset, terms))
    local_options = [
        item
        for item in scored
        if abs(item[2] - inverse_offset) <= measurements.center_tolerance
    ]
    if not local_options:
        return None
    best = max(
        local_options,
        key=lambda item: (
            item[0],
            -abs(item[2] - inverse_offset),
            -abs(item[1] - event_boundary),
        ),
    )
    score_after, boundary, inverse_offset, _ = best
    terms = continuity(
        boundary,
        inverse_offset,
        support_bins=enforce_score_gates,
    )
    (
        score_before,
        _,
        literal_gain,
        non_rule_gain,
        continuity_gain,
        support_bin_count,
        support_bin_span,
    ) = terms
    runner_up = (
        max(
            (
                item[0]
                for item in scored
                if abs(item[2] - inverse_offset)
                > measurements.center_tolerance
            ),
            default=0.0,
        )
        if enforce_score_gates
        else 0.0
    )
    offset_margin = score_after - runner_up
    support = float(
        measurements.non_rule[
            max(0, boundary - 2 * measurements.rail_length) : min(
                rectified.shape[0],
                boundary + 2 * measurements.rail_length,
            )
        ].sum()
    )
    if enforce_score_gates and (
        continuity_gain < 0.05
        or non_rule_gain < 0.05
        or offset_margin < 0.03
        or support < 12.0
        or support_bin_count < 2
        or support_bin_span < 2
    ):
        return None
    local_transforms = (
        FragmentTransform((0, boundary), inverse_dx=0),
        FragmentTransform(
            (boundary, rectified.shape[0]),
            inverse_dx=inverse_offset,
        ),
    )
    local_candidate = apply_fragment_transforms(
        rectified, local_transforms, partition_axis="y"
    )
    local_candidate = _compose_original_provenance(
        local_candidate,
        rectified_to_window,
        window.shape,
    )
    local_window_loss_fraction = float(
        (
            local_candidate.overlap_mask.sum()
            + local_candidate.uncovered_mask.sum()
        )
        / max(local_candidate.reconstruction.size, 1)
    )
    return _TwoRailBoundaryHypothesis(
        boundary=_map_local_rectified_boundary_to_full(
            seed,
            local_boundary=boundary,
            angle=angle,
            canonical_shape=canonical.shape,
        ),
        inverse_offset=inverse_offset,
        angle=angle,
        angle_margin=0.0,
        continuity_before=score_before,
        continuity_after=score_after,
        continuity_gain=continuity_gain,
        literal_gain=literal_gain,
        non_rule_gain=non_rule_gain,
        offset_margin=offset_margin,
        support_bin_count=support_bin_count,
        support_bin_span=support_bin_span,
        matched_span=matched_span,
        matched_span_threshold=0.35 * rectified.shape[1],
        matched_fraction=matched_fraction,
        pair_residual=pair_residual,
        maximum_inverse_offset=max_offset,
        local_window_loss_fraction=local_window_loss_fraction,
        boundary_stability=0.0,
        offset_stability=0.0,
        angle_competitor_count=0,
        angle_rejected_competitor_count=0,
        x0=seed.x0,
        x1=seed.x1,
        y0=seed.y0,
        y1=seed.y1,
    )


def _stable_two_rail_hypotheses(
    measurements: _RailMeasurements, seed: _TwoRailWindow
) -> list[_TwoRailBoundaryHypothesis]:
    """Retain modes stable to the required +/-0.25 degree perturbation."""
    window = measurements.canonical[seed.y0 : seed.y1, seed.x0 : seed.x1]
    observed_modes = _near_axis_hough_modes(
        window, rail_length=measurements.rail_length
    )
    # A two-endpoint rail fit is the bounded fallback when a short local rail
    # does not survive Hough's minimum-length vote.  It still comes directly
    # from the same detected rail pixels.  Zero plus at most two tolerance-
    # distinct non-zero modes are ever evaluated.
    if abs(seed.angle_hint) >= 0.125:
        observed_modes.insert(0, seed.angle_hint)
    distinct_modes: list[float] = []
    for mode in observed_modes:
        if abs(mode) < 0.125:
            continue
        if all(abs(mode - prior) > 0.25 for prior in distinct_modes):
            distinct_modes.append(float(mode))
        if len(distinct_modes) >= 2:
            break
    base_angles = [0.0, *distinct_modes]
    evaluated = {
        angle: _evaluate_two_rail_boundary(
            measurements.canonical,
            seed,
            angle=angle,
        )
        for angle in base_angles
    }
    comparison_evaluated = {
        angle: _evaluate_two_rail_boundary(
            measurements.canonical,
            seed,
            angle=angle,
            enforce_score_gates=False,
        )
        for angle in base_angles
    }
    stable: list[_TwoRailBoundaryHypothesis] = []
    for angle in base_angles:
        center = evaluated.get(angle)
        lower = _evaluate_two_rail_boundary(
            measurements.canonical,
            seed,
            angle=round(angle - 0.25, 2),
            enforce_score_gates=False,
        )
        upper = _evaluate_two_rail_boundary(
            measurements.canonical,
            seed,
            angle=round(angle + 0.25, 2),
            enforce_score_gates=False,
        )
        if center is None or lower is None or upper is None:
            continue
        boundary_stability = float(
            max(
                abs(lower.boundary - center.boundary),
                abs(upper.boundary - center.boundary),
            )
        )
        offset_stability = float(
            max(
                abs(lower.inverse_offset - center.inverse_offset),
                abs(upper.inverse_offset - center.inverse_offset),
            )
        )
        if (
            boundary_stability > 2.0
            or offset_stability > measurements.center_tolerance
        ):
            continue
        distinct_alternatives = [
            comparison_evaluated[other_angle]
            for other_angle in base_angles
            if other_angle != angle
        ]
        missing_alternative_angles = [
            other_angle
            for other_angle in base_angles
            if other_angle != angle
            and comparison_evaluated[other_angle] is None
        ]
        # A non-zero mode may uniquely explain an event that zero degrees
        # cannot even form.  Any missing retained non-zero competitor leaves
        # uniqueness unresolved and therefore forces abstention.
        if any(
            other_angle != 0.0 or angle == 0.0
            for other_angle in missing_alternative_angles
        ):
            continue
        competitor_scores = [
            candidate.continuity_after
            for candidate in distinct_alternatives
            if candidate is not None
        ]
        angle_margin = (
            center.continuity_after - max(competitor_scores)
            if competitor_scores
            else center.continuity_after
        )
        if angle_margin < 0.03:
            continue
        stable.append(
            replace(
                center,
                angle_margin=angle_margin,
                boundary_stability=boundary_stability,
                offset_stability=offset_stability,
                angle_competitor_count=len(competitor_scores),
                angle_rejected_competitor_count=len(
                    missing_alternative_angles
                ),
            )
        )
    return stable


def _map_full_rectified_boundary_to_local(
    boundary: int,
    *,
    angle: float,
    canonical_shape: tuple[int, int],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> int:
    """Express one full-canonical rectified cut in a local rectified frame."""
    height, width = canonical_shape
    full_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), angle, 1.0
    )
    canonical_source = cv2.invertAffineTransform(full_matrix) @ np.asarray(
        [width / 2.0, float(boundary), 1.0],
        dtype=np.float64,
    )
    local_width = x1 - x0
    local_height = y1 - y0
    local_matrix = cv2.getRotationMatrix2D(
        (local_width / 2.0, local_height / 2.0), angle, 1.0
    )
    local_destination = local_matrix @ np.asarray(
        [
            canonical_source[0] - x0,
            canonical_source[1] - y0,
            1.0,
        ],
        dtype=np.float64,
    )
    return int(round(float(local_destination[1])))


def _canonical_index_values_to_original(
    values: np.ndarray,
    *,
    canonical_width: int,
    original_width: int,
) -> np.ndarray:
    """Convert flattened transposed-canonical indices to original indices."""
    converted = np.full(values.shape, -1, dtype=np.int64)
    valid = values >= 0
    canonical_indices = values[valid]
    canonical_y = canonical_indices // canonical_width
    canonical_x = canonical_indices % canonical_width
    converted[valid] = canonical_x * original_width + canonical_y
    return converted


def _bounded_local_two_rail_repair(
    gray: np.ndarray,
    *,
    axis: Axis,
    angle: float,
    upper_boundary: int,
    lower_boundary: int,
    middle_inverse_offset: int,
    y0: int,
    y1: int,
) -> tuple[RepairCandidate, int, int, float, float] | None:
    """Rectify, repair, and map back one bounded canonical horizontal band."""
    canonical = gray if axis == "y" else gray.T
    canonical = np.ascontiguousarray(canonical)
    height, width = canonical.shape
    # The repair band spans the page in the displacement direction so its
    # provenance is complete, while remaining bounded to the union of the two
    # independently evaluated endpoint windows in the partition direction.
    x0, x1 = 0, width
    y0 = max(0, y0)
    y1 = min(height, y1)
    if y1 <= y0:
        return None
    window = canonical[y0:y1, x0:x1]
    rectified, rectified_to_window = _rectify_with_provenance(
        window, angle
    )
    local_upper = _map_full_rectified_boundary_to_local(
        upper_boundary,
        angle=angle,
        canonical_shape=canonical.shape,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )
    local_lower = _map_full_rectified_boundary_to_local(
        lower_boundary,
        angle=angle,
        canonical_shape=canonical.shape,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )
    if not 0 < local_upper < local_lower < rectified.shape[0]:
        return None

    local_transforms = (
        FragmentTransform((0, local_upper), inverse_dx=0),
        FragmentTransform(
            (local_upper, local_lower),
            inverse_dx=middle_inverse_offset,
        ),
        FragmentTransform(
            (local_lower, rectified.shape[0]),
            inverse_dx=0,
        ),
    )
    repaired = apply_fragment_transforms(
        rectified, local_transforms, partition_axis="y"
    )

    rectified_band = np.zeros(rectified.shape, dtype=np.uint8)
    rectified_band[local_upper:local_lower, :] = 255
    inverse_matrix = cv2.getRotationMatrix2D(
        (rectified.shape[1] / 2.0, rectified.shape[0] / 2.0),
        -angle,
        1.0,
    )
    local_band = (
        cv2.warpAffine(
            rectified_band,
            inverse_matrix,
            (rectified.shape[1], rectified.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    _, local_destination_to_rectified = _rectify_with_provenance(
        np.zeros(rectified.shape, dtype=np.uint8),
        -angle,
    )

    local_size = window.size
    local_identity = np.arange(local_size, dtype=np.int64).reshape(
        window.shape
    )
    local_destination_to_source = local_identity.copy()
    local_destination_to_source[local_band] = -1
    destination_ids = np.flatnonzero(
        local_band.ravel()
        & (local_destination_to_rectified.ravel() >= 0)
    )
    rectified_destination_ids = local_destination_to_rectified.ravel()[
        destination_ids
    ]
    rectified_source_ids = repaired.destination_to_source_map.ravel()[
        rectified_destination_ids
    ]
    retained = rectified_source_ids >= 0
    destination_ids = destination_ids[retained]
    rectified_source_ids = rectified_source_ids[retained]
    window_source_ids = rectified_to_window.ravel()[rectified_source_ids]
    retained = window_source_ids >= 0
    destination_ids = destination_ids[retained]
    window_source_ids = window_source_ids[retained]
    # Do not borrow a source pixel from outside the inverse-deskewed band:
    # those source pixels must retain their unchanged identity destinations.
    retained = local_band.ravel()[window_source_ids]
    destination_ids = destination_ids[retained]
    window_source_ids = window_source_ids[retained]
    local_destination_to_source.ravel()[destination_ids] = window_source_ids

    local_source_to_destination = local_identity.copy()
    local_source_to_destination[local_band] = -1
    if window_source_ids.size:
        _, first = np.unique(window_source_ids, return_index=True)
        local_source_to_destination.ravel()[window_source_ids[first]] = (
            destination_ids[first]
        )

    local_reconstruction = window.copy()
    local_reconstruction[local_band] = 255
    local_reconstruction.ravel()[destination_ids] = window.ravel()[
        window_source_ids
    ]
    local_overlap = (
        cv2.warpAffine(
            repaired.overlap_mask.astype(np.uint8) * 255,
            inverse_matrix,
            (rectified.shape[1], rectified.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    ) & local_band
    local_uncovered = local_band & (local_destination_to_source < 0)

    full_identity = np.arange(
        canonical.size, dtype=np.int64
    ).reshape(canonical.shape)
    destination_to_source = full_identity.copy()
    source_to_destination = full_identity.copy()

    def local_values_to_canonical(values: np.ndarray) -> np.ndarray:
        converted = np.full(values.shape, -1, dtype=np.int64)
        valid = values >= 0
        local_values = values[valid]
        local_y = local_values // window.shape[1]
        local_x = local_values % window.shape[1]
        converted[valid] = (
            (local_y + y0) * width + local_x + x0
        )
        return converted

    destination_to_source[y0:y1, x0:x1] = (
        local_values_to_canonical(local_destination_to_source)
    )
    source_to_destination[y0:y1, x0:x1] = (
        local_values_to_canonical(local_source_to_destination)
    )
    reconstruction = canonical.copy()
    reconstruction[y0:y1, x0:x1] = local_reconstruction
    overlap = np.zeros(canonical.shape, dtype=bool)
    overlap[y0:y1, x0:x1] = local_overlap
    uncovered = np.zeros(canonical.shape, dtype=bool)
    uncovered[y0:y1, x0:x1] = local_uncovered

    canonical_fragments = (
        FragmentTransform((0, upper_boundary), inverse_dx=0),
        FragmentTransform(
            (upper_boundary, lower_boundary),
            inverse_dx=middle_inverse_offset,
        ),
        FragmentTransform((lower_boundary, height), inverse_dx=0),
    )
    candidate = RepairCandidate(
        rotation_k_ccw=0,
        rotation_human="0",
        region=(0, 0, gray.shape[1], gray.shape[0]),
        partition_axis=axis,
        partition_angle_degrees=angle,
        fragments=canonical_fragments,
        source_to_destination_map=source_to_destination,
        destination_to_source_map=destination_to_source,
        overlap_mask=overlap,
        uncovered_mask=uncovered,
        cropped_source_mask=source_to_destination < 0,
        pixel_score_before=0.0,
        pixel_score_after=0.0,
        score_terms={},
        source_view=gray,
        reconstruction=reconstruction,
    )
    if axis == "x":
        original_fragments = tuple(
            FragmentTransform(
                fragment.interval,
                inverse_dy=fragment.inverse_dx,
            )
            for fragment in canonical_fragments
        )
        source_to_destination = _canonical_index_values_to_original(
            source_to_destination,
            canonical_width=width,
            original_width=gray.shape[1],
        ).T
        destination_to_source = _canonical_index_values_to_original(
            destination_to_source,
            canonical_width=width,
            original_width=gray.shape[1],
        ).T
        candidate = replace(
            candidate,
            fragments=original_fragments,
            source_to_destination_map=source_to_destination,
            destination_to_source_map=destination_to_source,
            overlap_mask=overlap.T,
            uncovered_mask=uncovered.T,
            cropped_source_mask=source_to_destination < 0,
            reconstruction=reconstruction.T,
        )

    final_loss = candidate.overlap_mask | candidate.uncovered_mask
    page_loss_fraction = float(
        final_loss.sum() / max(final_loss.size, 1)
    )
    local_loss_fraction = float(
        (local_overlap | local_uncovered).sum()
        / max(window.size, 1)
    )
    return (
        candidate,
        local_upper,
        local_lower,
        page_loss_fraction,
        local_loss_fraction,
    )


def _bounded_two_rail_candidates_for_axis(
    gray: np.ndarray, *, axis: Axis
) -> list[RepairCandidate]:
    """Recover one closed band from strict, local two-rail evidence."""
    measurements = _rail_measurements_for_axis(gray, axis=axis)
    if measurements is None:
        return []
    seeds = _two_rail_windows(measurements)
    hypotheses = [
        hypothesis
        for seed in seeds
        for hypothesis in _stable_two_rail_hypotheses(measurements, seed)
    ]
    if len(hypotheses) < 2:
        return []
    candidates: list[RepairCandidate] = []
    for upper, lower in combinations(
        sorted(hypotheses, key=lambda item: item.boundary), 2
    ):
        if upper.angle != lower.angle:
            continue
        pair_span = lower.boundary - upper.boundary
        if not (
            measurements.min_fragment <= pair_span
            <= 8 * measurements.rail_length
        ):
            continue
        if abs(lower.inverse_offset + upper.inverse_offset) > 2:
            continue
        combined_gain = upper.continuity_gain + lower.continuity_gain
        if combined_gain < 0.12:
            continue
        bounded_repair = _bounded_local_two_rail_repair(
            gray,
            axis=axis,
            angle=upper.angle,
            upper_boundary=upper.boundary,
            lower_boundary=lower.boundary,
            middle_inverse_offset=upper.inverse_offset,
            y0=min(upper.y0, lower.y0),
            y1=max(upper.y1, lower.y1),
        )
        if bounded_repair is None:
            continue
        (
            candidate,
            local_upper,
            local_lower,
            page_loss_fraction,
            local_loss_fraction,
        ) = bounded_repair
        if page_loss_fraction > 0.02:
            continue
        minimum_angle_margin = min(
            upper.angle_margin, lower.angle_margin
        )
        minimum_offset_margin = min(
            upper.offset_margin, lower.offset_margin
        )
        total_gain = (
            combined_gain
            + 0.5 * min(minimum_angle_margin, minimum_offset_margin)
            - 0.35 * page_loss_fraction
        )
        candidates.append(
            replace(
                candidate,
                partition_angle_degrees=upper.angle,
                pixel_score_before=0.5
                * (
                    upper.continuity_before
                    + lower.continuity_before
                ),
                pixel_score_after=0.5
                * (
                    upper.continuity_after
                    + lower.continuity_after
                ),
                score_terms={
                    "method_two_rail_bounded_local": 1.0,
                    "angle_degrees": upper.angle,
                    "angle_margin": minimum_angle_margin,
                    "angle_margin_threshold": 0.03,
                    "upper_angle_competitor_count": float(
                        upper.angle_competitor_count
                    ),
                    "lower_angle_competitor_count": float(
                        lower.angle_competitor_count
                    ),
                    "upper_angle_rejected_competitor_count": float(
                        upper.angle_rejected_competitor_count
                    ),
                    "lower_angle_rejected_competitor_count": float(
                        lower.angle_rejected_competitor_count
                    ),
                    "upper_continuity_gain": upper.continuity_gain,
                    "lower_continuity_gain": lower.continuity_gain,
                    "combined_continuity_gain": combined_gain,
                    "per_boundary_gain_threshold": 0.05,
                    "combined_gain_threshold": 0.12,
                    "upper_literal_gain": upper.literal_gain,
                    "lower_literal_gain": lower.literal_gain,
                    "upper_non_rule_gain": upper.non_rule_gain,
                    "lower_non_rule_gain": lower.non_rule_gain,
                    "offset_margin": minimum_offset_margin,
                    "offset_margin_threshold": 0.03,
                    "pair_span": float(pair_span),
                    "pair_span_limit": float(
                        8 * measurements.rail_length
                    ),
                    "upper_support_bins": float(
                        upper.support_bin_count
                    ),
                    "lower_support_bins": float(
                        lower.support_bin_count
                    ),
                    "upper_support_bin_span": float(
                        upper.support_bin_span
                    ),
                    "lower_support_bin_span": float(
                        lower.support_bin_span
                    ),
                    "minimum_support_bins": 2.0,
                    "matched_rail_count": 2.0,
                    "matched_fraction": min(
                        upper.matched_fraction,
                        lower.matched_fraction,
                    ),
                    "matched_span": min(
                        upper.matched_span,
                        lower.matched_span,
                    ),
                    "upper_matched_span": upper.matched_span,
                    "lower_matched_span": lower.matched_span,
                    "upper_matched_span_threshold": (
                        upper.matched_span_threshold
                    ),
                    "lower_matched_span_threshold": (
                        lower.matched_span_threshold
                    ),
                    "pair_residual": max(
                        upper.pair_residual,
                        lower.pair_residual,
                    ),
                    "center_tolerance": float(
                        measurements.center_tolerance
                    ),
                    "minimum_inverse_offset": 8.0,
                    "upper_maximum_inverse_offset": float(
                        upper.maximum_inverse_offset
                    ),
                    "lower_maximum_inverse_offset": float(
                        lower.maximum_inverse_offset
                    ),
                    "reciprocal_offset_residual": float(
                        abs(
                            lower.inverse_offset
                            + upper.inverse_offset
                        )
                    ),
                    "reciprocal_offset_tolerance": 2.0,
                    "boundary_stability": max(
                        upper.boundary_stability,
                        lower.boundary_stability,
                    ),
                    "boundary_stability_tolerance": 2.0,
                    "offset_stability": max(
                        upper.offset_stability,
                        lower.offset_stability,
                    ),
                    "offset_stability_tolerance": float(
                        measurements.center_tolerance
                    ),
                    "bounded_window_x0": 0.0,
                    "bounded_window_y0": float(
                        min(upper.y0, lower.y0)
                    ),
                    "bounded_window_x1": float(
                        measurements.canonical.shape[1]
                    ),
                    "bounded_window_y1": float(
                        max(upper.y1, lower.y1)
                    ),
                    "local_upper_boundary": float(local_upper),
                    "local_lower_boundary": float(local_lower),
                    "page_loss_fraction": page_loss_fraction,
                    "page_loss_fraction_limit": 0.02,
                    "local_window_loss_fraction": local_loss_fraction,
                    "upper_local_window_loss_fraction": (
                        upper.local_window_loss_fraction
                    ),
                    "lower_local_window_loss_fraction": (
                        lower.local_window_loss_fraction
                    ),
                    "fragment_count": 3.0,
                    "total_gain": total_gain,
                },
            )
        )
    return candidates


def _rule_anchor_candidates_for_axis(
    gray: np.ndarray, *, axis: Axis, max_fragments: int
) -> list[RepairCandidate]:
    """Recover fragment steps from global rail endpoint correspondences.

    A continuous form rail is deliberately not a seam vote.  Each candidate
    seam needs multiple thin, persistent rail components to end and start at
    the same y, then matches their x centers under one shared displacement.
    """
    measurements = _rail_measurements_for_axis(gray, axis=axis)
    if measurements is None:
        return []
    canonical = measurements.canonical
    local_contrast = measurements.local_contrast
    vertical = measurements.rail_mask
    rails = list(measurements.rails)
    rail_margin = measurements.rail_margin
    non_rule = measurements.non_rule
    rail_length = measurements.rail_length
    max_rail_width = measurements.max_rail_width
    min_fragment = measurements.min_fragment
    endpoint_tolerance = measurements.endpoint_tolerance
    center_tolerance = measurements.center_tolerance
    height, width = canonical.shape
    if len(rails) < 6:
        return []

    def matched_pairs_for_offset(
        left_centers: list[float], right_centers: list[float], offset: int
    ) -> list[tuple[int, int, float, float]]:
        """Greedy one-to-one center pairs (left_index, right_index, left, right)."""
        pairs = sorted(
            (
                abs((right - left) - offset),
                left_index,
                right_index,
                left,
                right,
            )
            for left_index, left in enumerate(left_centers)
            for right_index, right in enumerate(right_centers)
            if abs((right - left) - offset) <= center_tolerance
        )
        used_left: set[int] = set()
        used_right: set[int] = set()
        matched: list[tuple[int, int, float, float]] = []
        for _, left_index, right_index, left, right in pairs:
            if left_index in used_left or right_index in used_right:
                continue
            used_left.add(left_index)
            used_right.add(right_index)
            matched.append((left_index, right_index, left, right))
        return matched

    def matched_modes(
        left_centers: list[float], right_centers: list[float]
    ) -> list[tuple[float, int, float, float, int]]:
        """Return distinct one-to-one displacement modes for two rail sets."""
        if not left_centers or not right_centers:
            return []
        modes: list[tuple[float, int, float, float, int]] = []
        for offset in {
            int(np.copysign(np.floor(abs(right - left) + 0.5), right - left))
            for left in left_centers
            for right in right_centers
        }:
            matched = matched_pairs_for_offset(left_centers, right_centers, offset)
            if not matched:
                continue
            matched_count = len(matched)
            matched_span = (
                max(item[2] for item in matched) - min(item[2] for item in matched)
                if matched_count > 1
                else 0.0
            )
            span_fraction = matched_span / max(float(width), 1.0)
            fraction = matched_count / min(len(left_centers), len(right_centers))
            modes.append(
                (
                    matched_count * fraction * span_fraction,
                    matched_count,
                    span_fraction,
                    fraction,
                    int(offset),
                )
            )
        ordered = sorted(
            modes,
            key=lambda item: item[:4] + (-abs(item[4]), item[4]),
            reverse=True,
        )
        distinct: list[tuple[float, int, float, float, int]] = []
        for mode in ordered:
            if all(
                abs(mode[4] - prior[4]) > center_tolerance
                for prior in distinct
            ):
                distinct.append(mode)
            if len(distinct) >= 12:
                break
        return distinct

    def matched_mode(
        left_centers: list[float], right_centers: list[float]
    ) -> tuple[float, int, float, float, int] | None:
        modes = matched_modes(left_centers, right_centers)
        return modes[0] if modes else None

    full_ink = local_contrast.astype(np.float32) / 255.0
    full_ink[local_contrast < 3] = 0.0
    non_rule_ink = full_ink.copy()
    non_rule_ink[rail_margin] = 0.0

    shifted_profile = _shifted_profile
    profile_cosine = _profile_cosine

    def continuity_score(boundary: int, inverse_offset: int) -> float:
        """Compare fractional ink across a seam without reading its glyphs."""
        full_depth = max(8, min(16, height // 24))
        non_rule_depth = max(16, min(32, height // 12))
        if not (
            non_rule_depth <= boundary <= height - non_rule_depth
        ):
            return 0.0
        full_left = full_ink[
            boundary - full_depth : boundary
        ].mean(axis=0)
        full_right = full_ink[
            boundary : boundary + full_depth
        ].mean(axis=0)
        non_rule_left = non_rule_ink[
            boundary - non_rule_depth : boundary
        ].mean(axis=0)
        non_rule_right = non_rule_ink[
            boundary : boundary + non_rule_depth
        ].mean(axis=0)
        return 0.5 * profile_cosine(
            full_left, shifted_profile(full_right, inverse_offset)
        ) + 0.5 * profile_cosine(
            non_rule_left, shifted_profile(non_rule_right, inverse_offset)
        )

    def edge_continuity_score(boundary: int, inverse_offset: int) -> float:
        """Locate the literal cut between adjacent rows for a proposed shift."""
        if not 1 <= boundary < height:
            return 0.0
        return profile_cosine(
            full_ink[boundary - 1],
            shifted_profile(full_ink[boundary], inverse_offset),
        )

    trace_depth = max(8, min(20, rail_length))

    def trace_centers(start: int, end: int) -> list[float]:
        """Read thin rail centres at one side of a possible cut.

        Nearby rails can touch after a small strip displacement and therefore
        share a connected component.  A short side-specific trace keeps their
        actual positions distinct without turning continuous rails into votes.
        """
        if end <= start:
            return []
        occupied = vertical[start:end].sum(axis=0) >= (end - start) * 255 * 0.7
        centers: list[float] = []
        run_start: int | None = None
        for index, value in enumerate(np.r_[occupied, False]):
            if value and run_start is None:
                run_start = index
            elif not value and run_start is not None:
                run_width = index - run_start
                if run_width <= max_rail_width:
                    centers.append(float(run_start + (run_width - 1) / 2.0))
                run_start = None
        return centers

    def nearest_rail_id(center: float, row_start: int, row_end: int) -> int:
        """Attribute a traced run center to the measured rail it belongs to."""
        covering = [
            index
            for index, (_, start, end) in enumerate(rails)
            if start <= row_end and end >= row_start
        ]
        pool = covering if covering else range(len(rails))
        distance, index = min(
            ((abs(rails[index][0] - center), index) for index in pool),
            default=(float("inf"), -1),
        )
        return index if distance <= center_tolerance else -1

    def rail_extent(rail_id: int, fallback: tuple[int, int]) -> tuple[int, int]:
        if rail_id < 0:
            return fallback
        _, start, end = rails[rail_id]
        return (int(start), int(end))

    def traced_matched_pairs(
        left_centers: list[float],
        right_centers: list[float],
        offset: int,
        left_window: tuple[int, int],
        right_window: tuple[int, int],
    ) -> tuple[_MatchedRailPair, ...]:
        records: list[_MatchedRailPair] = []
        for _, _, left, right in matched_pairs_for_offset(
            left_centers, right_centers, offset
        ):
            ending_id = nearest_rail_id(left, *left_window)
            starting_id = nearest_rail_id(right, *right_window)
            records.append(
                _MatchedRailPair(
                    ending_rail_id=ending_id,
                    starting_rail_id=starting_id,
                    ending_center=float(left),
                    starting_center=float(right),
                    ending_extent=rail_extent(ending_id, left_window),
                    starting_extent=rail_extent(starting_id, right_window),
                )
            )
        return tuple(records)

    def component_matched_pairs(
        ending_rails: list[tuple[float, int]],
        starting_rails: list[tuple[float, int]],
        offset: int,
    ) -> tuple[_MatchedRailPair, ...]:
        records: list[_MatchedRailPair] = []
        for left_index, right_index, left, right in matched_pairs_for_offset(
            [center for center, _ in ending_rails],
            [center for center, _ in starting_rails],
            offset,
        ):
            ending_id = ending_rails[left_index][1]
            starting_id = starting_rails[right_index][1]
            records.append(
                _MatchedRailPair(
                    ending_rail_id=ending_id,
                    starting_rail_id=starting_id,
                    ending_center=float(left),
                    starting_center=float(right),
                    ending_extent=(int(rails[ending_id][1]), int(rails[ending_id][2])),
                    starting_extent=(
                        int(rails[starting_id][1]),
                        int(rails[starting_id][2]),
                    ),
                )
            )
        return tuple(records)

    starts = sorted((start, index) for index, (_, start, _) in enumerate(rails))
    ends = sorted((end, index) for index, (_, _, end) in enumerate(rails))
    boundary_seeds: set[int] = set()
    start_index = 0
    for end, _ in ends:
        while start_index < len(starts) and starts[start_index][0] < end - endpoint_tolerance:
            start_index += 1
        index = start_index
        while index < len(starts) and starts[index][0] <= end + endpoint_tolerance:
            boundary_seeds.add(int(round((end + starts[index][0]) / 2.0)))
            index += 1

    hypotheses: list[_AnchoredSeam] = []
    refinement_boundaries = set(
        int(value)
        for value in (
            np.flatnonzero(np.any(vertical[1:] != vertical[:-1], axis=1)) + 1
        )
    )
    for observed_boundary in sorted(boundary_seeds):
        ending = tuple(
            index for end, index in ends if abs(end - observed_boundary) <= endpoint_tolerance
        )
        starting = tuple(
            index
            for start, index in starts
            if abs(start - observed_boundary) <= endpoint_tolerance
        )
        if len(ending) < 3 or len(starting) < 3:
            continue
        ending_rails = sorted((rails[index][0], index) for index in ending)
        starting_rails = sorted((rails[index][0], index) for index in starting)
        component_modes = matched_modes(
            [center for center, _ in ending_rails],
            [center for center, _ in starting_rails],
        )
        if not component_modes:
            continue
        trace_window_left = (max(0, observed_boundary - trace_depth), observed_boundary)
        trace_window_right = (
            observed_boundary,
            min(height, observed_boundary + trace_depth),
        )
        trace_left = trace_centers(*trace_window_left)
        trace_right = trace_centers(*trace_window_right)
        trace_modes = matched_modes(trace_left, trace_right)
        modes = trace_modes if trace_modes else component_modes
        qualified = [
            mode
            for mode in modes
            if (
                abs(mode[4]) >= 8
                and mode[1] >= 3
                and mode[3] >= 0.5
                and mode[2] >= 0.35
                and mode[0] >= 2.0
            )
        ]
        if not qualified:
            continue
        support_window = non_rule[
            max(0, observed_boundary - 2 * rail_length) : min(
                height, observed_boundary + 2 * rail_length
            )
        ]
        non_rule_support = float(support_window.sum())
        if non_rule_support < 12.0:
            continue
        alias_options: list[
            tuple[
                float,
                float,
                float,
                float,
                int,
                int,
                int,
                tuple[float, int, float, float, int],
            ]
        ] = []
        for mode in qualified:
            observed_offset = mode[4]
            for inverse_offset in range(
                -observed_offset - center_tolerance,
                -observed_offset + center_tolerance + 1,
            ):
                boundary = max(
                    range(
                        max(min_fragment, observed_boundary - 3),
                        min(height - min_fragment, observed_boundary + 3) + 1,
                    ),
                    key=lambda item: (
                        edge_continuity_score(item, inverse_offset),
                        -abs(item - observed_boundary),
                    ),
                )
                continuity = continuity_score(
                    boundary, inverse_offset
                )
                edge_continuity = edge_continuity_score(
                    boundary, inverse_offset
                )
                alias_options.append(
                    (
                        continuity + 0.25 * edge_continuity,
                        continuity,
                        edge_continuity,
                        mode[0],
                        -abs(inverse_offset + observed_offset),
                        boundary,
                        inverse_offset,
                        mode,
                    )
                )
        if not alias_options:
            continue
        alias_options.sort(reverse=True)
        (
            best_alias_score,
            best_continuity,
            _,
            score,
            _,
            boundary,
            inverse_offset,
            best_mode,
        ) = alias_options[0]
        _, matched_count, _, _, observed_offset = best_mode
        second_alias_score = max(
            (
                item[0]
                for item in alias_options[1:]
                if abs(item[7][4] - observed_offset) > center_tolerance
            ),
            default=0.0,
        )
        alias_margin = best_alias_score - second_alias_score
        baseline_continuity = continuity_score(boundary, 0)
        if trace_modes:
            matched_pairs = traced_matched_pairs(
                trace_left,
                trace_right,
                observed_offset,
                trace_window_left,
                trace_window_right,
            )
        else:
            matched_pairs = component_matched_pairs(
                ending_rails, starting_rails, observed_offset
            )
        hypotheses.append(
            _AnchoredSeam(
                seam=_SeamHypothesis(
                    boundary=boundary,
                    relative_offset=inverse_offset,
                    similarity_before=baseline_continuity,
                    similarity_after=best_continuity,
                    edge_similarity_before=0.0,
                    edge_similarity_after=0.0,
                    shared_support=float(matched_count),
                    gain=best_continuity - baseline_continuity,
                    objective=best_alias_score + 0.02 * score,
                ),
                source="rail_endpoint",
                matched_rail_count=matched_count,
                non_rule_support=non_rule_support,
                alias_margin=alias_margin,
                matched_pairs=matched_pairs,
                support_interval=(
                    max(0, observed_boundary - 2 * rail_length),
                    min(height, observed_boundary + 2 * rail_length),
                ),
                objective=best_alias_score + 0.02 * score,
            )
        )

    selected: list[_AnchoredSeam] = []
    for item in sorted(
        hypotheses, key=lambda value: (-value.objective, value.seam.boundary)
    ):
        if all(
            abs(item.seam.boundary - prior.seam.boundary) > 2 * endpoint_tolerance
            for prior in selected
        ):
            selected.append(item)
    if not selected:
        return []

    # Once a true global endpoint event has established that this is a broken
    # lattice, retain small, immediately adjacent lattice steps in its coherent
    # page chain.  These are refinements, never standalone candidates: a smooth
    # one-pixel drift has no qualifying global event above and therefore still
    # abstains.  This matters when two rails only three pixels apart touch into
    # one CC at their shared cut.
    refinements: list[_AnchoredSeam] = []
    for observed_boundary in sorted(refinement_boundaries):
        step_left = trace_centers(observed_boundary - 1, observed_boundary)
        step_right = trace_centers(observed_boundary, observed_boundary + 1)
        mode = matched_mode(step_left, step_right)
        if mode is None:
            continue
        score, matched_count, span_fraction, fraction, observed_offset = mode
        if (
            abs(observed_offset) <= center_tolerance
            or matched_count < 3
            or fraction < 0.5
            or span_fraction < 0.35
            or score < 2.0
        ):
            continue
        support_interval = (
            max(0, observed_boundary - 2 * rail_length),
            min(height, observed_boundary + 2 * rail_length),
        )
        support = float(
            non_rule[support_interval[0] : support_interval[1]].sum()
        )
        if support < 12.0:
            continue
        inverse_offset = -observed_offset
        boundary = max(
            range(
                max(min_fragment, observed_boundary - 3),
                min(height - min_fragment, observed_boundary + 3) + 1,
            ),
            key=lambda item: (
                edge_continuity_score(item, inverse_offset),
                -abs(item - observed_boundary),
            ),
        )
        continuity = continuity_score(boundary, inverse_offset)
        baseline_continuity = continuity_score(boundary, 0)
        refinements.append(
            _AnchoredSeam(
                seam=_SeamHypothesis(
                    boundary=boundary,
                    relative_offset=inverse_offset,
                    similarity_before=baseline_continuity,
                    similarity_after=continuity,
                    edge_similarity_before=0.0,
                    edge_similarity_after=0.0,
                    shared_support=float(matched_count),
                    gain=continuity - baseline_continuity,
                    objective=continuity + 0.02 * score,
                ),
                source="rail_endpoint",
                matched_rail_count=matched_count,
                non_rule_support=support,
                # Legacy evidence tuple item [5]: the refinement path has
                # always fed continuity into the alias-margin slot; keep it.
                alias_margin=continuity,
                matched_pairs=traced_matched_pairs(
                    step_left,
                    step_right,
                    observed_offset,
                    (observed_boundary - 1, observed_boundary),
                    (observed_boundary, observed_boundary + 1),
                ),
                support_interval=support_interval,
                objective=continuity + 0.02 * score,
            )
        )
    for item in sorted(
        refinements, key=lambda value: (-value.objective, value.seam.boundary)
    ):
        if all(
            abs(item.seam.boundary - prior.seam.boundary) > 2 * endpoint_tolerance
            for prior in selected
        ):
            selected.append(item)
    selected.sort(key=lambda value: value.seam.boundary)

    def scored(
        candidate: RepairCandidate,
        evidence: _AnchoredSeam,
    ) -> RepairCandidate:
        objective = evidence.objective
        seam = evidence.seam
        matched_count = evidence.matched_rail_count
        support = evidence.non_rule_support
        continuity = seam.similarity_after
        alias_margin = evidence.alias_margin
        before = seam.similarity_before
        raw_gain = continuity - before
        loss_fraction = float(
            (candidate.overlap_mask.sum() + candidate.uncovered_mask.sum())
            / max(candidate.reconstruction.size, 1)
        )
        complexity = 0.015 * (len(candidate.fragments) - 1)
        loss_penalty = 0.35 * loss_fraction
        return replace(
            candidate,
            pixel_score_before=before,
            pixel_score_after=continuity,
            score_terms={
                "method_rule_anchor": 1.0,
                "rail_endpoint_score": objective,
                "non_rule_continuity": continuity,
                "non_rule_continuity_before": before,
                "non_rule_continuity_after": continuity,
                "non_rule_continuity_gain": raw_gain,
                "periodic_alias_margin": alias_margin,
                "matched_rail_count": float(matched_count),
                "matched_rail_span_fraction": objective
                / max(float(matched_count), 1.0),
                "non_rule_support": support,
                "observed_relative_offset": float(-seam.relative_offset),
                "complexity_penalty": complexity,
                "loss_fraction": loss_fraction,
                "loss_penalty": loss_penalty,
                "fragment_count": float(len(candidate.fragments)),
                "total_gain": raw_gain - complexity - loss_penalty,
            },
        )

    support_floor = 0.25 * max(item.non_rule_support for item in selected)
    page_region = (0, 0, gray.shape[1], gray.shape[0])

    full_candidates: list[RepairCandidate] = []
    alias_chain_confident = all(
        item.seam.similarity_after >= 0.08 and item.alias_margin >= 0.03
        for item in selected
    )
    if len(selected) <= max_fragments - 1 and alias_chain_confident:
        # Spec 10.2: the all-page chain passes the same joint scorer and
        # per-seam gates as regional components; the old best-seam score
        # inheritance is deliberately gone.
        joint = _joint_component_score(
            tuple(selected),
            gray=gray,
            region=page_region,
            axis=axis,
            measurements=measurements,
            support_floor=support_floor,
        )
        if joint is not None:
            seams = tuple(item.seam for item in selected)
            transforms = _transforms_from_seams(seams, height, axis, anchor=0)
            candidate = replace(
                apply_fragment_transforms(gray, transforms, partition_axis=axis),
                pixel_score_before=joint["mean_before"],
                pixel_score_after=joint["mean_after"],
                score_terms=_joint_score_terms(
                    joint, max(selected, key=lambda item: item.objective)
                ),
            )
            full_candidates.append(candidate)
            _COMPONENT_DIAGNOSTICS[_geometry_signature_key(candidate)] = (
                _component_diagnostics_record(
                    tuple(selected),
                    joint,
                    measurements=measurements,
                    region=page_region,
                    axis=axis,
                    support_floor=support_floor,
                )
            )

    # Spec 8.1 steps 4-8: bounded regional components between the all-page
    # chain and the independent single-seam candidates.
    component_candidates: list[RepairCandidate] = []
    expanded_regions = [
        _expand_region(
            region,
            rail_length=rail_length,
            min_fragment=min_fragment,
            page_shape=gray.shape,
        )
        for region in propose_candidate_regions(gray, max_regions=12)
    ]
    assigned = _assign_rail_seams_to_regions(selected, expanded_regions, axis=axis)
    rail_tracks = (
        _trace_rail_tracks(vertical, tuple(rails), max_rail_width=max_rail_width)
        if assigned
        else ()
    )
    for region_index in sorted(assigned):
        expanded = expanded_regions[region_index]
        ex0, ey0, ex1, ey1 = expanded
        part_low, part_high = (ey0, ey1) if axis == "y" else (ex0, ex1)
        rail_nodes = [
            item
            for item in assigned[region_index]
            if part_low + min_fragment
            <= item.seam.boundary
            <= part_high - min_fragment
        ]
        if not rail_nodes:
            continue
        # Spec 8.3: a glyph seam cannot replace or override a rail seam --
        # coincident glyph detections would only poison window contiguity.
        glyph_nodes = [
            item
            for item in _regional_glyph_seams(
                gray,
                expanded,
                axis=axis,
                min_fragment=min_fragment,
                measurements=measurements,
            )
            if all(
                abs(item.seam.boundary - rail.seam.boundary) >= min_fragment
                for rail in rail_nodes
            )
        ]
        for path in _build_regional_components(
            rail_nodes + glyph_nodes,
            axis=axis,
            max_fragments=max_fragments,
            min_fragment=min_fragment,
            center_tolerance=center_tolerance,
            rail_tracks=rail_tracks,
            displacement_bound=width,
        ):
            if (
                _joint_component_score(
                    path,
                    gray=gray,
                    region=expanded,
                    axis=axis,
                    measurements=measurements,
                    support_floor=support_floor,
                )
                is None
            ):
                continue
            # Spec 9.2: reconstruct under the loss-minimizing gauge and
            # re-apply the joint gates to the selected reconstruction.
            selection = _select_gauge(
                gray,
                path,
                region=expanded,
                axis=axis,
                measurements=measurements,
            )
            joint = _joint_component_score(
                path,
                gray=gray,
                region=expanded,
                axis=axis,
                measurements=measurements,
                support_floor=support_floor,
                candidate=selection.candidate,
            )
            if joint is None:
                continue
            rail_evidence = max(
                (item for item in path if item.source == "rail_endpoint"),
                key=lambda item: item.objective,
            )
            candidate = replace(
                selection.candidate,
                pixel_score_before=joint["mean_before"],
                pixel_score_after=joint["mean_after"],
                score_terms={
                    **_joint_score_terms(joint, rail_evidence),
                    "method_rule_anchor_component": 1.0,
                    "seam_count": joint["seam_count"],
                    "overlap_fraction": joint["overlap_fraction"],
                    "uncovered_fraction": joint["uncovered_fraction"],
                    "cropped_source_fraction": joint["cropped_source_fraction"],
                },
            )
            component_candidates.append(candidate)
            diagnostics_record = _component_diagnostics_record(
                path,
                joint,
                measurements=measurements,
                region=expanded,
                axis=axis,
                support_floor=support_floor,
            )
            diagnostics_record["gauge"] = {
                "selected_offset": selection.gauge_offset,
                "anchor_fragment_index": selection.anchor_fragment_index,
                "objective_tuple": list(selection.objective_tuple),
                "evaluated": [dict(record) for record in selection.evaluated],
            }
            _COMPONENT_DIAGNOSTICS[_geometry_signature_key(candidate)] = (
                diagnostics_record
            )

    radius = max(2 * rail_length, min(320, height // 8))
    local_candidates: list[RepairCandidate] = []
    for evidence in selected:
        seam = evidence.seam
        start, end = max(0, seam.boundary - radius), min(height, seam.boundary + radius)
        local_boundary = seam.boundary - start
        if local_boundary < min_fragment or end - start - local_boundary < min_fragment:
            continue
        if axis == "y":
            crop = gray[start:end, :]
            region = (0, start, gray.shape[1], end)
            transforms = (
                FragmentTransform((0, local_boundary)),
                FragmentTransform((local_boundary, crop.shape[0]), inverse_dx=seam.relative_offset),
            )
        else:
            crop = gray[:, start:end]
            region = (start, 0, end, gray.shape[0])
            transforms = (
                FragmentTransform((0, local_boundary)),
                FragmentTransform((local_boundary, crop.shape[1]), inverse_dy=seam.relative_offset),
            )
        scored_candidate = replace(
            scored(
                apply_fragment_transforms(crop, transforms, partition_axis=axis),
                evidence,
            ),
            region=region,
        )
        if scored_candidate.score_terms["total_gain"] > 0.035:
            local_candidates.append(scored_candidate)
    return full_candidates + component_candidates + local_candidates


def _expand_region(
    region: tuple[int, int, int, int],
    *,
    rail_length: int,
    min_fragment: int,
    page_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand an automatic region by max(2*rail_length, min_fragment) per side,
    clipped to the page (spec 8.1 step 4)."""
    height, width = page_shape
    margin = max(2 * rail_length, min_fragment)
    x0, y0, x1, y1 = region
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def _rail_seam_cross_interval(seam: _AnchoredSeam) -> tuple[float, float]:
    """Cross-axis span of a rail seam's matched endpoint centers — the
    rail-node counterpart of the glyph non-rule support bounding interval
    (spec 8.4 rule 3)."""
    centers = [
        center
        for pair in seam.matched_pairs
        for center in (pair.ending_center, pair.starting_center)
    ]
    if not centers:
        return (0.0, 0.0)
    return (min(centers), max(centers))


def _assign_rail_seams_to_regions(
    rail_seams: list[_AnchoredSeam],
    regions: list[tuple[int, int, int, int]],
    *,
    axis: Axis,
) -> dict[int, list[_AnchoredSeam]]:
    """Assign full-page rail seams to expanded regions (spec 8.1 step 5): the
    boundary must lie strictly inside the region's partition-axis interval and
    the seam's cross-axis span must overlap the region's orthogonal interval
    by at least 35% of the smaller interval."""
    assigned: dict[int, list[_AnchoredSeam]] = {}
    for index, (x0, y0, x1, y1) in enumerate(regions):
        if axis == "y":
            partition_interval, orthogonal_interval = (y0, y1), (x0, x1)
        else:
            partition_interval, orthogonal_interval = (x0, x1), (y0, y1)
        for seam in rail_seams:
            if not partition_interval[0] < seam.seam.boundary < partition_interval[1]:
                continue
            low, high = _rail_seam_cross_interval(seam)
            overlap = min(high, orthogonal_interval[1]) - max(
                low, orthogonal_interval[0]
            )
            smaller = min(
                high - low, orthogonal_interval[1] - orthogonal_interval[0]
            )
            if smaller <= 0 or overlap < 0.35 * smaller:
                continue
            assigned.setdefault(index, []).append(seam)
    return assigned


def _regional_glyph_seams(
    gray: np.ndarray,
    region: tuple[int, int, int, int],
    *,
    axis: Axis,
    min_fragment: int,
    measurements: _RailMeasurements,
) -> list[_AnchoredSeam]:
    """Run the existing rule-suppressed seam detector inside one expanded
    region crop and remap boundaries and support intervals to page
    coordinates (spec 8.1 step 6, 8.3). Detection is `_seam_hypotheses`
    unchanged — same bounded offset search, gain/support gates, NMS, and seed
    cap; this only relocates it and packages `_AnchoredSeam` records."""
    x0, y0, x1, y1 = region
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    evidence = _alignment_evidence(crop, build_content_masks(crop))
    hypotheses = _seam_hypotheses(crop, evidence, axis)
    if not hypotheses:
        return []
    partition_origin = y0 if axis == "y" else x0
    cross_origin = x0 if axis == "y" else y0
    partition_size = crop.shape[0] if axis == "y" else crop.shape[1]
    canonical_length = measurements.canonical.shape[0]
    span = 3  # the detector's fixed seam window
    seams: list[_AnchoredSeam] = []
    for hypothesis in hypotheses:
        boundary = hypothesis.boundary
        if boundary < min_fragment or partition_size - boundary < min_fragment:
            continue
        if axis == "y":
            band = evidence[max(0, boundary - span) : boundary + span, :].max(axis=0)
        else:
            band = evidence[:, max(0, boundary - span) : boundary + span].max(axis=1)
        occupied = np.flatnonzero(band > 0)
        if occupied.size == 0:
            continue
        page_boundary = boundary + partition_origin
        support_slice = measurements.non_rule[
            max(0, page_boundary - 2 * measurements.rail_length) : min(
                canonical_length, page_boundary + 2 * measurements.rail_length
            )
        ]
        seams.append(
            _AnchoredSeam(
                seam=replace(hypothesis, boundary=page_boundary),
                source="glyph_continuity",
                matched_rail_count=0,
                non_rule_support=float(support_slice.sum()),
                alias_margin=_glyph_seam_alias_margin(
                    crop,
                    evidence,
                    hypothesis,
                    axis=axis,
                    center_tolerance=measurements.center_tolerance,
                ),
                matched_pairs=(),
                support_interval=(
                    int(occupied[0]) + cross_origin,
                    int(occupied[-1]) + 1 + cross_origin,
                ),
                objective=hypothesis.objective,
            )
        )
    return seams


def _glyph_seam_alias_margin(
    crop: np.ndarray,
    evidence: np.ndarray,
    hypothesis: _SeamHypothesis,
    *,
    axis: Axis,
    center_tolerance: int,
) -> float:
    """Objective margin of the selected offset over every distinct competing
    offset at the same boundary, mirroring the rail path's alias margin (a
    missing competitor contributes 0.0). Uses the detector's own primitives
    and objective formula at one boundary; no detection is re-run."""
    partition_size = evidence.shape[1] if axis == "x" else evidence.shape[0]
    repair_size = evidence.shape[0] if axis == "x" else evidence.shape[1]
    max_offset = min(
        max(16, int(round(repair_size * 0.04))),
        max(8, int(repair_size * 0.20)),
        96,
    )
    min_offset = max(2, min(8, int(np.ceil(repair_size * 0.02))))
    boundary = hypothesis.boundary
    span = 3
    gradient_axis = 0 if axis == "x" else 1
    edge = np.abs(np.diff(crop.astype(np.int16), axis=gradient_axis)) > 10
    if gradient_axis == 0:
        edge = np.pad(edge, ((0, 1), (0, 0)))
    else:
        edge = np.pad(edge, ((0, 0), (0, 1)))
    edge = edge.astype(np.float32)
    if axis == "x":
        left = evidence[:, boundary - span : boundary].max(axis=1)
        right = evidence[:, boundary : boundary + span].max(axis=1)
        left_edge = edge[:, boundary - span : boundary].max(axis=1)
        right_edge = edge[:, boundary : boundary + span].max(axis=1)
    else:
        left = evidence[boundary - span : boundary, :].max(axis=0)
        right = evidence[boundary : boundary + span, :].max(axis=0)
        left_edge = edge[boundary - span : boundary, :].max(axis=0)
        right_edge = edge[boundary : boundary + span, :].max(axis=0)
    before, _ = _shifted_soft_iou(left, right, 0)
    edge_before, _ = _shifted_soft_iou(left_edge, right_edge, 0)

    def objective(offset: int) -> float:
        similarity, _ = _shifted_soft_iou(left, right, offset)
        edge_similarity, _ = _shifted_soft_iou(left_edge, right_edge, offset)
        return (
            (similarity - before)
            + (edge_similarity - edge_before)
            - 0.35 * abs(offset) / max(float(repair_size), 1.0)
        )

    selected = hypothesis.relative_offset
    competitor = max(
        (
            objective(offset)
            for offset in range(-max_offset, max_offset + 1)
            if abs(offset) >= min_offset
            and abs(offset - selected) > center_tolerance
        ),
        default=0.0,
    )
    return hypothesis.objective - competitor


def _node_cross_interval(seam: _AnchoredSeam) -> tuple[float, float]:
    """Cross-axis evidence footprint for component edge rule 3 (spec 8.4):
    the non-rule support bounding interval for a glyph node, the matched
    endpoint-center span for a rail node."""
    if seam.source == "glyph_continuity":
        return (float(seam.support_interval[0]), float(seam.support_interval[1]))
    return _rail_seam_cross_interval(seam)


def _intact_rail_gap(
    rail_tracks: tuple[_RailTrack, ...],
    low_boundary: int,
    high_boundary: int,
    *,
    center_tolerance: int,
) -> bool:
    """Spec 8.4 rule 4: at least three tracks crossing the full inter-node
    interval with >=75% of their pooled sampled offsets within
    ``center_tolerance`` of one constant certify the gap as intact.

    Crossing means extending strictly past BOTH boundaries: a rail segment
    spanning exactly one displaced fragment interval starts and ends at the
    cuts and is evidence of the seams, not against them."""
    offsets: list[float] = []
    crossing = 0
    for track in rail_tracks:
        start, end = track.extent
        if not (start < low_boundary and end > high_boundary):
            continue
        crossing += 1
        for row in range(low_boundary + 1, high_boundary):
            offsets.append(track.row_centers[row - start] - track.center)
    if crossing < 3 or not offsets:
        return False
    values = np.asarray(offsets, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    constant = float(np.median(finite))
    within = np.isfinite(values) & (np.abs(values - constant) <= center_tolerance)
    return float(within.mean()) >= 0.75


def _build_regional_components(
    anchored: list[_AnchoredSeam],
    *,
    axis: Axis,
    max_fragments: int,
    min_fragment: int,
    center_tolerance: int,
    rail_tracks: tuple[_RailTrack, ...],
    displacement_bound: int,
) -> list[tuple[_AnchoredSeam, ...]]:
    """Spec 8.4: deterministic region-local seam components.

    Retains at most eight rail and eight glyph nodes (descending objective,
    boundary, offset), connects boundary-adjacent retained nodes under edge
    rules 2-5, and enumerates contiguous windows of the retained node list
    (never splicing around an unconnectable node) of at most
    ``max_fragments - 1`` seams containing at least one rail seam, with every
    cumulative offset prefix within ``displacement_bound``.  Returns at most
    32 paths ordered by descending mean seam gain, fewer fragments, boundary
    tuple, offset tuple."""
    if axis not in ("y", "x"):
        raise ValueError("axis must be 'y' or 'x'")
    max_seams = max_fragments - 1
    if max_seams < 1:
        return []

    def cap_order(seam: _AnchoredSeam) -> tuple[float, int, int]:
        return (-seam.objective, seam.seam.boundary, seam.seam.relative_offset)

    rail_nodes = sorted(
        (item for item in anchored if item.source == "rail_endpoint"), key=cap_order
    )[:8]
    glyph_nodes = sorted(
        (item for item in anchored if item.source == "glyph_continuity"), key=cap_order
    )[:8]
    nodes = sorted(
        rail_nodes + glyph_nodes,
        key=lambda item: (item.seam.boundary, item.seam.relative_offset, item.source),
    )
    if not nodes:
        return []

    def connected(first: _AnchoredSeam, second: _AnchoredSeam) -> bool:
        if second.seam.boundary - first.seam.boundary < min_fragment:
            return False
        first_low, first_high = _node_cross_interval(first)
        second_low, second_high = _node_cross_interval(second)
        overlap = min(first_high, second_high) - max(first_low, second_low)
        smaller = min(first_high - first_low, second_high - second_low)
        if smaller <= 0 or overlap < 0.35 * smaller:
            return False
        if first.alias_margin < 0.03 or second.alias_margin < 0.03:
            return False
        return not _intact_rail_gap(
            rail_tracks,
            first.seam.boundary,
            second.seam.boundary,
            center_tolerance=center_tolerance,
        )

    edges = [connected(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    paths: list[tuple[_AnchoredSeam, ...]] = []
    for start in range(len(nodes)):
        cumulative = 0
        for stop in range(start, min(start + max_seams, len(nodes))):
            if stop > start and not edges[stop - 1]:
                break
            cumulative += nodes[stop].seam.relative_offset
            if abs(cumulative) > displacement_bound:
                break
            window = tuple(nodes[start : stop + 1])
            if any(item.source == "rail_endpoint" for item in window):
                paths.append(window)
    paths.sort(
        key=lambda path: (
            -sum(item.seam.gain for item in path) / len(path),
            len(path),
            tuple(item.seam.boundary for item in path),
            tuple(item.seam.relative_offset for item in path),
        )
    )
    return paths[:32]


def _joint_seam_continuity(
    measurements: _RailMeasurements,
    boundary: int,
    offset: int,
    *,
    part_interval: tuple[int, int],
    cross_interval: tuple[int, int],
) -> float:
    """Spec 10.1: the frozen rule-anchor continuity formula with depths from
    the region's partition length and profiles restricted to the region's
    cross-axis interval.  With the full page as the region this reproduces the
    legacy ``continuity_score`` closure exactly."""
    part_lo, part_hi = part_interval
    region_len = part_hi - part_lo
    full_depth = max(8, min(16, region_len // 24))
    non_rule_depth = max(16, min(32, region_len // 12))
    if not (part_lo + non_rule_depth <= boundary <= part_hi - non_rule_depth):
        return 0.0
    cross_lo, cross_hi = cross_interval
    contrast = measurements.local_contrast[:, cross_lo:cross_hi]
    margin = measurements.rail_margin[:, cross_lo:cross_hi]

    def ink(rows: np.ndarray) -> np.ndarray:
        values = rows.astype(np.float32) / 255.0
        values[rows < 3] = 0.0
        return values

    full_left = ink(contrast[boundary - full_depth : boundary]).mean(axis=0)
    full_right = ink(contrast[boundary : boundary + full_depth]).mean(axis=0)
    non_rule_left = ink(contrast[boundary - non_rule_depth : boundary])
    non_rule_left[margin[boundary - non_rule_depth : boundary]] = 0.0
    non_rule_right = ink(contrast[boundary : boundary + non_rule_depth])
    non_rule_right[margin[boundary : boundary + non_rule_depth]] = 0.0
    return 0.5 * _profile_cosine(
        full_left, _shifted_profile(full_right, offset)
    ) + 0.5 * _profile_cosine(
        non_rule_left.mean(axis=0),
        _shifted_profile(non_rule_right.mean(axis=0), offset),
    )


def _candidate_loss_fractions(
    candidate: RepairCandidate, source_size: int
) -> tuple[float, float, float, float]:
    """Spec 9.3 canonical loss terms: (overlap, uncovered, cropped-source,
    loss) fractions, with destination loss as the mask union."""
    size = max(candidate.reconstruction.size, 1)
    overlap_fraction = float(candidate.overlap_mask.sum()) / size
    uncovered_fraction = float(candidate.uncovered_mask.sum()) / size
    destination_loss = (
        float((candidate.overlap_mask | candidate.uncovered_mask).sum()) / size
    )
    cropped_fraction = float(candidate.cropped_source_mask.sum()) / max(
        source_size, 1
    )
    return (
        overlap_fraction,
        uncovered_fraction,
        cropped_fraction,
        max(destination_loss, cropped_fraction),
    )


def _standalone_rail_seam_ok(
    anchored: _AnchoredSeam,
    *,
    gray: np.ndarray,
    axis: Axis,
    measurements: _RailMeasurements,
    support_floor: float,
) -> bool:
    """Spec 10.2 frozen standalone predicate for one rail seam: the existing
    two-fragment local crop must independently justify the seam."""
    height = measurements.canonical.shape[0]
    width = measurements.canonical.shape[1]
    boundary = anchored.seam.boundary
    offset = anchored.seam.relative_offset
    before = _joint_seam_continuity(
        measurements,
        boundary,
        0,
        part_interval=(0, height),
        cross_interval=(0, width),
    )
    after = _joint_seam_continuity(
        measurements,
        boundary,
        offset,
        part_interval=(0, height),
        cross_interval=(0, width),
    )
    gain = after - before
    if gain < 0.05:
        return False
    radius = max(2 * measurements.rail_length, min(320, height // 8))
    start, end = max(0, boundary - radius), min(height, boundary + radius)
    local_boundary = boundary - start
    if (
        local_boundary < measurements.min_fragment
        or end - start - local_boundary < measurements.min_fragment
    ):
        return False
    if axis == "y":
        crop = gray[start:end, :]
        transforms = (
            FragmentTransform((0, local_boundary)),
            FragmentTransform((local_boundary, crop.shape[0]), inverse_dx=offset),
        )
    else:
        crop = gray[:, start:end]
        transforms = (
            FragmentTransform((0, local_boundary)),
            FragmentTransform((local_boundary, crop.shape[1]), inverse_dy=offset),
        )
    candidate = apply_fragment_transforms(crop, transforms, partition_axis=axis)
    _, _, _, loss_fraction = _candidate_loss_fractions(candidate, crop.size)
    if gain - 0.015 - 0.35 * loss_fraction <= 0.035:
        return False
    band = measurements.non_rule[
        max(0, boundary - 2 * measurements.rail_length) : min(
            height, boundary + 2 * measurements.rail_length
        )
    ]
    return float(band.sum()) >= support_floor


def _joint_component_score(
    component: tuple[_AnchoredSeam, ...],
    *,
    gray: np.ndarray,
    region: tuple[int, int, int, int],
    axis: Axis,
    measurements: _RailMeasurements,
    support_floor: float = 0.0,
    candidate: RepairCandidate | None = None,
) -> dict[str, float] | None:
    """Spec 10: joint scalar score for one component (or the all-page chain,
    with the full page as the region); None when any 10.2 gate fails.

    ``candidate`` supplies a gauge-selected reconstruction whose masks define
    the loss terms; without it the anchor-zero reconstruction is scored."""
    seams = tuple(sorted(component, key=lambda item: item.seam.boundary))
    if not seams:
        return None
    x0, y0, x1, y1 = region
    if axis == "y":
        part_interval, cross_interval = (y0, y1), (x0, x1)
    else:
        part_interval, cross_interval = (x0, x1), (y0, y1)
    befores: list[float] = []
    afters: list[float] = []
    for item in seams:
        before = _joint_seam_continuity(
            measurements,
            item.seam.boundary,
            0,
            part_interval=part_interval,
            cross_interval=cross_interval,
        )
        after = _joint_seam_continuity(
            measurements,
            item.seam.boundary,
            item.seam.relative_offset,
            part_interval=part_interval,
            cross_interval=cross_interval,
        )
        if after - before < 0.05:
            return None
        befores.append(before)
        afters.append(after)
    for item in seams:
        if item.source == "rail_endpoint" and not _standalone_rail_seam_ok(
            item,
            gray=gray,
            axis=axis,
            measurements=measurements,
            support_floor=support_floor,
        ):
            return None
    part_lo, part_hi = part_interval
    region_len = part_hi - part_lo
    local_seams = tuple(
        replace(item.seam, boundary=item.seam.boundary - part_lo) for item in seams
    )
    edges = (0, *(seam.boundary for seam in local_seams), region_len)
    if min(end - start for start, end in zip(edges, edges[1:])) < (
        measurements.min_fragment
    ):
        return None
    crop = gray[y0:y1, x0:x1]
    if candidate is None:
        transforms = _transforms_from_seams(
            local_seams, region_len, axis, anchor=0
        )
        candidate = apply_fragment_transforms(
            crop, transforms, partition_axis=axis
        )
    overlap_fraction, uncovered_fraction, cropped_fraction, loss_fraction = (
        _candidate_loss_fractions(candidate, crop.size)
    )
    mean_before = float(np.mean(befores))
    mean_after = float(np.mean(afters))
    raw_gain = mean_after - mean_before
    seam_count = len(seams)
    complexity_penalty = 0.015 * seam_count
    loss_penalty = 0.35 * loss_fraction
    total_gain = raw_gain - complexity_penalty - loss_penalty
    if total_gain <= 0.035:
        return None
    return {
        "mean_before": mean_before,
        "mean_after": mean_after,
        "raw_gain": raw_gain,
        "complexity_penalty": complexity_penalty,
        "loss_fraction": loss_fraction,
        "loss_penalty": loss_penalty,
        "overlap_fraction": overlap_fraction,
        "uncovered_fraction": uncovered_fraction,
        "cropped_source_fraction": cropped_fraction,
        "fragment_count": float(seam_count + 1),
        "seam_count": float(seam_count),
        "total_gain": total_gain,
    }


def _endpoint_probe_center(
    measurements: _RailMeasurements,
    rail_index: int,
    row_start: int,
    row_end: int,
) -> float | None:
    """Spec 12.2 steps 3-4: the rail's median occupied coordinate within the
    fixed probe, or None when the track is not present there.

    Presence uses the frozen 70% row-occupancy convention from
    ``trace_centers`` — the oriented opening jitters component extents by a
    pixel or two, so exact-coverage tests would reject genuine endpoints.
    Each row's pixels are attributed to this track only when their run center
    lies within ``center_tolerance`` of the track's aggregate center, so a
    displaced neighbor inside the window is never double-counted."""
    center, _, _ = measurements.rails[rail_index]
    width = measurements.rail_mask.shape[1]
    low = max(0, int(np.floor(center)) - measurements.max_rail_width)
    high = min(width, int(np.ceil(center)) + measurements.max_rail_width + 1)
    window = measurements.rail_mask[row_start:row_end, low:high] > 0
    occupied_rows = 0
    track_columns: list[np.ndarray] = []
    for row in window:
        columns = np.flatnonzero(row)
        if columns.size == 0:
            continue
        runs = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        nearest = min(
            runs, key=lambda run: abs(float(run.mean()) + low - center)
        )
        if abs(float(nearest.mean()) + low - center) > (
            measurements.center_tolerance
        ):
            continue
        occupied_rows += 1
        track_columns.append(nearest)
    if occupied_rows < (row_end - row_start) * 0.7:
        return None
    return float(np.median(np.concatenate(track_columns))) + low


def _strong_endpoints(
    measurements: _RailMeasurements,
    row_start: int,
    row_end: int,
    *,
    cross_interval: tuple[int, int],
) -> list[tuple[float, int]]:
    """Sorted (probe center, rail id) for tracks strong over one probe."""
    height = measurements.canonical.shape[0]
    if row_start < 0 or row_end > height:
        return []
    cross_low, cross_high = cross_interval
    endpoints: list[tuple[float, int]] = []
    for index, (center, _, _) in enumerate(measurements.rails):
        if not cross_low <= center < cross_high:
            continue
        probed = _endpoint_probe_center(measurements, index, row_start, row_end)
        if probed is not None:
            endpoints.append((probed, index))
    return sorted(endpoints)


def _pair_endpoints(
    before: list[tuple[float, int]],
    after: list[tuple[float, int]],
    *,
    offset_delta: float,
    tolerance: float,
) -> list[tuple[int, int]]:
    """Spec 12.2 step 5: monotone one-to-one in-tolerance pairing maximizing
    pair count, then minimizing total absolute residual, then choosing the
    lexicographically smaller rail-ID tuple.  Returns index pairs into the
    sorted inputs."""
    best: dict[tuple[int, int], tuple[int, float, tuple[int, ...], tuple]] = {}

    def solve(i: int, j: int) -> tuple[int, float, tuple[int, ...], tuple]:
        if i >= len(before) or j >= len(after):
            return (0, 0.0, (), ())
        key = (i, j)
        if key in best:
            return best[key]
        candidates = [solve(i + 1, j), solve(i, j + 1)]
        residual = abs(before[i][0] - after[j][0] + offset_delta)
        if residual <= tolerance:
            count, total, ids, pairs = solve(i + 1, j + 1)
            candidates.append(
                (
                    count + 1,
                    total + residual,
                    (before[i][1], after[j][1], *ids),
                    ((i, j), *pairs),
                )
            )
        result = min(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        )
        best[key] = result
        return result

    return list(solve(0, 0)[3])


def _endpoint_mask_presence(
    measurements: _RailMeasurements,
    position: float,
    row_start: int,
    row_end: int,
) -> bool:
    """Spec 12.4: does the source rail mask show a strong endpoint near this
    position within the fixed probe?"""
    height = measurements.canonical.shape[0]
    if row_start < 0 or row_end > height:
        return False
    width = measurements.rail_mask.shape[1]
    halfwidth = measurements.max_rail_width + measurements.center_tolerance
    low = max(0, int(np.floor(position - halfwidth)))
    high = min(width, int(np.ceil(position + halfwidth)) + 1)
    if low >= high:
        return False
    window = measurements.rail_mask[row_start:row_end, low:high] > 0
    return int(window.any(axis=1).sum()) >= (row_end - row_start) * 0.7


def _measure_endpoint_residuals(
    source_gray: np.ndarray,
    reconstructed: np.ndarray,
    *,
    region: tuple[int, int, int, int],
    axis: Axis,
    transforms: tuple[FragmentTransform, ...],
    measurements: _RailMeasurements,
) -> _EndpointReport:
    """Spec 12.2 frozen endpoint-measurement contract for one proposed
    regional reconstruction.

    All probes read the frozen source-side rail measurements; the proposed
    transforms are applied analytically to centers (internal residuals) and
    against the untouched outside probes (crop edges).  ``source_gray`` and
    ``reconstructed`` pin the contract's inputs; measurement itself relies on
    ``measurements`` (built from the source by the frozen implementation)."""
    del source_gray, reconstructed  # probes come from the frozen measurements
    x0, y0, x1, y1 = region
    if axis == "y":
        part_interval, cross_interval = (y0, y1), (x0, x1)
    else:
        part_interval, cross_interval = (x0, x1), (y0, y1)
    part_low, part_high = part_interval
    cross_low, cross_high = cross_interval
    height = measurements.canonical.shape[0]
    trace_depth = max(8, min(20, measurements.rail_length))
    tolerance = float(measurements.center_tolerance)

    def fragment_offset(transform: FragmentTransform) -> int:
        return transform.inverse_dx if axis == "y" else transform.inverse_dy

    ordered = sorted(transforms, key=lambda item: item.interval)
    internal: list[_EndpointResidual] = []
    crop_edges: list[_EndpointResidual] = []
    unmatched_strong = 0
    absent = 0
    clipped_ids: set[int] = set()

    def clip_check(endpoints: list[tuple[float, int]], offset: int) -> None:
        for center, rail_id in endpoints:
            destination = center + offset
            if not cross_low <= destination < cross_high:
                clipped_ids.add(rail_id)

    def measure_boundary(
        boundary: int,
        before_offset: int,
        after_offset: int,
        *,
        sink: list[_EndpointResidual],
        before_inside: bool,
        after_inside: bool,
    ) -> None:
        nonlocal unmatched_strong, absent
        before = _strong_endpoints(
            measurements,
            boundary - trace_depth,
            boundary,
            cross_interval=cross_interval,
        )
        after = _strong_endpoints(
            measurements,
            boundary,
            boundary + trace_depth,
            cross_interval=cross_interval,
        )
        if before_inside:
            clip_check(before, before_offset)
        if after_inside:
            clip_check(after, after_offset)
        delta = float(before_offset - after_offset)
        pairs = _pair_endpoints(
            before, after, offset_delta=delta, tolerance=tolerance
        )
        matched_before = {i for i, _ in pairs}
        matched_after = {j for _, j in pairs}
        for i, j in pairs:
            residual = abs(before[i][0] - after[j][0] + delta)
            sink.append(
                _EndpointResidual(
                    boundary=boundary,
                    before_rail_id=before[i][1],
                    after_rail_id=after[j][1],
                    before_center=before[i][0],
                    after_center=after[j][0],
                    residual_quarter_px=int(round(4 * residual)),
                )
            )
        for index, (center, _) in enumerate(before):
            if index in matched_before:
                continue
            # The mate's source position puts both destinations together:
            # mate + after_offset == center + before_offset.
            if _endpoint_mask_presence(
                measurements, center + delta, boundary, boundary + trace_depth
            ):
                unmatched_strong += 1
            else:
                absent += 1
        for index, (center, _) in enumerate(after):
            if index in matched_after:
                continue
            if _endpoint_mask_presence(
                measurements, center - delta, boundary - trace_depth, boundary
            ):
                unmatched_strong += 1
            else:
                absent += 1

    for first, second in zip(ordered, ordered[1:]):
        measure_boundary(
            first.interval[1] + part_low,
            fragment_offset(first),
            fragment_offset(second),
            sink=internal,
            before_inside=True,
            after_inside=True,
        )
    if part_low > 0:
        measure_boundary(
            part_low,
            0,
            fragment_offset(ordered[0]),
            sink=crop_edges,
            before_inside=False,
            after_inside=True,
        )
    if part_high < height:
        measure_boundary(
            part_high,
            fragment_offset(ordered[-1]),
            0,
            sink=crop_edges,
            before_inside=True,
            after_inside=False,
        )
    return _EndpointReport(
        internal=tuple(internal),
        crop_edges=tuple(crop_edges),
        unmatched_strong_source=unmatched_strong,
        source_endpoint_absent=absent,
        clipped_strong_rails=len(clipped_ids),
    )


def _interior_uncovered_pixels(
    candidate: RepairCandidate,
    transforms: tuple[FragmentTransform, ...],
    axis: Axis,
) -> int:
    """Spec 9.2: uncovered pixels excluding components hugging the outer edge
    a non-zero fragment vacated (content translated away from that edge)."""
    uncovered = candidate.uncovered_mask.astype(np.uint8)
    if not uncovered.any():
        return 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered, 8)
    height, width = uncovered.shape
    interior = 0
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(v) for v in stats[index])
        if axis == "y":
            part_range = (y, y + box_height)
            touches_low = x == 0
            touches_high = x + box_width == width
        else:
            part_range = (x, x + box_width)
            touches_low = y == 0
            touches_high = y + box_height == height
        overlapping = [
            (transform.inverse_dx if axis == "y" else transform.inverse_dy)
            for transform in transforms
            if transform.interval[0] < part_range[1]
            and transform.interval[1] > part_range[0]
        ]
        vacated_low = touches_low and any(offset > 0 for offset in overlapping)
        vacated_high = touches_high and any(offset < 0 for offset in overlapping)
        if not (vacated_low or vacated_high):
            interior += area
    return interior


def _select_gauge(
    gray: np.ndarray,
    component: tuple[_AnchoredSeam, ...],
    *,
    region: tuple[int, int, int, int],
    axis: Axis,
    measurements: _RailMeasurements,
) -> _GaugeSelection:
    """Spec 9.2: reconstruct every distinct cumulative offset as the common
    translation gauge and pick the frozen minimum lexicographic tuple."""
    seams = tuple(sorted(component, key=lambda item: item.seam.boundary))
    x0, y0, x1, y1 = region
    if axis == "y":
        part_interval, cross_interval = (y0, y1), (x0, x1)
    else:
        part_interval, cross_interval = (x0, x1), (y0, y1)
    part_low, part_high = part_interval
    region_len = part_high - part_low
    crop = gray[y0:y1, x0:x1]
    local_seams = tuple(
        replace(item.seam, boundary=item.seam.boundary - part_low)
        for item in seams
    )
    edges = (0, *(seam.boundary for seam in local_seams), region_len)
    cumulative = [0]
    for seam in local_seams:
        cumulative.append(cumulative[-1] + seam.relative_offset)
    fragment_lengths = [end - start for start, end in zip(edges, edges[1:])]
    sentinel = 4 * (measurements.center_tolerance + 1)
    trace_depth = max(8, min(20, measurements.rail_length))
    height = measurements.canonical.shape[0]

    source_masks = build_content_masks(crop)
    source_content = source_masks.glyph_mask | source_masks.structure_mask

    best: tuple | None = None
    evaluated: list[dict[str, float]] = []
    for gauge in sorted(set(cumulative)):
        anchor_index = cumulative.index(gauge)
        shifts = [offset - gauge for offset in cumulative]
        transforms = tuple(
            FragmentTransform(
                (start, end),
                inverse_dx=shift if axis == "y" else 0,
                inverse_dy=shift if axis == "x" else 0,
            )
            for (start, end), shift in zip(zip(edges, edges[1:]), shifts)
        )
        candidate = replace(
            apply_fragment_transforms(crop, transforms, partition_axis=axis),
            region=region,
        )
        destination_lost = candidate.overlap_mask | candidate.uncovered_mask
        destination_masks = build_content_masks(candidate.reconstruction)
        destination_content = (
            destination_masks.glyph_mask | destination_masks.structure_mask
        )
        non_rule_content_loss = int(
            (candidate.cropped_source_mask & source_content).sum()
        ) + int((destination_lost & destination_content).sum())
        interior_uncovered = _interior_uncovered_pixels(
            candidate, transforms, axis
        )
        overlap_pixels = int(candidate.overlap_mask.sum())
        report = _measure_endpoint_residuals(
            crop,
            candidate.reconstruction,
            region=region,
            axis=axis,
            transforms=transforms,
            measurements=measurements,
        )
        edge_residuals = [
            record.residual_quarter_px for record in report.crop_edges
        ]
        matched_by_boundary: dict[int, int] = {}
        for record in report.crop_edges:
            matched_by_boundary[record.boundary] = (
                matched_by_boundary.get(record.boundary, 0) + 1
            )
        mateless = 0
        for boundary, probe in (
            (part_low, (part_low, part_low + trace_depth)),
            (part_high, (part_high - trace_depth, part_high)),
        ):
            if boundary <= 0 or boundary >= height:
                continue
            strong_inside = len(
                _strong_endpoints(
                    measurements, probe[0], probe[1], cross_interval=cross_interval
                )
            )
            mateless += max(
                0, strong_inside - matched_by_boundary.get(boundary, 0)
            )
        if report.clipped_strong_rails or mateless:
            edge_residuals.append(sentinel)
        max_edge_residual = max(edge_residuals, default=0)
        total_mapping_loss = max(
            int(destination_lost.sum()), int(candidate.cropped_source_mask.sum())
        )
        objective = (
            non_rule_content_loss,
            interior_uncovered,
            overlap_pixels,
            max_edge_residual,
            total_mapping_loss,
            abs(gauge),
            anchor_index,
        )
        predicted_loss = sum(
            abs(shift) * length
            for shift, length in zip(shifts, fragment_lengths)
        )
        evaluated.append(
            {
                "gauge_offset": float(gauge),
                "anchor_fragment_index": float(anchor_index),
                "predicted_loss_pixels": float(predicted_loss),
                "measured_loss_pixels": float(total_mapping_loss),
                "non_rule_content_loss_pixels": float(non_rule_content_loss),
                "interior_uncovered_pixels": float(interior_uncovered),
                "overlap_pixels": float(overlap_pixels),
                "max_crop_edge_rail_residual_quarter_px": float(
                    max_edge_residual
                ),
            }
        )
        entry = (objective, gauge, anchor_index, candidate)
        if best is None or entry[0] < best[0]:
            best = entry
    assert best is not None
    return _GaugeSelection(
        gauge_offset=int(best[1]),
        anchor_fragment_index=int(best[2]),
        objective_tuple=best[0],
        evaluated=tuple(evaluated),
        candidate=best[3],
    )


def _joint_score_terms(
    joint: dict[str, float], evidence: _AnchoredSeam
) -> dict[str, float]:
    """Legacy-shaped rule-anchor score terms with the spec 10 joint values.

    Continuity/gain/loss/total come from the joint scorer; the per-event
    diagnostic terms keep the legacy convention of describing the strongest
    (max-objective) seam, so a one-seam chain stays byte-identical to the
    pre-correction scoring."""
    return {
        "method_rule_anchor": 1.0,
        "rail_endpoint_score": evidence.objective,
        "non_rule_continuity": joint["mean_after"],
        "non_rule_continuity_before": joint["mean_before"],
        "non_rule_continuity_after": joint["mean_after"],
        "non_rule_continuity_gain": joint["raw_gain"],
        "periodic_alias_margin": evidence.alias_margin,
        "matched_rail_count": float(evidence.matched_rail_count),
        "matched_rail_span_fraction": evidence.objective
        / max(float(evidence.matched_rail_count), 1.0),
        "non_rule_support": evidence.non_rule_support,
        "observed_relative_offset": float(-evidence.seam.relative_offset),
        "complexity_penalty": joint["complexity_penalty"],
        "loss_fraction": joint["loss_fraction"],
        "loss_penalty": joint["loss_penalty"],
        "fragment_count": joint["fragment_count"],
        "total_gain": joint["total_gain"],
    }


def _component_diagnostics_record(
    component: tuple[_AnchoredSeam, ...],
    joint: dict[str, float],
    *,
    measurements: _RailMeasurements,
    region: tuple[int, int, int, int],
    axis: Axis,
    support_floor: float,
) -> dict:
    """Spec 10.3 per-seam records for the diagnostics sidecar (never
    score_terms)."""
    x0, y0, x1, y1 = region
    if axis == "y":
        part_interval, cross_interval = (y0, y1), (x0, x1)
    else:
        part_interval, cross_interval = (x0, x1), (y0, y1)
    per_seam = []
    for item in sorted(component, key=lambda value: value.seam.boundary):
        before = _joint_seam_continuity(
            measurements,
            item.seam.boundary,
            0,
            part_interval=part_interval,
            cross_interval=cross_interval,
        )
        after = _joint_seam_continuity(
            measurements,
            item.seam.boundary,
            item.seam.relative_offset,
            part_interval=part_interval,
            cross_interval=cross_interval,
        )
        per_seam.append(
            {
                "boundary": int(item.seam.boundary),
                "source": item.source,
                "relative_offset": int(item.seam.relative_offset),
                "identity_score": before,
                "repaired_score": after,
                "gain": after - before,
                "matched_rail_count": int(item.matched_rail_count),
                "matched_center_interval": list(_rail_seam_cross_interval(item))
                if item.matched_pairs
                else None,
                "alias_margin": float(item.alias_margin),
                "support_interval": list(item.support_interval),
            }
        )
    return {
        "family": "method_rule_anchor",
        "region": list(region),
        "partition_axis": axis,
        "support_floor": support_floor,
        "per_seam": per_seam,
        "joint": dict(joint),
    }


def _trace_rail_tracks(
    rail_mask: np.ndarray,
    rails: tuple[tuple[float, int, int], ...],
    *,
    max_rail_width: int,
) -> tuple[_RailTrack, ...]:
    """Per-row traced centers for every measured rail (spec 8.4 rule 4
    witnesses), sampled from the frozen rail mask in a window around each
    rail's aggregate center; NaN where a row has no rail pixels."""
    columns = np.arange(rail_mask.shape[1], dtype=np.float64)
    tracks: list[_RailTrack] = []
    for center, start, end in rails:
        low = max(0, int(np.floor(center)) - max_rail_width)
        high = min(rail_mask.shape[1], int(np.ceil(center)) + max_rail_width + 1)
        window = rail_mask[start:end, low:high] > 0
        counts = window.sum(axis=1)
        weighted = window @ columns[low:high]
        row_centers = np.where(
            counts > 0, weighted / np.maximum(counts, 1), np.nan
        )
        tracks.append(
            _RailTrack(
                center=float(center),
                extent=(int(start), int(end)),
                row_centers=tuple(float(value) for value in row_centers),
            )
        )
    return tuple(tracks)


def _compose_candidates(
    gray: np.ndarray,
    first: RepairCandidate,
    second: RepairCandidate,
) -> RepairCandidate | None:
    """Spec 6 / test 15: compose two compatible partial mappings into one
    explicit reconstruction, or refuse.

    Composable only when the partials share the partition axis and cross
    extent, their partition intervals overlap or abut, and their displacement
    functions agree on all shared support (a 1-2px disagreement is tolerated
    only over content-free pixels; beyond 2px is a conflict).  The combined
    mapping uses the first candidate's displacement on shared intervals."""
    if first.partition_axis != second.partition_axis:
        return None
    axis = first.partition_axis
    fx0, fy0, fx1, fy1 = first.region
    sx0, sy0, sx1, sy1 = second.region
    if axis == "y":
        first_cross, second_cross = (fx0, fx1), (sx0, sx1)
        first_part, second_part = (fy0, fy1), (sy0, sy1)
    else:
        first_cross, second_cross = (fy0, fy1), (sy0, sy1)
        first_part, second_part = (fx0, fx1), (sx0, sx1)
    if first_cross != second_cross:
        return None
    if max(first_part[0], second_part[0]) > min(first_part[1], second_part[1]):
        return None

    def page_spans(
        candidate: RepairCandidate, part: tuple[int, int]
    ) -> list[tuple[int, int, int]]:
        return [
            (
                part[0] + fragment.interval[0],
                part[0] + fragment.interval[1],
                fragment.inverse_dx if axis == "y" else fragment.inverse_dy,
            )
            for fragment in candidate.fragments
        ]

    first_spans = page_spans(first, first_part)
    second_spans = page_spans(second, second_part)
    for a_start, a_end, a_shift in first_spans:
        for b_start, b_end, b_shift in second_spans:
            shared = (max(a_start, b_start), min(a_end, b_end))
            if shared[0] >= shared[1] or a_shift == b_shift:
                continue
            if abs(a_shift - b_shift) > 2:
                return None
            band = (
                gray[shared[0] : shared[1], first_cross[0] : first_cross[1]]
                if axis == "y"
                else gray[first_cross[0] : first_cross[1], shared[0] : shared[1]]
            )
            if (band < 180).any():
                return None

    low = min(first_part[0], second_part[0])
    high = max(first_part[1], second_part[1])
    breakpoints = sorted(
        {value for span in (*first_spans, *second_spans) for value in span[:2]}
    )
    merged: list[tuple[int, int, int]] = []
    for start, end in zip(breakpoints, breakpoints[1:]):
        shift = next(
            (
                span_shift
                for span_start, span_end, span_shift in (*first_spans, *second_spans)
                if span_start <= start and span_end >= end
            ),
            None,
        )
        if shift is None:
            return None
        if merged and merged[-1][1] == start and merged[-1][2] == shift:
            merged[-1] = (merged[-1][0], end, shift)
        else:
            merged.append((start, end, shift))
    if axis == "y":
        region = (first_cross[0], low, first_cross[1], high)
        crop = gray[low:high, first_cross[0] : first_cross[1]]
    else:
        region = (low, first_cross[0], high, first_cross[1])
        crop = gray[first_cross[0] : first_cross[1], low:high]
    transforms = tuple(
        FragmentTransform(
            (start - low, end - low),
            inverse_dx=shift if axis == "y" else 0,
            inverse_dy=shift if axis == "x" else 0,
        )
        for start, end, shift in merged
    )
    return replace(
        apply_fragment_transforms(crop, transforms, partition_axis=axis),
        region=region,
    )


def _pitch_edge_residual(
    measurements: _RailMeasurements,
    boundary: int,
    *,
    inside_before: bool,
    inside_offset: int,
    cross_interval: tuple[int, int],
    trace_depth: int,
) -> int | None:
    """Spec 9.2 crop-edge rail residual at one rejoin boundary for a given
    total inside offset; None when no strong inside endpoint is measurable."""
    inside_rows = (
        (boundary - trace_depth, boundary)
        if inside_before
        else (boundary, boundary + trace_depth)
    )
    outside_rows = (
        (boundary, boundary + trace_depth)
        if inside_before
        else (boundary - trace_depth, boundary)
    )
    inside = _strong_endpoints(
        measurements, inside_rows[0], inside_rows[1], cross_interval=cross_interval
    )
    if not inside:
        return None
    outside = _strong_endpoints(
        measurements, outside_rows[0], outside_rows[1], cross_interval=cross_interval
    )
    sentinel = 4 * (measurements.center_tolerance + 1)
    delta = float(inside_offset) if inside_before else -float(inside_offset)
    pairs = _pair_endpoints(
        inside if inside_before else outside,
        outside if inside_before else inside,
        offset_delta=delta,
        tolerance=float(measurements.center_tolerance),
    )
    matched_inside = {pair[0] if inside_before else pair[1] for pair in pairs}
    residuals = []
    before_list = inside if inside_before else outside
    after_list = outside if inside_before else inside
    for i, j in pairs:
        residuals.append(
            int(round(4 * abs(before_list[i][0] - after_list[j][0] + delta)))
        )
    if len(matched_inside) < len(inside):
        residuals.append(sentinel)
    cross_low, cross_high = cross_interval
    for center, _ in inside:
        if not cross_low <= center + inside_offset < cross_high:
            residuals.append(sentinel)
            break
    return max(residuals, default=sentinel)


def _refine_pitch_offsets(
    gray: np.ndarray,
    pitch_candidate: RepairCandidate,
    *,
    axis: Axis,
    measurements: _RailMeasurements,
) -> _PitchRefinement | None:
    """Spec 11: bounded residual refinement of an emitted pitch candidate.

    Every strip's delta in [-8, 8] is evaluated exhaustively with vectorized
    accumulation — the exact equivalent of the section 11.2 dynamic program
    for the family's <=5 strips, and unlike a chain DP it applies the
    path-level primary/loss/gain gates exactly.  ``gray`` is unused directly:
    the candidate's own source view and the frozen page measurements carry
    every measured quantity."""
    del gray
    terms = pitch_candidate.score_terms
    pitch = int(terms.get("row_pitch", 0))
    if pitch < 17:
        return None
    if abs(pitch_candidate.partition_angle_degrees) >= 0.75:
        # The candidate was solved on a rectified view whose provenance
        # cannot be recomposed here; leave it unrefined.
        return None
    strips = pitch_candidate.fragments
    strip_count = len(strips)
    if not 2 <= strip_count <= 5:
        return None
    source = pitch_candidate.source_view
    offsets = [
        fragment.inverse_dx if axis == "y" else fragment.inverse_dy
        for fragment in strips
    ]
    evidence = _alignment_evidence(source, build_content_masks(source))
    canonical_evidence = evidence if axis == "x" else evidence.T
    repair_size, partition_size = canonical_evidence.shape
    phase = int(terms.get("row_phase", 0))
    target = _comb_target(repair_size, pitch, phase)
    bin_width = max(2, pitch // 4)
    deltas = list(range(-8, 9))
    zero_index = 8

    primary = np.zeros((strip_count, 17))
    informative_total = 0
    for index, fragment in enumerate(strips):
        start, end = fragment.interval
        for bin_start in range(start, end, bin_width):
            profile = canonical_evidence[
                :, bin_start : min(bin_start + bin_width, end)
            ].sum(axis=1)
            row = np.asarray(
                [
                    _profile_shift_score(profile, target, offsets[index] + delta)
                    for delta in deltas
                ]
            )
            if float(row.max()) >= 0.15:
                primary[index] += row
                informative_total += 1
    if informative_total == 0:
        return None

    x0, y0, x1, y1 = pitch_candidate.region
    part_interval = (y0, y1) if axis == "y" else (x0, x1)
    cross_interval = (x0, x1) if axis == "y" else (y0, y1)
    part_low, part_high = part_interval
    strip_lengths = [f.interval[1] - f.interval[0] for f in strips]
    area = float(max(source.size, 1))

    continuity: list[dict[int, float]] = []
    for index in range(strip_count - 1):
        boundary = strips[index].interval[1] + part_low
        base = offsets[index + 1] - offsets[index]
        values: dict[int, float] = {}
        for relative in range(base - 16, base + 17):
            values[relative] = _joint_seam_continuity(
                measurements,
                boundary,
                relative,
                part_interval=part_interval,
                cross_interval=cross_interval,
            )
        continuity.append(values)

    trace_depth = max(8, min(20, measurements.rail_length))
    page_extent = measurements.canonical.shape[0]
    sentinel = 4 * (measurements.center_tolerance + 1)
    first_edge = [None] * 17
    last_edge = [None] * 17
    if part_low > 0:
        first_edge = [
            _pitch_edge_residual(
                measurements,
                part_low,
                inside_before=False,
                inside_offset=offsets[0] + delta,
                cross_interval=cross_interval,
                trace_depth=trace_depth,
            )
            for delta in deltas
        ]
    if part_high < page_extent:
        last_edge = [
            _pitch_edge_residual(
                measurements,
                part_high,
                inside_before=True,
                inside_offset=offsets[-1] + delta,
                cross_interval=cross_interval,
                trace_depth=trace_depth,
            )
            for delta in deltas
        ]

    shape = (17,) * strip_count

    def unary(values: np.ndarray, index: int) -> np.ndarray:
        dims = [1] * strip_count
        dims[index] = 17
        return values.reshape(dims)

    primary_path = np.zeros(shape)
    loss_pixels = np.zeros(shape)
    sum_abs_deltas = np.zeros(shape)
    for index in range(strip_count):
        primary_path = primary_path + unary(primary[index], index)
        magnitude = np.asarray(
            [abs(offsets[index] + delta) * strip_lengths[index] for delta in deltas],
            dtype=np.float64,
        )
        loss_pixels = loss_pixels + unary(magnitude, index)
        sum_abs_deltas = sum_abs_deltas + unary(
            np.abs(np.asarray(deltas, dtype=np.float64)), index
        )
    continuity_path = np.zeros(shape)
    valid = np.ones(shape, dtype=bool)
    original_continuity = []
    for index, values in enumerate(continuity):
        base = offsets[index + 1] - offsets[index]
        matrix = np.asarray(
            [
                [values[base + db - da] for db in deltas]
                for da in deltas
            ]
        )
        original_continuity.append(values[base])
        dims = [1] * strip_count
        dims[index] = 17
        dims[index + 1] = 17
        continuity_path = continuity_path + matrix.reshape(dims)
        valid &= (matrix >= values[base] - 0.01).reshape(dims)

    def edge_quality_values(residuals: list) -> np.ndarray | None:
        if all(value is None for value in residuals):
            return None
        return np.asarray(
            [
                max(0.0, 1.0 - (sentinel if value is None else value) / sentinel)
                for value in residuals
            ]
        )

    first_quality = edge_quality_values(first_edge)
    last_quality = edge_quality_values(last_edge)
    measurable_edges = sum(1 for q in (first_quality, last_quality) if q is not None)
    edge_quality = np.zeros(shape)
    if measurable_edges:
        if first_quality is not None:
            edge_quality = edge_quality + unary(first_quality, 0)
        if last_quality is not None:
            edge_quality = edge_quality + unary(last_quality, strip_count - 1)
        edge_quality = edge_quality / measurable_edges
    max_edge_residual = np.zeros(shape, dtype=np.int64)
    for residuals, position in ((first_edge, 0), (last_edge, strip_count - 1)):
        values = np.asarray(
            [0 if value is None else value for value in residuals], dtype=np.int64
        )
        max_edge_residual = np.maximum(max_edge_residual, unary(values, position))

    mean_continuity = continuity_path / (strip_count - 1)
    loss_fraction = loss_pixels / area
    mean_delta = sum_abs_deltas / (8.0 * strip_count)
    refinement_score = (
        mean_continuity
        + 0.10 * edge_quality
        - 0.35 * loss_fraction
        - 0.002 * mean_delta
    )
    primary_mean = primary_path / informative_total

    original_index = (zero_index,) * strip_count
    original_primary = float(primary_mean[original_index])
    original_loss = float(loss_fraction[original_index])
    original_mean_continuity = float(np.mean(original_continuity))

    primary_eligible = primary_mean >= original_primary - 0.005
    eligible = (
        valid
        & primary_eligible
        & (loss_fraction <= original_loss)
        & (mean_continuity >= original_mean_continuity + 0.03)
    )
    if not eligible.any():
        return None

    def tie_tuple(index_tuple: tuple[int, ...]) -> tuple:
        delta_tuple = tuple(deltas[i] for i in index_tuple)
        return (
            int(max_edge_residual[index_tuple]),
            float(loss_fraction[index_tuple]),
            sum(abs(d) for d in delta_tuple),
            delta_tuple,
        )

    masked = np.where(eligible, refinement_score, -np.inf)
    best_score = float(masked.max())
    tied = np.argwhere(masked >= best_score - 1e-9)
    best_index = min((tuple(int(v) for v in row) for row in tied), key=tie_tuple)
    best_deltas = tuple(deltas[i] for i in best_index)

    # Margin over every distinct eligible path (spec 11.3; "after primary
    # eligibility is applied" — the pool is the gated set, and the runner-up
    # is its best-scoring distinct member).
    margin_pool = np.where(eligible, refinement_score, -np.inf)
    margin_pool[best_index] = -np.inf
    runner_index_flat = int(np.argmax(margin_pool))
    runner_index = np.unravel_index(runner_index_flat, shape)
    runner_score = float(margin_pool[runner_index])
    runner_deltas = tuple(deltas[i] for i in runner_index)
    if runner_score > -np.inf and best_score - runner_score < 0.03:
        return None

    refined_transforms = tuple(
        FragmentTransform(
            fragment.interval,
            inverse_dx=fragment.inverse_dx + (delta if axis == "y" else 0),
            inverse_dy=fragment.inverse_dy + (delta if axis == "x" else 0),
        )
        for fragment, delta in zip(strips, best_deltas)
    )
    rebuilt = apply_fragment_transforms(
        source, refined_transforms, partition_axis=axis
    )
    _, _, _, refined_loss = _candidate_loss_fractions(rebuilt, source.size)
    complexity = 0.015 * (strip_count - 1)
    primary_best = float(primary_mean[best_index])
    row_comb_after = float(
        terms.get("row_comb_after", 0.0) + (primary_best - original_primary)
    )
    row_comb_gain = row_comb_after - float(terms.get("row_comb_before", 0.0))
    refined_candidate = replace(
        rebuilt,
        region=pitch_candidate.region,
        partition_angle_degrees=pitch_candidate.partition_angle_degrees,
        pixel_score_before=pitch_candidate.pixel_score_before,
        pixel_score_after=row_comb_after,
        score_terms={
            **terms,
            "pitch_refined": 1.0,
            "refinement_score": float(refinement_score[best_index]),
            "refinement_primary_before": original_primary,
            "refinement_primary_after": primary_best,
            "refinement_continuity_gain": float(
                mean_continuity[best_index] - original_mean_continuity
            ),
            "row_comb_after": row_comb_after,
            "row_comb_gain": row_comb_gain,
            "loss_fraction": refined_loss,
            "loss_penalty": 0.35 * refined_loss,
            "total_gain": row_comb_gain - complexity - 0.35 * refined_loss,
        },
    )
    return _PitchRefinement(
        refined=refined_candidate,
        unrefined=pitch_candidate,
        deltas=best_deltas,
        refinement_score=float(refinement_score[best_index]),
        runner_up=(runner_deltas, runner_score),
    )


# Per-seam component records keyed by geometry signature.  The page entry
# clears this before candidate generation and drains it into the diagnostics
# sidecar afterwards; the indirection preserves the monkeypatch-pinned
# `_rule_anchor_candidates(gray, *, max_fragments) -> list` boundary.
_COMPONENT_DIAGNOSTICS: dict[str, dict] = {}


_FAMILY_MODES = ("all_family_diagnostic", "active_default")

# Report schema version: 2 activates the spec 8.5 deterministic cross-family
# retention and the additive per-candidate retention fields.  Stage A1 runs
# predate this constant and correspond to version 1.
_SCHEMA_VERSION = 2

# Spec 8.5 fixed family order (geometric support, strongest first).  The
# bounded two-rail family participates in all_family_diagnostic mode only.
_FAMILY_ORDER = (
    "method_rule_anchor_component",
    "method_rule_anchor",
    "method_pitch",
    "method_seam",
    "method_two_rail_bounded_local",
)


def _solver_family(candidate: RepairCandidate) -> str:
    """Spec 8.1/8.5 family resolution: the component marker outranks the
    broader rule-anchor marker; unmarked candidates are the seam family."""
    for marker in (
        "method_rule_anchor_component",
        "method_rule_anchor",
        "method_pitch",
        "method_two_rail_bounded_local",
    ):
        if candidate.score_terms.get(marker) == 1.0:
            return marker
    return "method_seam"


def _retain_by_family(
    candidates: list[RepairCandidate], *, top_k: int, family_mode: str
) -> list[tuple[RepairCandidate, str, int, int, int]]:
    """Spec 8.5 deterministic cross-family retention.

    Returns (candidate, solver_family, family_rank, retention_round,
    retention_slot) in retention order: round-robin over viable families in
    the fixed family order, taking every family's next-ranked candidate
    before any family's following one, until ``top_k`` is filled."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if family_mode not in _FAMILY_MODES:
        raise ValueError(f"family_mode must be one of {_FAMILY_MODES}")
    families = tuple(
        family
        for family in _FAMILY_ORDER
        if family != "method_two_rail_bounded_local"
        or family_mode == "all_family_diagnostic"
    )
    grouped: dict[str, list[RepairCandidate]] = {family: [] for family in families}
    for candidate in candidates:
        family = _solver_family(candidate)
        if family in grouped:
            grouped[family].append(candidate)
    for members in grouped.values():
        members.sort(
            key=lambda candidate: (
                -candidate.score_terms["total_gain"],
                candidate.region,
                candidate.partition_axis,
                _candidate_geometry_signature(candidate),
            )
        )
    viable = [family for family in families if grouped[family]]
    retained: list[tuple[RepairCandidate, str, int, int, int]] = []
    round_index = 0
    while len(retained) < top_k:
        round_index += 1
        progressed = False
        for family in viable:
            if len(retained) >= top_k:
                break
            members = grouped[family]
            if len(members) >= round_index:
                retained.append(
                    (
                        members[round_index - 1],
                        family,
                        round_index,
                        round_index,
                        len(retained) + 1,
                    )
                )
                progressed = True
        if not progressed:
            break
    return retained


def _candidate_geometry_signature(candidate: RepairCandidate) -> tuple:
    """Complete geometry identity: region, axis, and every fragment transform."""
    return (
        candidate.region,
        candidate.partition_axis,
        tuple(
            (
                fragment.interval,
                fragment.inverse_dx,
                fragment.inverse_dy,
            )
            for fragment in candidate.fragments
        ),
    )


def _geometry_signature_key(candidate: RepairCandidate) -> str:
    """Serializable form of the geometry signature for diagnostics sidecars."""
    return repr(_candidate_geometry_signature(candidate))


def _rule_anchor_candidates(
    gray: np.ndarray, *, max_fragments: int
) -> list[RepairCandidate]:
    """Offer full-page rail-anchor candidates for both partition axes."""
    return _rule_anchor_family_candidates(
        gray, max_fragments=max_fragments, family_mode="all_family_diagnostic"
    )


def _rule_anchor_family_candidates(
    gray: np.ndarray, *, max_fragments: int, family_mode: str
) -> list[RepairCandidate]:
    """Rule-anchor families; active_default skips the bounded two-rail cost."""
    strong = _rule_anchor_candidates_for_axis(
        gray, axis="y", max_fragments=max_fragments
    ) + _rule_anchor_candidates_for_axis(gray, axis="x", max_fragments=max_fragments)
    bounded: list[RepairCandidate] = []
    if family_mode != "active_default":
        bounded = _bounded_two_rail_candidates_for_axis(
            gray, axis="y"
        ) + _bounded_two_rail_candidates_for_axis(gray, axis="x")
    distinct: dict[tuple, RepairCandidate] = {}
    for candidate in (*strong, *bounded):
        signature = _candidate_geometry_signature(candidate)
        if signature not in distinct:
            distinct[signature] = candidate
    return list(distinct.values())


def propose_candidate_regions(
    gray: np.ndarray, *, max_regions: int = 12
) -> list[tuple[int, int, int, int]]:
    """Propose generic text/structure clusters without diagnosing damage."""
    image = _validate_gray(gray)
    if max_regions < 1:
        raise ValueError("max_regions must be positive")
    height, width = image.shape
    if height <= 512 and width <= 512:
        return [(0, 0, width, height)]

    dark = (image < 145).astype(np.uint8)
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, width // 30), 1)),
    )
    vertical = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, height // 30))),
    )
    content = dark & ~(horizontal | vertical)
    kernel_x = max(15, min(41, width // 60))
    kernel_y = max(15, min(41, height // 60))
    clustered = cv2.dilate(
        content,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_x, kernel_y)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
    proposals: list[tuple[float, tuple[int, int, int, int]]] = []
    page_area = float(max(image.size, 1))
    min_width = max(60, width // 24)
    min_height = max(45, height // 30)
    pad_x = max(12, kernel_x)
    pad_y = max(12, kernel_y)
    for index in range(1, count):
        x, y, box_width, box_height, area = [int(v) for v in stats[index]]
        if box_width < min_width or box_height < min_height:
            continue
        if box_width * box_height > page_area * 0.45:
            continue
        if box_width > width * 0.70 or box_height > height * 0.70:
            continue
        touches_top_or_bottom = y <= height * 0.04 or y + box_height >= height * 0.96
        touches_left_or_right = x <= width * 0.04 or x + box_width >= width * 0.96
        shallow_horizontal = box_height < height * 0.10 and box_width > width * 0.15
        shallow_vertical = box_width < width * 0.10 and box_height > height * 0.15
        if (touches_top_or_bottom and shallow_horizontal) or (
            touches_left_or_right and shallow_vertical
        ):
            continue
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + box_width + pad_x), min(
            height, y + box_height + pad_y
        )
        density = float(content[y0:y1, x0:x1].sum()) / max(
            float((x1 - x0) * (y1 - y0)), 1.0
        )
        score = float(area) * (0.5 + density)
        proposals.append((score, (x0, y0, x1, y1)))

    selected: list[tuple[int, int, int, int]] = []
    for _, region in sorted(proposals, key=lambda item: (-item[0], item[1])):
        x0, y0, x1, y1 = region
        duplicate = False
        for prior in selected:
            px0, py0, px1, py1 = prior
            intersection = max(0, min(x1, px1) - max(x0, px0)) * max(
                0, min(y1, py1) - max(y0, py0)
            )
            union = (x1 - x0) * (y1 - y0) + (px1 - px0) * (
                py1 - py0
            ) - intersection
            if union and intersection / union >= 0.75:
                duplicate = True
                break
        if not duplicate:
            selected.append(region)
        if len(selected) >= max_regions:
            break
    return selected


def generate_page_repair_candidates(
    gray: np.ndarray,
    *,
    max_fragments: int = 5,
    top_k: int = 8,
    max_regions: int = 12,
) -> list[RepairCandidate]:
    """Run the region-level solver over generic proposals from a whole page."""
    candidates, _ = _generate_page_repair_candidates_with_diagnostics(
        gray,
        max_fragments=max_fragments,
        top_k=top_k,
        max_regions=max_regions,
        family_mode="all_family_diagnostic",
    )
    return candidates


def _generate_page_repair_candidates_with_diagnostics(
    gray: np.ndarray,
    *,
    max_fragments: int = 5,
    top_k: int = 8,
    max_regions: int = 12,
    family_mode: str = "all_family_diagnostic",
) -> tuple[list[RepairCandidate], dict[str, dict]]:
    """Page candidates plus a diagnostics sidecar keyed by geometry signature."""
    image = _validate_gray(gray)
    if not 2 <= max_fragments <= 12:
        raise ValueError("max_fragments must be between 2 and 12")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if family_mode not in _FAMILY_MODES:
        raise ValueError(f"family_mode must be one of {_FAMILY_MODES}")
    _COMPONENT_DIAGNOSTICS.clear()
    candidates: list[RepairCandidate] = []
    if family_mode == "all_family_diagnostic":
        # The public path must keep flowing through the historical
        # `_rule_anchor_candidates` boundary, whose exact signature existing
        # tests pin via monkeypatch.
        candidates.extend(
            _rule_anchor_candidates(image, max_fragments=max_fragments)
        )
    else:
        candidates.extend(
            _rule_anchor_family_candidates(
                image, max_fragments=max_fragments, family_mode=family_mode
            )
        )
    for x0, y0, x1, y1 in propose_candidate_regions(
        image, max_regions=max_regions
    ):
        crop = image[y0:y1, x0:x1]
        for candidate in generate_repair_candidates(
            crop, max_fragments=max_fragments, top_k=top_k
        ):
            candidates.append(
                replace(candidate, region=(x0, y0, x1, y1))
            )

    # Spec 8.1 step 11 / spec 11: residual refinement of eligible emitted
    # pitch candidates (schema v2 feature; the unrefined candidate stays as
    # the exposed family alternative).
    if _SCHEMA_VERSION >= 2:
        pitch_measurements: dict[str, _RailMeasurements | None] = {}
        refined_candidates: list[RepairCandidate] = []
        for candidate in candidates:
            if candidate.score_terms.get("method_pitch") != 1.0 or (
                candidate.score_terms.get("pitch_refined")
            ):
                continue
            axis = candidate.partition_axis
            if axis not in pitch_measurements:
                pitch_measurements[axis] = _rail_measurements_for_axis(
                    image, axis=axis
                )
            axis_measurements = pitch_measurements[axis]
            if axis_measurements is None:
                continue
            refinement = _refine_pitch_offsets(
                image, candidate, axis=axis, measurements=axis_measurements
            )
            if refinement is None:
                continue
            refined_candidates.append(refinement.refined)
            _COMPONENT_DIAGNOSTICS[
                _geometry_signature_key(refinement.refined)
            ] = {
                "family": "method_pitch",
                "pitch_refinement": {
                    "deltas": list(refinement.deltas),
                    "refinement_score": refinement.refinement_score,
                    "runner_up_deltas": list(refinement.runner_up[0]),
                    "runner_up_score": refinement.runner_up[1],
                    "unrefined_signature": _geometry_signature_key(candidate),
                },
            }
        candidates.extend(refined_candidates)

    # Support is measured in canonical seam windows.  Horizontal and vertical
    # hypotheses have different profile dimensions and ink densities, so only
    # compare rule anchors that share a partition axis.
    strongest_rule_support: dict[Axis, float] = {}
    for candidate in candidates:
        if candidate.score_terms.get("method_rule_anchor") != 1.0:
            continue
        strongest_rule_support[candidate.partition_axis] = max(
            strongest_rule_support.get(candidate.partition_axis, 0.0),
            float(candidate.score_terms["non_rule_support"]),
        )
    candidates = [
        candidate
        for candidate in candidates
        if candidate.score_terms.get("method_rule_anchor") != 1.0
        or float(candidate.score_terms["non_rule_support"])
        >= 0.25 * strongest_rule_support[candidate.partition_axis]
    ]
    # Spec 8.5 (schema version 2): deterministic family-local round-robin
    # retention; the retained set is then serialized in the existing legacy
    # descending-total_gain order so `candidates[rank - 1]` keeps meaning.
    retention = _retain_by_family(candidates, top_k=top_k, family_mode=family_mode)
    retained_candidates = sorted(
        (record[0] for record in retention),
        key=lambda item: (
            -item.score_terms["total_gain"],
            item.region,
            item.partition_axis,
        ),
    )
    diagnostics: dict[str, dict] = {}
    for candidate in retained_candidates:
        key = _geometry_signature_key(candidate)
        if key in _COMPONENT_DIAGNOSTICS:
            diagnostics[key] = _COMPONENT_DIAGNOSTICS[key]
    return retained_candidates, diagnostics
