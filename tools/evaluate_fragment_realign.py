#!/usr/bin/env python3
"""Offline whole-PDF evaluator for local fragment realignment.

This is deliberately a laboratory tool.  It renders scan pages through the
same hidden-text-safe PDF path as production, discovers geometry without
fixture metadata, freezes that geometry, and only then runs symmetric OCR and
joins reviewed exemplar annotations for reporting.  It never imports or calls
``process_pdf``.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pymupdf


SOLUTION_ROOT = Path(__file__).resolve().parent.parent
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from mib_pipeline import fields, fragment_realign, model, ocr, pdf_loader  # noqa: E402


AMBIGUITY_MARGIN = 0.02
NEUTRAL_PARSE_CASE_ID = "MIB-999999"
ROTATION_HUMAN = {
    0: "0",
    1: "90 CCW",
    2: "180",
    3: "90 CW",
}
ARTIFACT_ONLY = "artifact_only"
FIELD_RECOVERY = "field_recovery"
HARD_ATTEMPT = "hard_attempt"
NEGATIVE_CONTROL = "negative_control"


# Reporting metadata only.  The discovery phase completes and is hashed before
# this table is read.  Nothing here is passed to region proposal, geometry
# ranking, OCR, or parsing.
REVIEWED_ANNOTATIONS: dict[tuple[str, int], dict[str, Any]] = {
    ("MIB-000802", 3): {
        "rotation_k_ccw": 3,
        "role": FIELD_RECOVERY,
        "expected_fields": {
            "home_world": "Titan Freeport",
            "visa_class": "XW-2",
            "arrival_date": "2026-05-04",
            "declared_purpose": "translation",
        },
    },
    ("MIB-000027", 1): {
        "rotation_k_ccw": 0,
        "role": FIELD_RECOVERY,
        "expected_fields": {"sponsor_id": "SPN-1345"},
        "ambiguity_note": "SPN-1545 must remain visible as an alternative if near-tied.",
    },
    ("MIB-000063", 2): {
        "rotation_k_ccw": 0,
        "role": FIELD_RECOVERY,
        "expected_fields": {"sponsor_id": "SPN-1680"},
    },
    ("MIB-000063", 4): {
        "rotation_k_ccw": 0,
        "role": HARD_ATTEMPT,
        "expected_fields": {"sponsor_id": "SPN-1680"},
        "ambiguity_note": "Slant/overlap attempt; abstention is allowed with visible provenance.",
    },
    ("MIB-000178", 5): {
        "rotation_k_ccw": 0,
        "role": FIELD_RECOVERY,
        # Spec 14 post-freeze targets: both the Applicant and the Visa Class.
        # Loaded only after the geometry freeze (spec 13); never an input to
        # region proposal, geometry, ranking, OCR, or parsing.
        "expected_fields": {
            "applicant_name": "Nexvara Zarix",
            "visa_class": "XW-1",
        },
    },
    ("MIB-000420", 1): {
        "rotation_k_ccw": 1,
        "role": ARTIFACT_ONLY,
        "expected_artifacts": ["SAMPLE DENIAL", "blue stamp"],
    },
    ("MIB-000420", 3): {
        "rotation_k_ccw": 0,
        "role": ARTIFACT_ONLY,
        "expected_artifacts": ["SAMPLE DENIAL", "COPY ARTIFACT", "SCAN TAB"],
    },
    ("MIB-000420", 4): {
        "rotation_k_ccw": 3,
        "role": HARD_ATTEMPT,
        "expected_artifacts": ["gray-anchor field block"],
        "ambiguity_note": "Applicant name remains diagnostic-only.",
    },
    ("MIB-000817", 3): {
        "rotation_k_ccw": 0,
        "role": FIELD_RECOVERY,
        "expected_fields": {"sponsor_id": "SPN-2167"},
    },
    ("MIB-000931", 4): {
        "rotation_k_ccw": 3,
        "role": FIELD_RECOVERY,
        "expected_fields": {
            "case_id": "MIB-000931",
            "applicant_name": "Ixotari Xandane",
            "species_code": "CENTAURI_SYNTH",
            "home_world": "Mars Dome-7",
            "visa_class": "DIP-1",
            "sponsor_id": "SPN-2703",
            "arrival_date": "2026-02-16",
            "declared_purpose": "translation",
        },
    },
    ("MIB-000977", 0): {
        "role": NEGATIVE_CONTROL,
        "expected_behavior": "abstain; ordinary small deskew only",
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
    return _sha256_bytes(header + contiguous.tobytes())


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "pdf"


def _write_png(output_dir: Path, relative: str, image: np.ndarray) -> str:
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), np.ascontiguousarray(array)):
        raise RuntimeError(f"failed to write PNG asset: {path}")
    return path.relative_to(output_dir).as_posix()


def _read_gray(output_dir: Path, relative: str) -> np.ndarray:
    image = cv2.imread(str(output_dir / relative), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read frozen image asset: {relative}")
    return image


def _provenance_image(mapping: np.ndarray) -> np.ndarray:
    """Color an integer provenance map without implying grayscale intensity."""
    values = np.asarray(mapping, dtype=np.int64)
    valid = values >= 0
    hsv = np.zeros((*values.shape, 3), dtype=np.uint8)
    if np.any(valid):
        normalized = values[valid] % 180
        hsv[..., 0][valid] = normalized.astype(np.uint8)
        hsv[..., 1][valid] = 220
        hsv[..., 2][valid] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _partition_overlay(source: np.ndarray, candidate: fragment_realign.RepairCandidate) -> np.ndarray:
    overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    height, width = source.shape
    angle = np.deg2rad(float(candidate.partition_angle_degrees))
    slope = float(np.tan(angle))
    for fragment in candidate.fragments[:-1]:
        boundary = fragment.interval[1]
        if candidate.partition_axis == "y":
            y0 = int(round(boundary - slope * width / 2.0))
            y1 = int(round(boundary + slope * width / 2.0))
            cv2.line(overlay, (0, y0), (width - 1, y1), (0, 0, 255), 2)
        else:
            x0 = int(round(boundary - slope * height / 2.0))
            x1 = int(round(boundary + slope * height / 2.0))
            cv2.line(overlay, (x0, 0), (x1, height - 1), (0, 0, 255), 2)
    for fragment in candidate.fragments:
        start, end = fragment.interval
        if candidate.partition_axis == "y":
            origin = (width // 2, int(round((start + end) / 2)))
        else:
            origin = (int(round((start + end) / 2)), height // 2)
        destination = (
            origin[0] + int(fragment.inverse_dx),
            origin[1] + int(fragment.inverse_dy),
        )
        cv2.arrowedLine(overlay, origin, destination, (0, 170, 0), 2, tipLength=0.25)
    return overlay


def _generate_view_candidates(
    gray: np.ndarray,
    *,
    max_fragments: int,
    top_k: int,
    family_mode: str,
) -> tuple[list[fragment_realign.RepairCandidate], dict[str, dict]]:
    """Single swappable boundary between page evaluation and geometry core."""
    return fragment_realign._generate_page_repair_candidates_with_diagnostics(
        gray,
        max_fragments=max_fragments,
        top_k=top_k,
        max_regions=12,
        family_mode=family_mode,
    )


def _candidate_retention_records(
    candidates: Sequence[fragment_realign.RepairCandidate],
    *,
    top_k: int,
    family_mode: str,
) -> list[dict | None]:
    """Spec 8.5 schema-v2 per-candidate retention fields, aligned to the
    candidate list order; None per candidate before schema version 2 so
    Stage A1 reports stay byte-identical."""
    if getattr(fragment_realign, "_SCHEMA_VERSION", 1) < 2:
        return [None] * len(candidates)
    retention = fragment_realign._retain_by_family(
        list(candidates), top_k=top_k, family_mode=family_mode
    )
    by_identity = {
        id(candidate): (family, family_rank, retention_round, retention_slot)
        for candidate, family, family_rank, retention_round, retention_slot in (
            retention
        )
    }
    records: list[dict | None] = []
    for candidate in candidates:
        entry = by_identity.get(id(candidate))
        records.append(
            None
            if entry is None
            else {
                "solver_family": entry[0],
                "family_rank": entry[1],
                "retention_round": entry[2],
                "retention_slot": entry[3],
            }
        )
    return records


def _candidate_closure_record(
    view: np.ndarray,
    candidate: fragment_realign.RepairCandidate,
    measurements_by_axis: dict[str, Any],
) -> dict[str, Any] | None:
    """Spec 12.2 closure measurement summary for one candidate, computed at
    geometry-discovery time (annotation-blind) from the frozen contract."""
    axis = candidate.partition_axis
    if axis not in measurements_by_axis:
        measurements_by_axis[axis] = fragment_realign._rail_measurements_for_axis(
            view, axis=axis
        )
    measurements = measurements_by_axis[axis]
    if measurements is None:
        return None
    report = fragment_realign._measure_endpoint_residuals(
        view,
        candidate.reconstruction,
        region=candidate.region,
        axis=axis,
        transforms=candidate.fragments,
        measurements=measurements,
    )
    interior = fragment_realign._interior_uncovered_pixels(
        candidate, candidate.fragments, axis
    )
    internal = [record.residual_quarter_px for record in report.internal]
    edges: dict[int, dict[str, Any]] = {}
    for record in report.crop_edges:
        entry = edges.setdefault(
            record.boundary,
            {
                "boundary": record.boundary,
                "matched_pairs": 0,
                "max_jump_quarter_px": 0,
                "max_worsening_quarter_px": 0,
            },
        )
        entry["matched_pairs"] += 1
        entry["max_jump_quarter_px"] = max(
            entry["max_jump_quarter_px"], record.residual_quarter_px
        )
        identity_quarter = int(
            round(4 * abs(record.before_center - record.after_center))
        )
        if identity_quarter <= 4:  # previously continuous rail
            entry["max_worsening_quarter_px"] = max(
                entry["max_worsening_quarter_px"],
                max(0, record.residual_quarter_px - identity_quarter),
            )
    # Rejoin boundaries with no matched pair must still be gated (fewer than
    # two strong continuing tracks), unless they coincide with a page edge.
    x0, y0, x1, y1 = candidate.region
    part_low, part_high = (y0, y1) if axis == "y" else (x0, x1)
    page_extent = measurements.canonical.shape[0]
    for boundary in (part_low, part_high):
        if 0 < boundary < page_extent and boundary not in edges:
            edges[boundary] = {
                "boundary": boundary,
                "matched_pairs": 0,
                "max_jump_quarter_px": 0,
                "max_worsening_quarter_px": 0,
            }
    return {
        "internal_matched": len(internal),
        "internal_median_quarter_px": float(np.median(internal)) if internal else 0.0,
        "internal_max_quarter_px": int(max(internal, default=0)),
        "unmatched_strong_source": int(report.unmatched_strong_source),
        "source_endpoint_absent": int(report.source_endpoint_absent),
        "clipped_strong_rails": int(report.clipped_strong_rails),
        "interior_uncovered_pixels": int(interior),
        "crop_edges": [edges[boundary] for boundary in sorted(edges)],
    }


def _serialized_view_candidates(
    output_dir: Path,
    run_id: str,
    view: np.ndarray,
    candidates: Sequence[fragment_realign.RepairCandidate],
    *,
    top_k: int,
    family_mode: str,
) -> list[dict[str, Any]]:
    """Serialize one view's retained candidates with the schema-v2 retention
    and closure records (both omitted before schema version 2)."""
    retention_records = _candidate_retention_records(
        candidates, top_k=top_k, family_mode=family_mode
    )
    measurements_by_axis: dict[str, Any] = {}
    serialized: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        record = _serialize_candidate_geometry(
            output_dir, run_id, rank, view, candidate
        )
        retention = retention_records[rank - 1]
        if retention is not None:
            record.update(retention)
            closure = _candidate_closure_record(
                view, candidate, measurements_by_axis
            )
            if closure is not None:
                record["closure"] = closure
        serialized.append(record)
    return serialized


def _candidate_crop_before(
    view: np.ndarray,
    candidate: fragment_realign.RepairCandidate,
) -> np.ndarray:
    reconstruction = candidate.reconstruction
    if candidate.source_view.shape == reconstruction.shape:
        return np.ascontiguousarray(candidate.source_view)
    x0, y0, x1, y1 = candidate.region
    crop = view[y0:y1, x0:x1]
    if crop.shape != reconstruction.shape:
        raise ValueError(
            f"candidate region {candidate.region} does not match reconstruction "
            f"{reconstruction.shape}"
        )
    return np.ascontiguousarray(crop)


def _full_after(
    view: np.ndarray,
    candidate: fragment_realign.RepairCandidate,
) -> np.ndarray:
    x0, y0, x1, y1 = candidate.region
    if (x0, y0, x1, y1) == (0, 0, view.shape[1], view.shape[0]):
        return np.ascontiguousarray(candidate.reconstruction)
    result = view.copy()
    if result[y0:y1, x0:x1].shape != candidate.reconstruction.shape:
        return result
    result[y0:y1, x0:x1] = candidate.reconstruction
    return result


def _mapping_consistent(candidate: fragment_realign.RepairCandidate) -> bool:
    source_to_destination = candidate.source_to_destination_map
    destination_to_source = candidate.destination_to_source_map
    source_indices = np.flatnonzero(source_to_destination.ravel() >= 0)
    if source_indices.size == 0:
        return False
    destinations = source_to_destination.ravel()[source_indices]
    in_bounds = (destinations >= 0) & (destinations < destination_to_source.size)
    if not np.all(in_bounds):
        return False
    return bool(
        np.all(destination_to_source.ravel()[destinations] == source_indices)
    )


def _moved_glyph_pixels(
    candidate: fragment_realign.RepairCandidate,
    glyph_mask: np.ndarray,
) -> int:
    moved = np.zeros(glyph_mask.shape, dtype=bool)
    for fragment in candidate.fragments:
        if not (fragment.inverse_dx or fragment.inverse_dy):
            continue
        start, end = fragment.interval
        if candidate.partition_axis == "y":
            moved[start:end, :] = True
        else:
            moved[:, start:end] = True
    visible = candidate.source_to_destination_map >= 0
    if visible.shape != moved.shape:
        return 0
    return int(np.count_nonzero(glyph_mask & moved & visible))


def _serialize_candidate_geometry(
    output_dir: Path,
    run_id: str,
    rank: int,
    view: np.ndarray,
    candidate: fragment_realign.RepairCandidate,
) -> dict[str, Any]:
    prefix = f"assets/{run_id}-candidate-{rank:02d}"
    before = _candidate_crop_before(view, candidate)
    after = np.ascontiguousarray(candidate.reconstruction)
    if before.shape != after.shape:
        raise ValueError("before and after candidate crops must have identical shapes")
    masks = fragment_realign.build_content_masks(before)
    full_after = _full_after(view, candidate)
    assets = {
        "crop_before": _write_png(output_dir, f"{prefix}-crop-before.png", before),
        "crop_after": _write_png(output_dir, f"{prefix}-crop-after.png", after),
        "full_after": _write_png(output_dir, f"{prefix}-full-after.png", full_after),
        "absolute_difference": _write_png(
            output_dir,
            f"{prefix}-absolute-difference.png",
            cv2.absdiff(before, after),
        ),
        "partition_overlay": _write_png(
            output_dir,
            f"{prefix}-partition-overlay.png",
            _partition_overlay(before, candidate),
        ),
        "glyph_mask": _write_png(output_dir, f"{prefix}-glyph-mask.png", masks.glyph_mask),
        "rule_mask": _write_png(output_dir, f"{prefix}-rule-mask.png", masks.rule_mask),
        "structure_mask": _write_png(
            output_dir,
            f"{prefix}-structure-mask.png",
            masks.structure_mask,
        ),
        "source_to_destination": _write_png(
            output_dir,
            f"{prefix}-source-to-destination.png",
            _provenance_image(candidate.source_to_destination_map),
        ),
        "destination_to_source": _write_png(
            output_dir,
            f"{prefix}-destination-to-source.png",
            _provenance_image(candidate.destination_to_source_map),
        ),
        "overlap_mask": _write_png(
            output_dir,
            f"{prefix}-overlap-mask.png",
            candidate.overlap_mask,
        ),
        "uncovered_mask": _write_png(
            output_dir,
            f"{prefix}-uncovered-mask.png",
            candidate.uncovered_mask,
        ),
        "cropped_source_mask": _write_png(
            output_dir,
            f"{prefix}-cropped-source-mask.png",
            candidate.cropped_source_mask,
        ),
    }
    asset_hashes = {
        name: _sha256_file(output_dir / relative)
        for name, relative in assets.items()
    }
    return {
        "rank": rank,
        "region": [int(value) for value in candidate.region],
        "partition_axis": candidate.partition_axis,
        "partition_angle_degrees": float(candidate.partition_angle_degrees),
        "fragments": [
            {
                "interval": [int(value) for value in fragment.interval],
                "inverse_dx": int(fragment.inverse_dx),
                "inverse_dy": int(fragment.inverse_dy),
            }
            for fragment in candidate.fragments
        ],
        "pixel_score_before": float(candidate.pixel_score_before),
        "pixel_score_after": float(candidate.pixel_score_after),
        "score_terms": {
            str(key): float(value) for key, value in candidate.score_terms.items()
        },
        "provenance_contract": {
            "source_to_destination": (
                "source-shaped flattened destination indices; -1 means no visible destination"
            ),
            "destination_to_source": (
                "destination-shaped flattened source indices; -1 means uncovered destination"
            ),
            "reciprocal_visible_mapping": _mapping_consistent(candidate),
        },
        "moved_glyph_pixels": _moved_glyph_pixels(candidate, masks.glyph_mask),
        "overlap_pixels": int(np.count_nonzero(candidate.overlap_mask)),
        "uncovered_pixels": int(np.count_nonzero(candidate.uncovered_mask)),
        "cropped_source_pixels": int(np.count_nonzero(candidate.cropped_source_mask)),
        "source_crop_sha256": _sha256_array(before),
        "reconstruction_sha256": _sha256_array(after),
        "assets": assets,
        "asset_sha256": asset_hashes,
    }


def _discover_geometry(
    pdf_paths: Sequence[Path],
    output_dir: Path,
    *,
    max_fragments: int,
    top_k: int,
    family_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Discover and persist geometry without reading any fixture annotations."""
    started = time.monotonic()
    runs: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for pdf_number, pdf_path in enumerate(pdf_paths):
        pdf_hash = _sha256_file(pdf_path)
        input_record = {
            "input_index": pdf_number,
            "source_pdf": str(pdf_path),
            "source_pdf_sha256": pdf_hash,
        }
        input_records.append(input_record)
        with pymupdf.open(pdf_path) as metadata_doc:
            page_metadata = pdf_loader.load_pages(metadata_doc)
        for page_meta in page_metadata:
            if page_meta.kind != model.PageKind.SCAN:
                continue
            # render_gray mutates its page while redacting hidden text, so each
            # render owns a newly opened document.
            with pymupdf.open(pdf_path) as render_doc:
                gray = ocr.render_gray(render_doc[page_meta.index], page_meta)
            page_base = (
                f"p{pdf_number:03d}-{_safe_name(pdf_path.stem)}-"
                f"page{page_meta.index + 1:03d}"
            )
            original_asset = _write_png(
                output_dir,
                f"assets/{page_base}-full-original.png",
                gray,
            )
            rendered_hash = _sha256_array(gray)
            for rotation_k_ccw in range(4):
                view_started = time.monotonic()
                view = (
                    gray
                    if rotation_k_ccw == 0
                    else np.ascontiguousarray(np.rot90(gray, k=rotation_k_ccw))
                )
                run_id = f"{page_base}-rot{rotation_k_ccw}"
                rotated_asset = _write_png(
                    output_dir,
                    f"assets/{run_id}-full-before.png",
                    view,
                )
                candidates, candidate_diagnostics = _generate_view_candidates(
                    view,
                    max_fragments=max_fragments,
                    top_k=top_k,
                    family_mode=family_mode,
                )
                serialized = _serialized_view_candidates(
                    output_dir,
                    run_id,
                    view,
                    candidates,
                    top_k=top_k,
                    family_mode=family_mode,
                )
                runs.append(
                    {
                        "record_id": run_id,
                        "input_index": pdf_number,
                        "source_pdf": str(pdf_path),
                        "source_pdf_sha256": pdf_hash,
                        "page_index": int(page_meta.index),
                        "pdf_page_number": int(page_meta.index + 1),
                        "page_kind": page_meta.kind.value,
                        "rotation_k_ccw": rotation_k_ccw,
                        "rotation_human": ROTATION_HUMAN[rotation_k_ccw],
                        "render_dpi": int(ocr.RENDER_DPI),
                        "rendered_page_sha256": rendered_hash,
                        "rotated_view_sha256": _sha256_array(view),
                        "geometry_candidate_count": len(serialized),
                        "geometry_gate": (
                            "triggered"
                            if serialized
                            else "abstained: geometry generator returned no candidate"
                        ),
                        "geometry_elapsed_seconds": time.monotonic() - view_started,
                        "assets": {
                            "full_original": original_asset,
                            "full_before": rotated_asset,
                        },
                        "candidates": serialized,
                        "candidate_diagnostics": candidate_diagnostics,
                    }
                )
                del candidates, view
    return runs, input_records, time.monotonic() - started


def _words_to_lines(
    words: Iterable[ocr.OcrWord],
    *,
    page_index: int,
) -> list[model.Line]:
    groups: dict[tuple[int, int, int], list[ocr.OcrWord]] = {}
    for word in words:
        groups.setdefault(word.line_key, []).append(word)
    lines: list[model.Line] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda word: word.x)
        lines.append(
            model.Line(
                text=" ".join(word.text for word in group),
                page_index=page_index,
                source=model.Source.OCR,
                conf=float(np.mean([word.conf for word in group])),
            )
        )
    return lines


def _ocr_and_parse(
    image: np.ndarray,
    *,
    page_index: int,
    engine: ocr.OcrEngine,
) -> dict[str, Any]:
    """Run the exact same raw+sparse OCR and fresh-page parser on either view."""
    contiguous = np.ascontiguousarray(image)
    words = list(engine.words(contiguous, sparse=False))
    words.extend(engine.words(contiguous, sparse=True))
    lines = _words_to_lines(words, page_index=page_index)
    page = model.Page(index=page_index, kind=model.PageKind.SCAN, lines=list(lines))
    candidates, flag_candidates, findings = fields.collect_candidates(
        [page],
        NEUTRAL_PARSE_CASE_ID,
    )
    evidence = fields.reconcile(candidates, flag_candidates, findings)
    values = {
        key: value for key, value in sorted(evidence.values.items()) if value is not None
    }
    average_confidence = float(np.mean([line.conf for line in lines])) if lines else 0.0
    reading_score = (
        2.0 * len(values)
        + float(sum(bool(value) for value in evidence.known.values()))
        + average_confidence
    )
    return {
        "lines": [
            {"text": line.text, "confidence": float(line.conf)}
            for line in lines
        ],
        "parsed": {
            "values": values,
            "known": {key: bool(value) for key, value in sorted(evidence.known.items())},
            "confidence": {
                key: float(value) for key, value in sorted(evidence.conf.items())
            },
            "risk_flags": sorted(evidence.flags),
            "risk_flags_known": bool(evidence.flags_known),
            "finding": (
                {
                    "label": evidence.finding.label,
                    "reason": evidence.finding.reason,
                    "confidence": float(evidence.finding.conf),
                }
                if evidence.finding is not None
                else None
            ),
        },
        "reading_score": reading_score,
        "ocr_word_count": len(words),
        "average_line_confidence": average_confidence,
    }


# Spec 6 frozen geometric-uniqueness constants (distinct from the legacy
# semantic AMBIGUITY_MARGIN, which stays at its frozen value).
GEOMETRIC_UNIQUENESS_MARGIN = 0.03
CROSS_FAMILY_AGREEMENT_PX = 2


def _fragment_boxes(
    candidate: dict[str, Any],
) -> list[tuple[tuple[int, int, int, int], int, int]]:
    """Page-coordinate fragment boxes with their displacement vectors."""
    x0, y0, x1, y1 = candidate["region"]
    axis = candidate["partition_axis"]
    boxes = []
    for fragment in candidate["fragments"]:
        start, end = fragment["interval"]
        box = (
            (x0, y0 + start, x1, y0 + end)
            if axis == "y"
            else (x0 + start, y0, x0 + end, y1)
        )
        boxes.append((box, int(fragment["inverse_dx"]), int(fragment["inverse_dy"])))
    return boxes


def _boxes_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _shared_support_disagreement(
    a: dict[str, Any], b: dict[str, Any]
) -> int | None:
    """Worst Chebyshev destination disagreement over shared source pixels,
    or None when the candidates share no source support."""
    worst: int | None = None
    for box_a, dxa, dya in _fragment_boxes(a):
        for box_b, dxb, dyb in _fragment_boxes(b):
            if _boxes_intersect(box_a, box_b):
                difference = max(abs(dxa - dxb), abs(dya - dyb))
                worst = difference if worst is None else max(worst, difference)
    return worst


def _closure_gates_pass(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """Spec 12.2 / 6.1 region_repair closure gates on the frozen per-candidate
    closure measurement record."""
    closure = candidate.get("closure")
    if not closure:
        return False, ["no closure measurement recorded"]
    failures: list[str] = []
    if int(closure["unmatched_strong_source"]) > 0:
        failures.append("unmatched strong endpoint (closure failure)")
    if float(closure["internal_median_quarter_px"]) > 4:
        failures.append("internal median residual above 1px")
    if int(closure["internal_max_quarter_px"]) > 8:
        failures.append("internal max residual above 2px")
    if int(closure["clipped_strong_rails"]) > 0:
        failures.append("strong source rail clipped")
    if int(closure["interior_uncovered_pixels"]) > 0:
        failures.append("interior uncovered component")
    if int(candidate.get("overlap_pixels", 0)) > 0:
        failures.append("transformation-created destination overlap")
    for edge in closure["crop_edges"]:
        boundary = edge["boundary"]
        if int(edge["matched_pairs"]) < 2:
            failures.append(
                f"fewer than two strong continuing rail tracks at {boundary}"
            )
        if int(edge["max_jump_quarter_px"]) > 8:
            failures.append(f"crop-edge jump above 2px at {boundary}")
        if int(edge["max_worsening_quarter_px"]) > 4:
            failures.append(
                f"previously continuous rail worsened above 1px at {boundary}"
            )
    return not failures, failures


def _salvage_gates_pass(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """Spec 12.3 partial-salvage bounds.

    The per-pixel field-support gates are enforced conservatively: any
    interior uncovered component in the region forces abstention (the frozen
    OCR word interface carries no geometry, so exact field boxes are not
    observable; region-wide enforcement is strictly tighter)."""
    x0, y0, x1, y1 = candidate["region"]
    area = max((x1 - x0) * (y1 - y0), 1)
    destination_loss = (
        int(candidate.get("overlap_pixels", 0))
        + int(candidate.get("uncovered_pixels", 0))
    ) / area
    source_loss = int(candidate.get("cropped_source_pixels", 0)) / area
    failures: list[str] = []
    if destination_loss > 0.07:
        failures.append("destination loss fraction above 7%")
    if source_loss > 0.07:
        failures.append("source loss fraction above 7%")
    closure = candidate.get("closure") or {}
    if int(closure.get("interior_uncovered_pixels", 0)) > 0:
        failures.append("interior uncovered component in the region")
    return not failures, failures


def _regional_outcome(
    primary: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> tuple[str, str]:
    """Spec 6: regional_reconstruction_outcome in the prerequisite order —
    geometric uniqueness first, then region_repair closure, then
    partial_field_salvage, then geometry_only."""
    family = primary.get("solver_family") or "method_seam"
    primary_gain = float(primary["score_terms"].get("total_gain", 0.0))
    for candidate in candidates:
        if candidate is primary:
            continue
        if not _boxes_intersect(
            tuple(primary["region"]), tuple(candidate["region"])
        ):
            continue
        # Arthur's 2026-07-29 ruling (specs/2026-07-29-s6-passing-candidates-
        # ruling.md): "passing candidates" means closure-passing — a candidate
        # failing its own closure gates cannot veto, in either layer.
        if not _closure_gates_pass(candidate)[0]:
            continue
        if candidate.get("solver_family", "method_seam") == family:
            gain = float(candidate["score_terms"].get("total_gain", 0.0))
            # A "distinct transform" is a genuinely different mapping: the
            # displacement functions must disagree somewhere on shared
            # support.  A re-crop of the same displacement is the same
            # transform and cannot create ambiguity.
            disagreement = _shared_support_disagreement(primary, candidate)
            if (disagreement or 0) > 0 and (
                primary_gain - gain < GEOMETRIC_UNIQUENESS_MARGIN
            ):
                return (
                    "underdetermined",
                    "family-local ambiguity: a distinct transform sits inside "
                    f"the frozen {GEOMETRIC_UNIQUENESS_MARGIN} total_gain margin",
                )
        else:
            disagreement = _shared_support_disagreement(primary, candidate)
            if disagreement is not None and (
                disagreement > CROSS_FAMILY_AGREEMENT_PX
            ):
                return (
                    "underdetermined",
                    "cross-family conflict: passing families map shared source "
                    f"pixels more than {CROSS_FAMILY_AGREEMENT_PX}px apart",
                )
    closed, closure_failures = _closure_gates_pass(primary)
    if closed:
        return ("region_repair", "regional closure gates passed")
    if primary.get("field_recovery_status") == "pixel_supported":
        salvage_ok, salvage_failures = _salvage_gates_pass(primary)
        if salvage_ok:
            return (
                "partial_field_salvage",
                "pixel-supported field with bounded transform loss; closure "
                "failed: " + "; ".join(closure_failures),
            )
        return (
            "geometry_only",
            "salvage bounds failed: " + "; ".join(salvage_failures),
        )
    return ("geometry_only", "closure gates failed: " + "; ".join(closure_failures))


def _report_candidate_outcome(
    checks: dict[str, bool],
    *,
    ambiguous_fields: Sequence[str],
) -> dict[str, str]:
    """Separate a recovered field from an unproven whole-page repair."""
    if all(checks.values()):
        return {
            "field_recovery_status": "pixel_supported",
            "full_page_geometry_status": "unverified",
            "classification": "underdetermined",
            "acceptance_reason": (
                "partial field recovery is pixel-supported; it does not prove "
                "full-page alignment"
            ),
        }
    if checks["geometry_score_improves"] and checks["reciprocal_provenance"]:
        if ambiguous_fields:
            reason = (
                "abstained: near-tied candidates disagree on parser-valid "
                f"fields {', '.join(ambiguous_fields)}"
            )
        elif not checks["reading_score_improves"]:
            reason = "abstained: geometry improved but symmetric OCR/parser score did not"
        elif not checks["new_or_changed_parser_valid_value"]:
            reason = (
                "abstained: OCR confidence changed but no parser-valid field "
                "was newly recovered or changed"
            )
        elif not checks["moved_glyph_support"]:
            reason = "abstained: no visible glyph pixel intersects a non-zero fragment"
        else:
            reason = "abstained: acceptance gate incomplete"
        return {
            "field_recovery_status": "underdetermined",
            "full_page_geometry_status": "unverified",
            "classification": "underdetermined",
            "acceptance_reason": reason,
        }
    return {
        "field_recovery_status": "pixel_unsupported",
        "full_page_geometry_status": "pixel_unsupported",
        "classification": "pixel_unsupported",
        "acceptance_reason": (
            "rejected: geometry score or reciprocal provenance gate failed"
        ),
    }


def _select_reporting_candidate(
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Choose report evidence without changing the frozen geometry ranking."""
    accepted = [
        candidate
        for candidate in candidates
        if candidate.get("field_recovery_status") == "pixel_supported"
    ]
    if accepted:
        return min(accepted, key=lambda candidate: int(candidate.get("rank", 1))), (
            "selected the highest-ranked candidate with pixel-supported field recovery"
        )
    return min(candidates, key=lambda candidate: int(candidate.get("rank", 1))), (
        "no candidate had pixel-supported field recovery; selected geometry rank 1 "
        "to explain the abstention"
    )


def _evaluate_ocr(
    frozen_runs: list[dict[str, Any]],
    output_dir: Path,
    *,
    engine: ocr.OcrEngine,
) -> float:
    started = time.monotonic()
    cache: dict[str, dict[str, Any]] = {}
    cache_elapsed: dict[str, float] = {}
    for run in frozen_runs:
        run_ocr_started = time.monotonic()
        for candidate in run["candidates"]:
            before_asset = candidate["assets"]["crop_before"]
            before = cache.get(before_asset)
            if before is None:
                before_started = time.monotonic()
                before = _ocr_and_parse(
                    _read_gray(output_dir, before_asset),
                    page_index=run["page_index"],
                    engine=engine,
                )
                cache[before_asset] = before
                cache_elapsed[before_asset] = time.monotonic() - before_started
            after_started = time.monotonic()
            after = _ocr_and_parse(
                _read_gray(output_dir, candidate["assets"]["crop_after"]),
                page_index=run["page_index"],
                engine=engine,
            )
            candidate["baseline_ocr"] = before
            candidate["repaired_ocr"] = after
            candidate["baseline_ocr_elapsed_seconds"] = cache_elapsed[before_asset]
            candidate["repaired_ocr_elapsed_seconds"] = time.monotonic() - after_started
            candidate["reading_score_gain"] = (
                float(after["reading_score"]) - float(before["reading_score"])
            )

        if not run["candidates"]:
            run["classification"] = "pixel_unsupported"
            run["field_recovery_status"] = "pixel_unsupported"
            run["full_page_geometry_status"] = "pixel_unsupported"
            run["acceptance_reason"] = run["geometry_gate"]
            run["selected_candidate_rank"] = None
            run["candidate_selection_reason"] = (
                "no geometric candidate survived the frozen core gate"
            )
            run["ambiguous_fields"] = []
            if getattr(fragment_realign, "_SCHEMA_VERSION", 1) >= 2:
                run["selected_candidate_retention_slot"] = None
                run["selected_candidate_family_rank"] = None
                run["regional_reconstruction_outcome"] = None
                run["regional_reconstruction_reason"] = (
                    "no geometric candidate survived the frozen core gate"
                )
            run["ocr_and_parse_elapsed_seconds"] = time.monotonic() - run_ocr_started
            continue

        best_gain = max(
            float(candidate["score_terms"].get("total_gain", 0.0))
            for candidate in run["candidates"]
        )
        near = [
            candidate
            for candidate in run["candidates"]
            if best_gain - float(candidate["score_terms"].get("total_gain", 0.0))
            <= AMBIGUITY_MARGIN
        ]
        values_by_field: dict[str, set[str]] = {}
        for candidate in near:
            for field, value in candidate["repaired_ocr"]["parsed"]["values"].items():
                values_by_field.setdefault(field, set()).add(str(value))
        ambiguous_fields = sorted(
            field for field, values in values_by_field.items() if len(values) > 1
        )
        run["ambiguous_fields"] = ambiguous_fields

        for candidate in run["candidates"]:
            score_before = float(candidate["pixel_score_before"])
            score_after = float(candidate["pixel_score_after"])
            total_gain = float(candidate["score_terms"].get("total_gain", 0.0))
            baseline_values = candidate["baseline_ocr"]["parsed"]["values"]
            repaired_values = candidate["repaired_ocr"]["parsed"]["values"]
            new_or_changed_value = any(
                field not in baseline_values or baseline_values[field] != value
                for field, value in repaired_values.items()
            )
            checks = {
                "geometry_score_improves": score_after > score_before and total_gain > 0,
                "reading_score_improves": float(candidate["reading_score_gain"]) > 0,
                "new_or_changed_parser_valid_value": new_or_changed_value,
                "moved_glyph_support": int(candidate["moved_glyph_pixels"]) > 0,
                "reciprocal_provenance": bool(
                    candidate["provenance_contract"]["reciprocal_visible_mapping"]
                ),
                "no_parser_ambiguity": not ambiguous_fields,
            }
            candidate["acceptance_checks"] = checks
            candidate.update(
                _report_candidate_outcome(
                    checks,
                    ambiguous_fields=ambiguous_fields,
                )
            )
        primary, selection_reason = _select_reporting_candidate(run["candidates"])
        run["selected_candidate_rank"] = int(primary.get("rank", 1))
        run["candidate_selection_reason"] = selection_reason
        if "retention_slot" in primary:
            # Spec 13 schema v2: retention/family positions of the selected
            # candidate, with the legacy rank preserved as a diagnostic.
            run["selected_candidate_retention_slot"] = int(
                primary["retention_slot"]
            )
            run["selected_candidate_family_rank"] = int(primary["family_rank"])
            outcome, outcome_reason = _regional_outcome(
                primary, run["candidates"]
            )
            run["regional_reconstruction_outcome"] = outcome
            run["regional_reconstruction_reason"] = outcome_reason
        for key in (
            "classification",
            "field_recovery_status",
            "full_page_geometry_status",
            "acceptance_reason",
        ):
            run[key] = primary[key]
        run["ocr_and_parse_elapsed_seconds"] = time.monotonic() - run_ocr_started
    return time.monotonic() - started


def _case_id_from_path(path: str) -> str | None:
    match = re.search(r"MIB-\d{6}", Path(path).name, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _join_reporting_annotations(runs: list[dict[str, Any]]) -> None:
    """Post-freeze reporting join.  Never called from discovery or ranking."""
    for run in runs:
        case_id = _case_id_from_path(run["source_pdf"])
        annotation = REVIEWED_ANNOTATIONS.get((case_id, run["page_index"])) if case_id else None
        if annotation is None:
            run["evaluation_annotation"] = None
        else:
            run["evaluation_annotation"] = {
                **annotation,
                "selected_rotation_matches": (
                    "rotation_k_ccw" not in annotation
                    or annotation["rotation_k_ccw"] == run["rotation_k_ccw"]
                ),
            }
        for candidate in run["candidates"]:
            candidate["label_match"] = None
            candidate["label_conflicts"] = []
            if annotation is None:
                continue
            rotation_matches = (
                "rotation_k_ccw" not in annotation
                or annotation["rotation_k_ccw"] == run["rotation_k_ccw"]
            )
            if not rotation_matches:
                continue
            expected = annotation.get("expected_fields", {})
            repaired = candidate.get("repaired_ocr", {}).get("parsed", {}).get("values", {})
            compared = {
                field: repaired[field]
                for field in expected
                if field in repaired
            }
            conflicts = sorted(
                field for field, value in compared.items() if value != expected[field]
            )
            candidate["label_conflicts"] = conflicts
            candidate["label_match"] = (
                bool(expected)
                and len(compared) == len(expected)
                and not conflicts
            )
            expected_artifacts = annotation.get("expected_artifacts", [])
            if expected_artifacts:
                before_text = " ".join(
                    line["text"] for line in candidate["baseline_ocr"]["lines"]
                ).casefold()
                after_text = " ".join(
                    line["text"] for line in candidate["repaired_ocr"]["lines"]
                ).casefold()
                checks = []
                for artifact in expected_artifacts:
                    # Descriptive visual targets cannot be reduced to an OCR
                    # substring.  Keep them explicit for human scoring.
                    machine_readable = artifact.upper() == artifact
                    checks.append(
                        {
                            "artifact": artifact,
                            "before_ocr_match": (
                                artifact.casefold() in before_text
                                if machine_readable
                                else None
                            ),
                            "after_ocr_match": (
                                artifact.casefold() in after_text
                                if machine_readable
                                else None
                            ),
                            "visual_review_required": True,
                        }
                    )
                candidate["artifact_only_checks"] = checks


def _summary(runs: Sequence[dict[str, Any]]) -> dict[str, int]:
    classifications = ("pixel_supported", "underdetermined", "pixel_unsupported")
    summary = {
        "scan_page_rotations": len(runs),
        "triggered_rotations": sum(bool(run["candidates"]) for run in runs),
        "candidate_count": sum(len(run["candidates"]) for run in runs),
    }
    for classification in classifications:
        summary[classification] = sum(
            run.get("classification") == classification for run in runs
        )
    summary["label_matching_candidates"] = sum(
        candidate.get("label_match") is True
        for run in runs
        for candidate in run["candidates"]
    )
    summary["label_conflicting_candidates"] = sum(
        bool(candidate.get("label_conflicts"))
        for run in runs
        for candidate in run["candidates"]
    )
    return summary


def _write_json_records(
    output_dir: Path,
    runs: list[dict[str, Any]],
) -> None:
    results_path = output_dir / "results.jsonl"
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as stream:
        for run in runs:
            payload = _canonical_json(run)
            stream.write(payload + "\n")
            (records_dir / f"{run['record_id']}.json").write_text(
                json.dumps(run, default=_json_default, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def _img(relative: str, caption: str) -> str:
    return (
        '<figure><img loading="lazy" src="'
        + html.escape(relative, quote=True)
        + '" alt="'
        + html.escape(caption, quote=True)
        + '"><figcaption>'
        + html.escape(caption)
        + "</figcaption></figure>"
    )


def _record_signature_key(record: dict[str, Any]) -> str:
    """The geometry-signature key for a SERIALIZED candidate record,
    mirroring fragment_realign._geometry_signature_key."""
    return repr(
        (
            tuple(record["region"]),
            record["partition_axis"],
            tuple(
                (tuple(f["interval"]), f["inverse_dx"], f["inverse_dy"])
                for f in record["fragments"]
            ),
        )
    )


def _json_pre(value: Any) -> str:
    return "<pre>" + html.escape(
        json.dumps(value, default=_json_default, indent=2, sort_keys=True)
    ) + "</pre>"


def _status_lead_html(run: dict[str, Any]) -> str:
    """Render the report's field-level and page-level claims independently."""
    classification = html.escape(str(run.get("classification", "unclassified")))
    field_status = html.escape(str(run.get("field_recovery_status", "unreported")))
    page_status = html.escape(str(run.get("full_page_geometry_status", "unreported")))
    reason = html.escape(str(run.get("acceptance_reason", run.get("geometry_gate", ""))))
    return (
        '<p class="status-lead"><strong>Overall: '
        + classification
        + "</strong><br>Field recovery: <strong>"
        + field_status
        + "</strong><br>Full-page geometry: <strong>"
        + page_status
        + "</strong><br>"
        + reason
        + "</p>"
    )


def _write_html(
    output_dir: Path,
    runs: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    summary = manifest["summary"]
    cards: list[str] = []
    for run in runs:
        annotation = run.get("evaluation_annotation")
        card = [
            f'<article id="{html.escape(run["record_id"], quote=True)}">',
            f"<h2>{html.escape(run['record_id'])}</h2>",
            _status_lead_html(run),
            '<p><a href="records/'
            + html.escape(run["record_id"], quote=True)
            + '.json">Download this JSON record</a></p>',
            '<div class="grid">',
            _img(run["assets"]["full_original"], "Full original rendered page"),
            _img(
                run["assets"]["full_before"],
                f"Full page before repair — {run['rotation_human']} (np.rot90 k={run['rotation_k_ccw']})",
            ),
            "</div>",
            "<h3>Geometry freeze</h3>",
            _json_pre(
                {
                    "source_pdf_sha256": run["source_pdf_sha256"],
                    "rendered_page_sha256": run["rendered_page_sha256"],
                    "rotated_view_sha256": run["rotated_view_sha256"],
                    "candidate_count": run["geometry_candidate_count"],
                    "geometry_elapsed_seconds": run["geometry_elapsed_seconds"],
                    "ambiguous_fields": run.get("ambiguous_fields", []),
                }
            ),
        ]
        if not run["candidates"]:
            card.append("<p>No geometric candidate survived the frozen core gate.</p>")
        else:
            selected_rank = int(run.get("selected_candidate_rank") or 1)
            primary = next(
                (
                    candidate
                    for candidate in run["candidates"]
                    if int(candidate.get("rank", 1)) == selected_rank
                ),
                run["candidates"][0],
            )
            selected_heading = (
                "Selected field-recovery candidate"
                if primary.get("field_recovery_status") == "pixel_supported"
                else "Selected abstention explanation"
            )
            assets = primary["assets"]
            card.extend(
                [
                    "<h3>"
                    + selected_heading
                    + " — geometry rank "
                    + str(selected_rank)
                    + "</h3>",
                    "<p>"
                    + html.escape(
                        str(
                            run.get(
                                "candidate_selection_reason",
                                "selected geometry rank 1",
                            )
                        )
                    )
                    + "</p>",
                    '<div class="grid">',
                    _img(
                        assets["full_after"],
                        (
                            "Full page after primary candidate — field-level recovery "
                            "only; full-page geometry remains unverified"
                        ),
                    ),
                    _img(assets["partition_overlay"], "Partition boundaries and inverse displacement"),
                    _img(assets["crop_before"], "Native crop before"),
                    _img(assets["crop_after"], "Native crop after"),
                    _img(assets["absolute_difference"], "Absolute pixel difference"),
                    _img(assets["glyph_mask"], "Glyph mask"),
                    _img(assets["rule_mask"], "Rule mask"),
                    _img(assets["structure_mask"], "Structure and patch mask"),
                    _img(assets["source_to_destination"], "Source-to-destination provenance"),
                    _img(assets["destination_to_source"], "Destination-to-source provenance"),
                    _img(assets["overlap_mask"], "Overlap mask"),
                    _img(assets["uncovered_mask"], "Uncovered mask"),
                    _img(assets["cropped_source_mask"], "Cropped source mask"),
                    "</div>",
                    "<h3>Regional reconstruction outcome</h3>",
                    "<p><strong class=\"outcome-"
                    + html.escape(
                        str(run.get("regional_reconstruction_outcome", "none")),
                        quote=True,
                    )
                    + "\">"
                    + html.escape(
                        str(run.get("regional_reconstruction_outcome", "n/a"))
                    )
                    + "</strong> — "
                    + html.escape(str(run.get("regional_reconstruction_reason", "")))
                    + "</p>",
                    "<p>Retention slot "
                    + html.escape(str(run.get("selected_candidate_retention_slot")))
                    + ", family rank "
                    + html.escape(str(run.get("selected_candidate_family_rank")))
                    + " (legacy geometry rank "
                    + html.escape(str(run.get("selected_candidate_rank")))
                    + " — diagnostic)</p>",
                    *(
                        [
                            "<h3>Component / gauge diagnostics</h3>",
                            _json_pre(
                                {
                                    key: value
                                    for key, value in run.get(
                                        "candidate_diagnostics", {}
                                    )
                                    .get(_record_signature_key(primary), {})
                                    .items()
                                    if key
                                    in (
                                        "per_seam",
                                        "joint",
                                        "gauge",
                                        "pitch_refinement",
                                        "support_floor",
                                    )
                                }
                            ),
                        ]
                        if run.get("candidate_diagnostics", {}).get(
                            _record_signature_key(primary)
                        )
                        else []
                    ),
                    "<h3>Geometry and score terms</h3>",
                    _json_pre(
                        {
                            key: primary[key]
                            for key in (
                                "rank",
                                "solver_family",
                                "family_rank",
                                "retention_slot",
                                "classification",
                                "field_recovery_status",
                                "full_page_geometry_status",
                                "acceptance_reason",
                                "region",
                                "partition_axis",
                                "partition_angle_degrees",
                                "fragments",
                                "pixel_score_before",
                                "pixel_score_after",
                                "score_terms",
                                "moved_glyph_pixels",
                                "overlap_pixels",
                                "uncovered_pixels",
                                "cropped_source_pixels",
                                "closure",
                                "provenance_contract",
                                "acceptance_checks",
                            )
                            if key in primary
                        }
                    ),
                    "<h3>Symmetric OCR and parser delta</h3>",
                    '<div class="grid">',
                    "<section><h4>Before</h4>"
                    + _json_pre(primary["baseline_ocr"])
                    + "</section>",
                    "<section><h4>After</h4>"
                    + _json_pre(primary["repaired_ocr"])
                    + "</section>",
                    "</div>",
                    "<h3>Alternatives</h3>",
                    '<div class="grid alternatives">',
                ]
            )
            # Spec 8.5: reports present candidates by retention_slot; the
            # legacy rank stays visible as a diagnostic.  Pre-schema-v2
            # records have no slot and keep the legacy order.
            for alternative in sorted(
                run["candidates"],
                key=lambda item: (
                    int(item.get("retention_slot") or item.get("rank", 1)),
                    int(item.get("rank", 1)),
                ),
            ):
                if int(alternative.get("rank", 1)) == selected_rank:
                    continue
                alternative_status = html.escape(
                    str(alternative.get("field_recovery_status", "unreported"))
                )
                slot = alternative.get("retention_slot")
                heading = (
                    f"retention slot {int(slot)} (geometry rank "
                    f"{alternative['rank']})"
                    if slot is not None
                    else f"geometry rank {alternative['rank']}"
                )
                rejected = (
                    str(alternative.get("classification")) == "pixel_unsupported"
                )
                card.append(
                    (
                        '<details class="rejected"><summary>REJECTED candidate'
                        " — collapsed (" + heading + ")</summary>"
                        '<div class="rejection-banner">REJECTED: '
                        + html.escape(str(alternative.get("acceptance_reason", "")))
                        + "</div>"
                        if rejected
                        else ""
                    )
                    + "<section><h4>Alternative — "
                    + heading
                    + "</h4>"
                    + "<p>Field recovery: <strong>"
                    + alternative_status
                    + "</strong> — "
                    + html.escape(str(alternative.get("acceptance_reason", "")))
                    + "</p>"
                    + _img(alternative["assets"]["crop_after"], "Reconstruction")
                    + _json_pre(
                        {
                            "fragments": alternative["fragments"],
                            "score_terms": alternative["score_terms"],
                            "classification": alternative.get("classification"),
                            "acceptance_reason": alternative.get("acceptance_reason"),
                            "repaired_values": alternative.get("repaired_ocr", {})
                            .get("parsed", {})
                            .get("values", {}),
                        }
                    )
                    + "</section>"
                    + ("</details>" if rejected else "")
                )
            card.append("</div>")
        card.extend(
            [
                "<h3>Post-freeze evaluation label</h3>",
                _json_pre(annotation if annotation is not None else {"annotation": None}),
                "</article>",
            ]
        )
        cards.append("".join(card))

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fragment realignment evidence report</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1500px; padding: 24px; background: #f4f1ea; color: #181818; }}
header, article {{ background: white; border: 1px solid #c9c3b8; border-radius: 10px; margin: 0 0 24px; padding: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
figure {{ margin: 0; border: 1px solid #ddd; padding: 8px; background: #fff; }}
img {{ width: 100%; height: auto; image-rendering: auto; }}
figcaption {{ margin-top: 6px; font-size: 0.9rem; color: #444; }}
pre {{ overflow: auto; background: #f7f7f7; border: 1px solid #ddd; padding: 10px; white-space: pre-wrap; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #bbb; padding: 6px 10px; text-align: left; }}
.alternatives section {{ min-width: 0; }}
.outcome-region_repair {{ background: #1a7f37; color: #fff; padding: 2px 8px; border-radius: 4px; }}
.outcome-partial_field_salvage {{ background: #9a6700; color: #fff; padding: 2px 8px; border-radius: 4px; }}
.outcome-geometry_only {{ background: #57606a; color: #fff; padding: 2px 8px; border-radius: 4px; }}
.outcome-underdetermined {{ background: #cf222e; color: #fff; padding: 2px 8px; border-radius: 4px; }}
details.rejected summary {{ color: #cf222e; font-weight: bold; cursor: pointer; }}
.rejection-banner {{ border: 2px solid #cf222e; color: #cf222e; padding: 4px 8px; margin: 4px 0; font-weight: bold; }}
</style>
</head>
<body>
<header>
<h1>Fragment realignment evidence report</h1>
<p>Geometry was discovered and hashed before OCR or reviewed annotations were joined.
All before/after OCR pairs use the same native crop, engine instance, raw pass,
sparse pass, parser, and settings.</p>
<p><a href="results.jsonl">Download results.jsonl</a> ·
<a href="run_manifest.json">Download run_manifest.json</a></p>
<table>
<tbody>
{''.join(f'<tr><th>{html.escape(str(key))}</th><td>{value}</td></tr>' for key, value in summary.items())}
</tbody>
</table>
<h2>Frozen run contract</h2>
{_json_pre(manifest["frozen_parameters"])}
</header>
{''.join(cards)}
</body>
</html>
"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def _versions(engine: ocr.OcrEngine) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pymupdf": getattr(pymupdf, "VersionBind", "unknown"),
        "ocr_engine": f"{type(engine).__module__}.{type(engine).__name__}",
        "platform": platform.platform(),
    }
    try:
        if type(engine).__name__ == "PytesseractEngine":
            import pytesseract

            versions["tesseract"] = str(pytesseract.get_tesseract_version())
        elif type(engine).__name__ == "TesserocrEngine":
            import tesserocr

            versions["tesserocr"] = str(tesserocr.tesseract_version())
    except Exception as exc:
        versions["ocr_version_probe"] = f"unavailable: {type(exc).__name__}"
    return versions


def _geometry_core_contract() -> dict[str, Any]:
    """Reader-facing snapshot of the bounds frozen by the core source hash."""
    return {
        "page_region_proposals": {
            "max_regions": 12,
            "small_page_full_region_max_dimension": 512,
            "dark_threshold": 145,
            "maximum_region_fraction_of_page": 0.45,
            "maximum_region_span_of_either_page_dimension": 0.70,
            "edge_footer_rejection": (
                "reject shallow edge-touching horizontal or vertical clusters"
            ),
            "minimum_region_width": "max(60, page_width // 24)",
            "minimum_region_height": "max(45, page_height // 30)",
        },
        "seam_solver": {
            "minimum_fragment_width": "max(4, partition_size // 24)",
            "seam_span_pixels": 3,
            "maximum_absolute_offset": (
                "min(max(16, round(repair_size * 0.04)), "
                "max(8, repair_size * 0.20), 96)"
            ),
            "minimum_absolute_offset": (
                "max(2, min(8, ceil(repair_size * 0.02)))"
            ),
            "edge_gradient_threshold": 10,
            "offset_penalty_weight": 0.35,
            "minimum_occupancy_support": 0.55,
            "minimum_similarity_after": 0.82,
            "minimum_similarity_gain": 0.06,
            "minimum_edge_support": 0.02,
            "minimum_edge_similarity_after": 0.35,
            "minimum_edge_gain": 0.08,
            "maximum_seam_hypotheses": 8,
            "seam_nonmaximum_suppression_radius": 2,
        },
        "row_pitch_solver": {
            "period_bounds": "8 <= period <= repair_size // 3",
            "maximum_distinct_pitches": 2,
            "bin_width": "max(2, pitch // 4)",
            "maximum_absolute_offset": "min(96, pitch * 3)",
            "minimum_row_comb_gain": 0.08,
            "minimum_distinct_row_support_per_fragment": 3,
            "minimum_textlike_components": (
                "25 when region area >= 50000 pixels; otherwise 10"
            ),
            "minimum_textlike_share_of_occupancy": 0.25,
            "viterbi_state_magnitude_penalty": 0.004,
            "viterbi_state_change_magnitude_penalty": 0.018,
            "viterbi_state_change_penalty": 0.12,
            "common_skew_normalization": (
                "nearest-neighbor rectification with source-pixel provenance"
            ),
            "residual_skew_abstention": (
                "reject small monotonic offset ramps explained by common angle"
            ),
        },
        "candidate_gate": {
            "fragment_count": "2..max_fragments, with max_fragments <= 12",
            "complexity_penalty_per_extra_fragment": 0.015,
            "overlap_and_uncovered_penalty_weight": 0.35,
            "minimum_total_gain_exclusive": 0.035,
        },
    }


def evaluate_pdfs(
    pdf_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    max_fragments: int = 5,
    top_k: int = 8,
    family_mode: str = "all_family_diagnostic",
    engine: ocr.OcrEngine | None = None,
) -> dict[str, Any]:
    """Evaluate whole PDFs and write a source-faithful evidence package."""
    inputs = [Path(path).expanduser().resolve() for path in pdf_paths]
    if not inputs:
        raise ValueError("at least one PDF is required")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input PDFs not found: {', '.join(missing)}")
    if not 2 <= max_fragments <= 12:
        raise ValueError("max_fragments must be between 2 and 12")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if family_mode not in fragment_realign._FAMILY_MODES:
        raise ValueError(
            f"family_mode must be one of {fragment_realign._FAMILY_MODES}"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    total_started = time.monotonic()
    runs, input_records, geometry_seconds = _discover_geometry(
        inputs,
        destination,
        max_fragments=max_fragments,
        top_k=top_k,
        family_mode=family_mode,
    )

    # This digest is the freeze boundary: it covers source hashes, every tested
    # scan page/rotation, every candidate transform/score, and frozen PNG hashes.
    # The diagnostics sidecar is additive metadata and stays outside the digest
    # so legacy freeze hashes remain comparable across schema-compatible runs.
    geometry_freeze_payload = [
        {
            key: value
            for key, value in run.items()
            if key not in {"geometry_elapsed_seconds", "candidate_diagnostics"}
        }
        for run in runs
    ]
    geometry_freeze_sha256 = _sha256_bytes(
        _canonical_json(geometry_freeze_payload).encode("utf-8")
    )
    (destination / "geometry_freeze.json").write_text(
        json.dumps(
            {
                "geometry_freeze_sha256": geometry_freeze_sha256,
                "runs": geometry_freeze_payload,
            },
            default=_json_default,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    shared_engine = engine if engine is not None else ocr.default_engine()
    ocr_seconds = _evaluate_ocr(runs, destination, engine=shared_engine)
    _join_reporting_annotations(runs)

    summary = _summary(runs)
    source_hashes = {
        "evaluator": _sha256_file(Path(__file__).resolve()),
        "geometry_core": _sha256_file(Path(fragment_realign.__file__).resolve()),
    }
    manifest = {
        "schema_version": int(getattr(fragment_realign, "_SCHEMA_VERSION", 1)),
        "purpose": "offline laboratory evaluation; no production integration",
        "inputs": input_records,
        "geometry_freeze_sha256": geometry_freeze_sha256,
        "annotations_joined_after_geometry_freeze": True,
        "frozen_parameters": {
            "rotations_k_ccw": [0, 1, 2, 3],
            "rotation_convention": "np.rot90; positive k is counter-clockwise",
            "max_fragments": max_fragments,
            "top_k": top_k,
            "family_mode": family_mode,
            "ambiguity_margin_total_gain": AMBIGUITY_MARGIN,
            "ocr_passes": ["raw psm", "sparse psm"],
            "same_region_before_after": True,
            "geometry_core": _geometry_core_contract(),
            "geometry_core_binding": (
                "the code_state_sha256.geometry_core digest binds this reader-facing "
                "snapshot to the exact implementation used"
            ),
            "acceptance_gates": [
                "pixel_score_after > pixel_score_before and total_gain > 0",
                "repaired reading score > baseline reading score",
                "at least one parser-valid field is new or changed",
                "visible glyph pixels intersect a non-zero fragment",
                "source/destination provenance is reciprocal",
                "no parser-valid conflict inside ambiguity margin",
            ],
        },
        "versions": _versions(shared_engine),
        "code_state_sha256": source_hashes,
        "timing_seconds": {
            "geometry": geometry_seconds,
            "ocr_and_parse": ocr_seconds,
            "total": time.monotonic() - total_started,
        },
        "summary": summary,
        "outputs": [
            "geometry_freeze.json",
            "results.jsonl",
            "run_manifest.json",
            "report.html",
            "assets/",
            "records/",
        ],
    }
    _write_json_records(destination, runs)
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_html(destination, runs, manifest)
    return manifest


def _expand_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.pdf")))
        else:
            paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a whole-PDF fragment realignment evidence report.",
    )
    parser.add_argument("pdfs", nargs="+", help="PDF paths or directories containing PDFs")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for report.html, JSON, and lossless PNG assets",
    )
    parser.add_argument("--max-fragments", type=int, default=5, choices=range(2, 13))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--family-mode",
        default="all_family_diagnostic",
        choices=list(fragment_realign._FAMILY_MODES),
        help="Candidate family selection passed to the geometry core",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = evaluate_pdfs(
        _expand_inputs(args.pdfs),
        args.output_dir,
        max_fragments=args.max_fragments,
        top_k=args.top_k,
        family_mode=args.family_mode,
    )
    print(json.dumps(manifest["summary"], sort_keys=True))
    print(f"report: {Path(args.output_dir).resolve() / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
