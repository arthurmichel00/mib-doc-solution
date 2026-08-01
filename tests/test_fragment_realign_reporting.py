from __future__ import annotations

from pathlib import Path

import numpy as np

from tools import evaluate_fragment_realign


def test_partial_field_recovery_does_not_claim_full_page_alignment() -> None:
    checks = {
        "geometry_score_improves": True,
        "reading_score_improves": True,
        "new_or_changed_parser_valid_value": True,
        "moved_glyph_support": True,
        "reciprocal_provenance": True,
        "no_parser_ambiguity": True,
    }

    outcome = evaluate_fragment_realign._report_candidate_outcome(
        checks,
        ambiguous_fields=[],
    )

    assert outcome["field_recovery_status"] == "pixel_supported"
    assert outcome["full_page_geometry_status"] == "unverified"
    assert outcome["classification"] == "underdetermined"
    assert (
        outcome["acceptance_reason"]
        == "partial field recovery is pixel-supported; it does not prove "
        "full-page alignment"
    )


def test_nonaccepted_candidates_preserve_existing_abstention_levels() -> None:
    geometry_only = {
        "geometry_score_improves": True,
        "reading_score_improves": False,
        "new_or_changed_parser_valid_value": False,
        "moved_glyph_support": True,
        "reciprocal_provenance": True,
        "no_parser_ambiguity": True,
    }
    geometry_failed = {
        **geometry_only,
        "geometry_score_improves": False,
    }

    geometry_only_outcome = evaluate_fragment_realign._report_candidate_outcome(
        geometry_only,
        ambiguous_fields=[],
    )
    geometry_failed_outcome = evaluate_fragment_realign._report_candidate_outcome(
        geometry_failed,
        ambiguous_fields=[],
    )

    assert geometry_only_outcome == {
        "field_recovery_status": "underdetermined",
        "full_page_geometry_status": "unverified",
        "classification": "underdetermined",
        "acceptance_reason": (
            "abstained: geometry improved but symmetric OCR/parser score did not"
        ),
    }
    assert geometry_failed_outcome == {
        "field_recovery_status": "pixel_unsupported",
        "full_page_geometry_status": "pixel_unsupported",
        "classification": "pixel_unsupported",
        "acceptance_reason": (
            "rejected: geometry score or reciprocal provenance gate failed"
        ),
    }


def test_html_lead_names_field_and_full_page_statuses_separately() -> None:
    lead = evaluate_fragment_realign._status_lead_html(
        {
            "classification": "underdetermined",
            "field_recovery_status": "pixel_supported",
            "full_page_geometry_status": "unverified",
            "acceptance_reason": (
                "partial field recovery is pixel-supported; it does not prove "
                "full-page alignment"
            ),
        }
    )

    assert "<strong>Overall: underdetermined</strong>" in lead
    assert "Field recovery: <strong>pixel_supported</strong>" in lead
    assert "Full-page geometry: <strong>unverified</strong>" in lead
    assert "does not prove full-page alignment" in lead


def test_ocr_evaluation_copies_partial_recovery_statuses_to_run(monkeypatch) -> None:
    run = {
        "page_index": 2,
        "candidates": [
            {
                "assets": {
                    "crop_before": "before.png",
                    "crop_after": "after.png",
                },
                "pixel_score_before": 0.0,
                "pixel_score_after": 0.8,
                "score_terms": {"total_gain": 0.8},
                "moved_glyph_pixels": 24,
                "provenance_contract": {
                    "reciprocal_visible_mapping": True,
                },
            }
        ],
    }

    def fake_read_gray(_output_dir: Path, relative: str) -> np.ndarray:
        value = 0 if relative == "before.png" else 1
        return np.full((1, 1), value, dtype=np.uint8)

    def fake_ocr_and_parse(
        image: np.ndarray,
        *,
        page_index: int,
        engine: object,
    ) -> dict[str, object]:
        del page_index, engine
        recovered = int(image[0, 0]) == 1
        return {
            "parsed": {
                "values": {"sponsor_id": "SPN-1680"} if recovered else {},
            },
            "reading_score": 2.0 if recovered else 0.0,
        }

    monkeypatch.setattr(evaluate_fragment_realign, "_read_gray", fake_read_gray)
    monkeypatch.setattr(
        evaluate_fragment_realign,
        "_ocr_and_parse",
        fake_ocr_and_parse,
    )

    evaluate_fragment_realign._evaluate_ocr(
        [run],
        Path("."),
        engine=object(),
    )

    assert run["field_recovery_status"] == "pixel_supported"
    assert run["full_page_geometry_status"] == "unverified"
    assert run["classification"] == "underdetermined"


def test_ocr_evaluation_selects_later_pixel_supported_candidate_without_reordering(
    monkeypatch,
) -> None:
    def candidate(rank: int) -> dict[str, object]:
        return {
            "rank": rank,
            "assets": {
                "crop_before": "before.png",
                "crop_after": f"after-{rank}.png",
            },
            "pixel_score_before": 0.0,
            "pixel_score_after": 0.8,
            "score_terms": {"total_gain": 1.0 - rank / 100.0},
            "moved_glyph_pixels": 24,
            "provenance_contract": {
                "reciprocal_visible_mapping": True,
            },
        }

    run = {
        "page_index": 2,
        "candidates": [candidate(1), candidate(3)],
    }

    def fake_read_gray(_output_dir: Path, relative: str) -> np.ndarray:
        value = 3 if relative == "after-3.png" else 0
        return np.full((1, 1), value, dtype=np.uint8)

    def fake_ocr_and_parse(
        image: np.ndarray,
        *,
        page_index: int,
        engine: object,
    ) -> dict[str, object]:
        del page_index, engine
        recovered = int(image[0, 0]) == 3
        return {
            "parsed": {
                "values": {"sponsor_id": "SPN-1680"} if recovered else {},
            },
            "reading_score": 2.0 if recovered else 0.0,
        }

    monkeypatch.setattr(evaluate_fragment_realign, "_read_gray", fake_read_gray)
    monkeypatch.setattr(
        evaluate_fragment_realign,
        "_ocr_and_parse",
        fake_ocr_and_parse,
    )

    evaluate_fragment_realign._evaluate_ocr(
        [run],
        Path("."),
        engine=object(),
    )

    assert [item["rank"] for item in run["candidates"]] == [1, 3]
    assert run["candidates"][0]["field_recovery_status"] == "underdetermined"
    assert run["candidates"][1]["field_recovery_status"] == "pixel_supported"
    assert run["selected_candidate_rank"] == 3
    assert (
        run["candidate_selection_reason"]
        == "selected the highest-ranked candidate with pixel-supported field recovery"
    )
    assert run["field_recovery_status"] == "pixel_supported"
    assert run["full_page_geometry_status"] == "unverified"
    assert run["classification"] == "underdetermined"


def test_html_labels_selected_candidate_and_rejected_alternatives_clearly(
    tmp_path: Path,
) -> None:
    asset_names = (
        "full_after",
        "partition_overlay",
        "crop_before",
        "crop_after",
        "absolute_difference",
        "glyph_mask",
        "rule_mask",
        "structure_mask",
        "source_to_destination",
        "destination_to_source",
        "overlap_mask",
        "uncovered_mask",
        "cropped_source_mask",
    )

    def candidate(
        rank: int,
        *,
        field_recovery_status: str,
        acceptance_reason: str,
    ) -> dict[str, object]:
        return {
            "rank": rank,
            "assets": {name: f"{name}-{rank}.png" for name in asset_names},
            "classification": "underdetermined",
            "field_recovery_status": field_recovery_status,
            "full_page_geometry_status": "unverified",
            "acceptance_reason": acceptance_reason,
            "region": [0, 0, 10, 10],
            "partition_axis": "y",
            "partition_angle_degrees": 0.0,
            "fragments": [],
            "pixel_score_before": 0.1,
            "pixel_score_after": 0.8,
            "score_terms": {"total_gain": 0.7},
            "moved_glyph_pixels": 10,
            "overlap_pixels": 0,
            "uncovered_pixels": 0,
            "cropped_source_pixels": 0,
            "provenance_contract": {"reciprocal_visible_mapping": True},
            "acceptance_checks": {},
            "baseline_ocr": {"parsed": {"values": {}}},
            "repaired_ocr": {"parsed": {"values": {}}},
        }

    run = {
        "record_id": "example",
        "classification": "underdetermined",
        "field_recovery_status": "pixel_supported",
        "full_page_geometry_status": "unverified",
        "acceptance_reason": "partial field recovery",
        "selected_candidate_rank": 3,
        "candidate_selection_reason": (
            "selected the highest-ranked candidate with pixel-supported field recovery"
        ),
        "assets": {
            "full_original": "original.png",
            "full_before": "before.png",
        },
        "rotation_human": "0",
        "rotation_k_ccw": 0,
        "source_pdf_sha256": "source",
        "rendered_page_sha256": "rendered",
        "rotated_view_sha256": "rotated",
        "geometry_candidate_count": 2,
        "geometry_elapsed_seconds": 1.0,
        "ambiguous_fields": [],
        "evaluation_annotation": None,
        "candidates": [
            candidate(
                1,
                field_recovery_status="underdetermined",
                acceptance_reason="abstained",
            ),
            candidate(
                3,
                field_recovery_status="pixel_supported",
                acceptance_reason="partial field recovery",
            ),
        ],
    }

    evaluate_fragment_realign._write_html(
        tmp_path,
        [run],
        {"summary": {}, "frozen_parameters": {}},
    )
    report = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "Selected field-recovery candidate — geometry rank 3" in report
    assert "Alternative — geometry rank 1" in report
    assert "Field recovery: <strong>underdetermined</strong>" in report
    assert "Alternative — geometry rank 3" not in report
