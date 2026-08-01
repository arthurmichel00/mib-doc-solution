#!/usr/bin/env python3
"""Build the focused, post-probe fragment-realignment review gallery."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytesseract


SOLUTION_ROOT = Path(__file__).resolve().parent.parent
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from mib_pipeline.fragment_realign import (  # noqa: E402
    FragmentTransform,
    RepairCandidate,
    apply_fragment_transforms,
)


@dataclass(frozen=True)
class ReviewCase:
    number: int
    case_id: str
    page: int
    rotation: str
    source_name: str
    mode: str
    status: str
    note: str
    axis: str | None = None
    region: tuple[int, int, int, int] | None = None
    transforms: tuple[FragmentTransform, ...] = ()
    display_region: tuple[int, int, int, int] | None = None
    deskew_degrees: float = 0.0


SOURCE_ROOT = Path(
    "/private/tmp/claude-501/"
    "-Users-arthurmichel-ASM-PROJECTS-8090/"
    "06194f12-2116-4383-ab1c-fa79fa770ac5/"
    "scratchpad/fragment-realign-v3/assets"
)

CASES = (
    ReviewCase(
        2,
        "MIB-000027",
        2,
        "0 degrees",
        "p000-MIB-000027-page002-rot0-full-before.png",
        "automatic local candidate",
        "automatic OCR recovery confirmed",
        "Vertical form rails identify the sponsor cut at page y=590; "
        "the lower band receives inverse dx=+214. Symmetric OCR changes from "
        "no exact sponsor value to SPN-1345.",
        "y",
        (0, 270, 2448, 910),
        (
            FragmentTransform((0, 320)),
            FragmentTransform((320, 640), inverse_dx=214),
        ),
        (0, 0, 1320, 640),
    ),
    ReviewCase(
        3,
        "MIB-000063",
        3,
        "0 degrees",
        "p001-MIB-000063-page003-rot0-full-before.png",
        "automatic local candidate",
        "automatic OCR recovery confirmed",
        "Vertical form rails identify the sponsor cut at page y=478; "
        "the lower band receives inverse dx=+14. Symmetric OCR changes from "
        "no exact sponsor value to SPN-1680.",
        "y",
        (0, 158, 2448, 798),
        (
            FragmentTransform((0, 320)),
            FragmentTransform((320, 640), inverse_dx=14),
        ),
        (0, 0, 1500, 640),
    ),
    ReviewCase(
        4,
        "MIB-000063",
        5,
        "0 degrees, then +2.0 degree deskew attempt",
        "p001-MIB-000063-page005-rot0-full-before.png",
        "diagnostic hard attempt",
        "underdetermined",
        "The bands are slanted and several white/gray pasted rectangles overwrite "
        "source pixels. These line-state offsets are a hard attempt, not an "
        "automatically accepted reconstruction.",
        "y",
        (0, 0, 2448, 1100),
        (
            FragmentTransform((0, 472)),
            FragmentTransform((472, 604), inverse_dx=-5),
            FragmentTransform((604, 674), inverse_dx=-19),
            FragmentTransform((674, 762), inverse_dx=128),
            FragmentTransform((762, 874), inverse_dx=99),
            FragmentTransform((874, 953), inverse_dx=61),
            FragmentTransform((953, 1025)),
            FragmentTransform((1025, 1100), inverse_dx=-39),
        ),
        (0, 0, 1550, 1050),
        2.0,
    ),
    ReviewCase(
        5,
        "MIB-000178",
        6,
        "0 degrees",
        "p002-MIB-000178-page006-rot0-full-before.png",
        "automatic local candidate",
        "automatic OCR recovery confirmed",
        "Vertical form rails identify the visa-row cut at page y=556; "
        "the lower band receives inverse dx=-36. Symmetric OCR changes from "
        "no exact visa value to Visa Class: XW-1.",
        "y",
        (0, 236, 2448, 876),
        (
            FragmentTransform((0, 320)),
            FragmentTransform((320, 640), inverse_dx=-36),
        ),
        (0, 0, 1350, 640),
    ),
    ReviewCase(
        10,
        "MIB-000931",
        5,
        "90 degrees clockwise",
        "p003-MIB-000931-page005-rot3-full-before.png",
        "diagnostic eight-strip prototype",
        "prototype works; automatic full-chain discovery failed",
        "The prototype uses the seven line seams found in the field block and "
        "non-rule continuity to choose row-pitch aliases. It is intentionally "
        "kept separate from the automatic-success count.",
        "x",
        (461, 0, 1211, 720),
        (
            FragmentTransform((0, 132), inverse_dy=0),
            FragmentTransform((132, 243), inverse_dy=34),
            FragmentTransform((243, 317), inverse_dy=-14),
            FragmentTransform((317, 419), inverse_dy=-70),
            FragmentTransform((419, 486), inverse_dy=17),
            FragmentTransform((486, 542), inverse_dy=98),
            FragmentTransform((542, 630), inverse_dy=126),
            FragmentTransform((630, 750), inverse_dy=-10),
        ),
        (0, 0, 750, 720),
    ),
    ReviewCase(
        11,
        "MIB-000977",
        1,
        "0 degrees",
        "p005-MIB-000977-page001-rot0-full-before.png",
        "automatic negative control",
        "correct abstention",
        "No strip candidate was emitted in any of the four tested rotations. "
        "Ordinary fine deskew remains the correct operation.",
    ),
)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(image)):
        raise RuntimeError(f"failed to write {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deskew(image: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0.0:
        return image
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), degrees, 1.0
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _full_after(
    source: np.ndarray,
    region: tuple[int, int, int, int],
    candidate: RepairCandidate,
) -> np.ndarray:
    x0, y0, x1, y1 = region
    result = source.copy()
    result[y0:y1, x0:x1] = candidate.reconstruction
    return result


def _overlay(source: np.ndarray, candidate: RepairCandidate) -> np.ndarray:
    overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    height, width = source.shape
    for fragment in candidate.fragments[:-1]:
        boundary = fragment.interval[1]
        if candidate.partition_axis == "y":
            cv2.line(overlay, (0, boundary), (width - 1, boundary), (0, 0, 255), 2)
        else:
            cv2.line(overlay, (boundary, 0), (boundary, height - 1), (0, 0, 255), 2)
    for fragment in candidate.fragments:
        start, end = fragment.interval
        if candidate.partition_axis == "y":
            origin = (width // 2, (start + end) // 2)
        else:
            origin = ((start + end) // 2, height // 2)
        destination = (
            origin[0] + fragment.inverse_dx,
            origin[1] + fragment.inverse_dy,
        )
        cv2.arrowedLine(overlay, origin, destination, (0, 150, 0), 2, tipLength=0.2)
    return overlay


def _crop(image: np.ndarray, region: tuple[int, int, int, int] | None) -> np.ndarray:
    if region is None:
        return image
    x0, y0, x1, y1 = region
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def _ocr(image: np.ndarray) -> str:
    return pytesseract.image_to_string(image, config="--oem 1 --psm 6").strip()


def _mapping_is_reciprocal(candidate: RepairCandidate) -> bool:
    forward = candidate.source_to_destination_map.ravel()
    reverse = candidate.destination_to_source_map.ravel()
    sources = np.flatnonzero(forward >= 0)
    destinations = forward[sources]
    return bool(
        sources.size
        and np.all(destinations < reverse.size)
        and np.all(reverse[destinations] == sources)
    )


def _img(path: str, caption: str) -> str:
    return (
        "<figure><a href=\""
        + html.escape(path, quote=True)
        + "\"><img loading=\"lazy\" src=\""
        + html.escape(path, quote=True)
        + "\" alt=\""
        + html.escape(caption, quote=True)
        + "\"></a><figcaption>"
        + html.escape(caption)
        + "</figcaption></figure>"
    )


def build_gallery(output_dir: Path) -> None:
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    cards: list[str] = []

    for item in CASES:
        source_path = SOURCE_ROOT / item.source_name
        source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise FileNotFoundError(source_path)
        prefix = f"{item.number:02d}-{item.case_id.lower()}-p{item.page}"
        full_before_name = f"assets/{prefix}-full-before.png"
        shutil.copy2(source_path, output_dir / full_before_name)
        source_for_repair = _deskew(source, item.deskew_degrees)
        record: dict[str, object] = {
            "number": item.number,
            "case_id": item.case_id,
            "pdf_page": item.page,
            "rotation": item.rotation,
            "mode": item.mode,
            "status": item.status,
            "note": item.note,
            "source_sha256": _sha256(source_path),
            "transforms": [
                {
                    "interval": transform.interval,
                    "inverse_dx": transform.inverse_dx,
                    "inverse_dy": transform.inverse_dy,
                }
                for transform in item.transforms
            ],
        }

        images = [
            _img(full_before_name, "Full page before"),
        ]
        if item.region is None or item.axis is None:
            record["automatic_candidate_count_all_tested_rotations"] = 0
            tight_name = f"assets/{prefix}-unchanged.png"
            _write_png(output_dir / tight_name, source)
            images.append(_img(tight_name, "Unchanged: strip solver abstained"))
        else:
            x0, y0, x1, y1 = item.region
            source_crop = np.ascontiguousarray(source_for_repair[y0:y1, x0:x1])
            candidate = apply_fragment_transforms(
                source_crop,
                item.transforms,
                partition_axis=item.axis,
            )
            full_after = _full_after(source_for_repair, item.region, candidate)
            display_before = _crop(source_crop, item.display_region)
            display_after = _crop(candidate.reconstruction, item.display_region)
            display_overlay = _crop(_overlay(source_crop, candidate), item.display_region)
            paths = {
                "full_after": f"assets/{prefix}-full-after.png",
                "before": f"assets/{prefix}-before.png",
                "after": f"assets/{prefix}-after.png",
                "overlay": f"assets/{prefix}-overlay.png",
                "overlap": f"assets/{prefix}-overlap.png",
                "uncovered": f"assets/{prefix}-uncovered.png",
            }
            _write_png(output_dir / paths["full_after"], full_after)
            _write_png(output_dir / paths["before"], display_before)
            _write_png(output_dir / paths["after"], display_after)
            _write_png(output_dir / paths["overlay"], display_overlay)
            _write_png(
                output_dir / paths["overlap"],
                candidate.overlap_mask.astype(np.uint8) * 255,
            )
            _write_png(
                output_dir / paths["uncovered"],
                candidate.uncovered_mask.astype(np.uint8) * 255,
            )
            before_ocr = _ocr(display_before)
            after_ocr = _ocr(display_after)
            loss_fraction = float(
                (candidate.overlap_mask.sum() + candidate.uncovered_mask.sum())
                / candidate.reconstruction.size
            )
            record.update(
                {
                    "region": item.region,
                    "partition_axis": item.axis,
                    "reciprocal_visible_provenance": _mapping_is_reciprocal(candidate),
                    "loss_fraction": loss_fraction,
                    "before_ocr": before_ocr,
                    "after_ocr": after_ocr,
                    "assets": paths,
                }
            )
            images.extend(
                (
                    _img(paths["full_after"], "Full page after candidate"),
                    _img(paths["before"], "Tight before"),
                    _img(paths["after"], "Tight after"),
                    _img(paths["overlay"], "Detected/prototype seams and inverse shifts"),
                )
            )

        records.append(record)
        detail = {
            key: record[key]
            for key in (
                "mode",
                "status",
                "region",
                "partition_axis",
                "transforms",
                "reciprocal_visible_provenance",
                "loss_fraction",
                "before_ocr",
                "after_ocr",
            )
            if key in record
        }
        cards.append(
            "<article id=\"example-"
            + str(item.number)
            + "\"><h2>"
            + html.escape(
                f"{item.number}. {item.case_id} - PDF page {item.page} - {item.status}"
            )
            + "</h2><p><strong>"
            + html.escape(item.mode)
            + ".</strong> "
            + html.escape(item.note)
            + "</p><div class=\"grid\">"
            + "".join(images)
            + "</div><details><summary>Geometry, provenance, and OCR validation</summary><pre>"
            + html.escape(json.dumps(detail, indent=2))
            + "</pre></details></article>"
        )

    (output_dir / "focused-review.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Focused fragment realignment review</title>
<style>
body{font:16px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1550px;padding:24px;background:#f3f0e8;color:#181818}
header,article{background:#fff;border:1px solid #c9c3b8;border-radius:10px;padding:20px;margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
figure{margin:0;border:1px solid #ddd;padding:8px}img{display:block;width:100%;height:auto}
figcaption{margin-top:6px;color:#444}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}
a{color:#0645ad}
</style></head><body>
<header><h1>Focused fragment realignment review</h1>
<p>Examples #2, #3, #4, #5, and #10 from Arthur's review, plus clean
control #11. Automatic candidates and diagnostic prototypes are labeled
separately. Click any image for its lossless PNG. The JSON evidence is
available at <a href="focused-review.json">focused-review.json</a>.</p>
</header>
""" + "".join(cards) + "</body></html>"
    (output_dir / "focused-review.html").write_text(document, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    build_gallery(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
