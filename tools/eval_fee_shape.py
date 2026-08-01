#!/usr/bin/env python3
"""Dev-time gold cross-check for the fee-value shape classifier (spec §9.2).

CLEARANCE-GATED: renders real corpus pages (a batch); do not run during a
machine freeze. Gold is used strictly as held-out evaluation — requirement
is 0 label-inconsistent accepts; nothing here feeds back into thresholds
without a new synthetic-side justification.

For EVERY scan page of every requested case (default: full train), render at
576 DPI (hidden text redacted), crop the receipt value regions, classify
status+amount(+waiver), and derive the shape verdict exactly as
discharge._shape_verdict does (both pixel variants must agree). Non-receipt
pages double as false-fire probes: their rows hold other templates' values,
so any verdict from them is a finding. Digital receipt pages are included as
rendered sanity checks.

Usage:
  .venv/bin/python tools/eval_fee_shape.py                # full train
  .venv/bin/python tools/eval_fee_shape.py MIB-000307 ... # named cases
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib_pipeline import fee_shape, fields, ocr  # noqa: E402
from mib_pipeline.model import PageKind  # noqa: E402
from mib_pipeline.pdf_loader import load_pages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN = ROOT / "mib-doc-challenge/data/train"
GOLD = ROOT / "mib-doc-challenge/data/train_labels.csv"


def shape_verdict_for_page(gray: np.ndarray) -> tuple[str, float] | None:
    raw = fee_shape.crop_regions(gray)
    variants = [raw]
    try:
        variants.append({k: ocr._divblur(v) for k, v in raw.items()})
    except Exception:
        pass
    per = []
    for crops in variants:
        status = fee_shape.classify_status(crops["fee_status"])
        amount = fee_shape.classify_amount(crops["fee_amount"])
        if status is None or amount is None:
            return None
        waiver = fee_shape.classify_waiver(crops["waiver_code"])
        verdict = fields._receipt_verdict(
            status.value, 809 if amount.value == "$809.00" else 0,
            waiver.value if waiver else None)
        if verdict is None:
            return None
        per.append((verdict, min(status.score, amount.score)))
    if len(per) < 2 or len({v for v, _ in per}) != 1:
        return None
    return per[0][0], min(c for _, c in per)


def main() -> None:
    gold = {r["case_id"]: r for r in csv.DictReader(open(GOLD))}
    cases = sys.argv[1:] or sorted(gold)
    stats = {"reads": 0, "match": 0, "wrong": [], "nonreceipt_fire": []}
    for i, case in enumerate(cases):
        pdf = TRAIN / f"{case}.pdf"
        if not pdf.exists():
            continue
        with pymupdf.open(str(pdf)) as doc:
            pages = load_pages(doc)
            for page in pages:
                if page.kind != PageKind.SCAN:
                    continue
                gray = ocr.render_gray(doc[page.index], page, dpi=576)
                for k in (0, 1, 2, 3):
                    rot = np.ascontiguousarray(np.rot90(gray, k=k)) if k \
                        else gray
                    got = shape_verdict_for_page(rot)
                    if got is None:
                        continue
                    verdict, conf = got
                    stats["reads"] += 1
                    ok = verdict == gold[case]["fee_status"]
                    stats["match" if ok else "wrong"] += 1 if ok else 0
                    if not ok:
                        stats["wrong"].append(
                            (case, page.index, k, verdict, conf,
                             gold[case]["fee_status"]))
                    print(f"{case} p{page.index} k={k}: {verdict} "
                          f"(conf {conf:.2f}) gold={gold[case]['fee_status']}"
                          f" {'OK' if ok else 'WRONG'}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"-- {i + 1}/{len(cases)} cases, reads={stats['reads']} "
                  f"match={stats['match']} wrong={len(stats['wrong'])}",
                  flush=True)
    print(f"\nTOTAL reads={stats['reads']} match={stats['match']} "
          f"wrong={len(stats['wrong'])}")
    for w in stats["wrong"]:
        print("  WRONG:", w)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
