#!/usr/bin/env python3
"""Dev harness: run the pipeline on train PDFs and analyze against labels.

Usage:
    python tools/run_train.py --pdf-dir ../carved/data/train \
        --labels ../mib-doc-challenge/data/train_labels.csv \
        --out /tmp/preds.jsonl [--limit 50] [--cases MIB-000116,...]

Reports adjudication confusion, catastrophic false approvals, per-field
extraction accuracy (evaluator normalization), and per-decision-path stats.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIELDS = [
    "applicant_name", "species_code", "home_world", "visa_class",
    "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
    "fee_status",
]
WEIGHTS = {
    "applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
    "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3,
    "risk_flags": 8, "fee_status": 4,
}


def normalize(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def normalize_flags(value: str) -> str:
    text = normalize(value)
    if text in ("", "none", "null", "unknown"):
        return "none"
    parts = sorted(p.strip() for p in text.split("|") if p.strip())
    return "|".join(parts)


def _init_worker() -> None:
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    import cv2

    cv2.setNumThreads(1)


def _process(pdf_path: str) -> dict:
    from mib_pipeline.pipeline import process_pdf

    start = time.time()
    row = process_pdf(pdf_path)
    row["_secs"] = round(time.time() - start, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cases")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    labels = {r["case_id"]: r for r in csv.DictReader(open(args.labels))}
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if args.cases:
        wanted = set(args.cases.split(","))
        pdfs = [p for p in pdfs if p.stem in wanted]
    if args.limit:
        pdfs = pdfs[: args.limit]

    t0 = time.time()
    rows: list[dict] = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool, \
            open(args.out, "w") as out, open(args.out + ".dev", "w") as dev:
        for row in pool.imap_unordered(_process, [str(p) for p in pdfs], chunksize=4):
            rows.append(row)
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            out.write(json.dumps(public, sort_keys=True) + "\n")
            dev.write(json.dumps(row, sort_keys=True) + "\n")
    wall = time.time() - t0
    secs = [r["_secs"] for r in rows]
    print(f"\n{len(rows)} packets in {wall:.1f}s wall "
          f"({wall/len(rows):.2f} s/pdf wall, {sum(secs)/len(secs):.2f} s/pdf cpu)")

    confusion: Counter = Counter()
    path_stats: dict[str, Counter] = defaultdict(Counter)
    field_hits: Counter = Counter()
    field_misses: dict[str, list] = defaultdict(list)
    cfa: list[str] = []
    for row in rows:
        truth = labels.get(row["case_id"])
        if truth is None:
            continue
        gold, pred, path = truth["adjudication"], row["adjudication"], row["_path"]
        confusion[(gold, pred)] += 1
        path_stats[path]["n"] += 1
        path_stats[path]["correct"] += int(gold == pred)
        path_stats[path][f"gold_{gold}"] += 1
        if gold == "DENIED" and pred == "APPROVED":
            cfa.append(row["case_id"])
        for fld in FIELDS:
            norm = normalize_flags if fld == "risk_flags" else normalize
            if norm(row[fld]) == norm(truth[fld]):
                field_hits[fld] += 1
            else:
                field_misses[fld].append((row["case_id"], row[fld], truth[fld]))

    n = sum(confusion.values())
    correct = sum(v for (g, p), v in confusion.items() if g == p)
    print(f"\nadjudication: {correct}/{n} = {correct/n:.3f}")
    print(f"catastrophic false approvals: {len(cfa)} {cfa[:10]}")
    print("\nconfusion (gold -> pred):")
    for (g, p), v in sorted(confusion.items()):
        marker = "  <-- CFA" if (g, p) == ("DENIED", "APPROVED") else ""
        print(f"  {g:13s} -> {p:13s} {v:4d}{marker}")

    print("\nper-path:")
    for path in sorted(path_stats):
        s = path_stats[path]
        golds = {k[5:]: v for k, v in s.items() if k.startswith("gold_")}
        print(f"  {path:26s} n={s['n']:4d} acc={s['correct']/s['n']:.3f} gold={golds}")

    print("\nextraction per field:")
    raw = sum(WEIGHTS[f] * field_hits[f] for f in FIELDS)
    for fld in FIELDS:
        print(f"  {fld:17s} {field_hits[fld]:4d}/{n} = {field_hits[fld]/n:.3f}")
    print(f"  weighted extraction ~= {50 * raw / (45 * n):.2f}/50")

    print("\nsample misses per field (up to 3):")
    for fld in FIELDS:
        for case_id, got, want in field_misses[fld][:3]:
            print(f"  {fld:17s} {case_id} got={got!r} want={want!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
