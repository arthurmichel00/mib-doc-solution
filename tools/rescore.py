#!/usr/bin/env python3
"""Recompute decisions/confidences from a dev dump without re-running OCR.

Each decision path implies its policy label, so calibration/decision-layer
changes can be evaluated by replaying the stored paths through the current
mib_pipeline.decision code.

Usage:
    python tools/rescore.py --dev /tmp/preds.jsonl.dev --out /tmp/preds2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib_pipeline import decision  # noqa: E402


def policy_label_for(path: str) -> str:
    if path == "N0_note_conflict":
        return "NEEDS_REVIEW"
    if path.startswith("N0_note_"):
        return path.removeprefix("N0_note_").upper()
    # N1 template-rescue paths (MIB_REASON_ADJ=1) carry their label the
    # same way; without this branch an N1 row replays as NEEDS_REVIEW.
    if path.startswith("N1_reason_"):
        return path.removeprefix("N1_reason_").upper()
    if path.startswith(("R1_", "R2_", "R3_", "R4_", "R5_", "R6_", "R7_")):
        return "DENIED"
    if path.startswith("R11"):
        return "APPROVED"
    return "NEEDS_REVIEW"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w") as out:
        for line in open(args.dev):
            row = json.loads(line)
            path = row.pop("_path")
            row.pop("_secs", None)
            adjudication, confidence = decision.decide(policy_label_for(path), path)
            row["adjudication"] = adjudication
            row["confidence"] = round(confidence, 4)
            out.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
