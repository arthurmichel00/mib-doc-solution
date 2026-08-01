#!/usr/bin/env python3
"""Fit per-decision-path calibration constants from a dev run.

Usage:
    python tools/fit_calibration.py --dev /tmp/preds.jsonl.dev \
        --labels ../mib-doc-challenge/data/train_labels.csv \
        [--holdout-seed 8090 --holdout 200]

Reads the dev dump written by run_train.py (rows include _path), splits
train into fit/holdout with a fixed seed, and reports per-path accuracy
with shrinkage toward the global rate:

    q_hat = (correct + m * q_global) / (n + m),  m = 10

Prints a PATH_STATS block ready to paste into mib_pipeline/calibration.py
(fit on ALL cases for shipping) plus honest holdout numbers from the
fit-split-only estimates. Pinned paths keep their pin; only accuracy and
error splits are refit.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

SHRINKAGE_M = 10
CLASSES = ("APPROVED", "DENIED", "NEEDS_REVIEW")


def path_table(rows: list[dict], labels: dict[str, dict]) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        truth = labels.get(row["case_id"])
        if truth is None:
            continue
        stats = table[row["_path"]]
        stats["n"] += 1
        stats["correct"] += int(row["adjudication"] == truth["adjudication"])
        if row["adjudication"] != truth["adjudication"]:
            stats[f"err_{truth['adjudication']}"] += 1
    return table


def shrunk(stats: Counter, q_global: float) -> float:
    return (stats["correct"] + SHRINKAGE_M * q_global) / (stats["n"] + SHRINKAGE_M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--holdout-seed", type=int, default=8090)
    args = ap.parse_args()

    labels = {r["case_id"]: r for r in csv.DictReader(open(args.labels))}
    rows = [json.loads(line) for line in open(args.dev)]
    rows = [r for r in rows if r["case_id"] in labels]

    ids = sorted(labels)
    rng = random.Random(args.holdout_seed)
    holdout_ids = set(rng.sample(ids, args.holdout))
    fit_rows = [r for r in rows if r["case_id"] not in holdout_ids]
    hold_rows = [r for r in rows if r["case_id"] in holdout_ids]

    fit_table = path_table(fit_rows, labels)
    q_global = sum(s["correct"] for s in fit_table.values()) / max(
        1, sum(s["n"] for s in fit_table.values()))

    print(f"# fit on {len(fit_rows)} cases, global acc {q_global:.3f}")
    print("# holdout check (fit-split estimates applied to holdout):")
    hold_table = path_table(hold_rows, labels)
    brier_sum, n_hold = 0.0, 0
    for row in hold_rows:
        q = shrunk(fit_table[row["_path"]], q_global)
        correct = row["adjudication"] == labels[row["case_id"]]["adjudication"]
        brier_sum += (q - correct) ** 2
        n_hold += 1
    hold_acc = sum(s["correct"] for s in hold_table.values()) / max(1, n_hold)
    print(f"#   holdout n={n_hold} acc={hold_acc:.3f} "
          f"mean_brier={brier_sum / max(1, n_hold):.4f} "
          f"calibration~={20 * max(0, 1 - 2 * brier_sum / max(1, n_hold)):.2f}/20")

    full_table = path_table(rows, labels)
    q_global_full = sum(s["correct"] for s in full_table.values()) / len(rows)
    print("\n# paste into calibration.py (fit on all 1000):")
    for path in sorted(full_table):
        s = full_table[path]
        q = shrunk(s, q_global_full)
        errs = {c: s[f"err_{c}"] for c in CLASSES if s[f"err_{c}"]}
        total_err = sum(errs.values()) or 1
        split = ", ".join(f"{c}: {v / total_err:.2f}" for c, v in errs.items())
        print(f'    "{path}": n={s["n"]:4d} acc={q:.3f} err_split={{{split}}}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
