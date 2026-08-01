#!/usr/bin/env python3
"""MIB Doc Challenge entrypoint.

Usage: python solution.py <input_dir> <output_predictions_path>

Processes every MIB-######.pdf in input_dir with a 4-worker process pool
and writes one JSONL prediction per packet, flushing incrementally so
partial runs still score. Every expected case is guaranteed a row: packets
that raise fall back inside the worker, and packets whose worker dies
outright (segfault in native code, OOM kill) or stalls are retried in a
fresh pool and, failing that, written as calibrated NEEDS_REVIEW fallbacks
by the parent process.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

WORKERS = int(os.environ.get("MIB_WORKERS", "4"))
_CASE_STEM_RE = re.compile(r"^MIB-[0-9]{6}$")
# Rounds guard against a pathological PDF that kills its worker on every
# attempt; anything still unwritten after the last round gets a fallback.
_MAX_ROUNDS = 3
# Watchdog allowance per case and per round: generous vs the observed
# per-case ceiling (~29 s); catches native-code hangs that in-worker
# deadlines cannot interrupt.
_CASE_BUDGET_SECONDS = 240.0


def _init_worker() -> None:
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    import cv2

    cv2.setNumThreads(1)


def _process(pdf_path: str) -> dict:
    from mib_pipeline.pipeline import process_pdf

    return process_pdf(pdf_path)


def _fallback(case_id: str) -> dict:
    from mib_pipeline.pipeline import _fallback_row

    row = _fallback_row(case_id)
    row.pop("_path", None)
    row.pop("_discharge", None)
    return row


def _run_round(pdfs: list[Path], out, written: set[str]) -> None:
    """One pool round; tolerates worker death and stalls mid-round.

    Rows are written and flushed as they complete, so a broken pool or a
    round timeout never loses finished work — unfinished cases simply stay
    unwritten for the next round.
    """
    context = mp.get_context("spawn")
    budget = 60.0 + _CASE_BUDGET_SECONDS * max(1, len(pdfs)) / WORKERS
    with cf.ProcessPoolExecutor(WORKERS, mp_context=context,
                                initializer=_init_worker) as pool:
        futures = {pool.submit(_process, str(p)): p.stem for p in pdfs}
        try:
            for future in cf.as_completed(futures, timeout=budget):
                case_id = futures[future]
                try:
                    row = future.result()
                except Exception:
                    # ordinary failures already fall back inside the worker,
                    # so this future rode along with a dead worker; leave
                    # the case for a retry round
                    continue
                row.pop("_path", None)
                row.pop("_discharge", None)
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                written.add(case_id)
                if len(written) % 100 == 0:
                    print(f"{len(written)} packets done", file=sys.stderr)
        except (cf.TimeoutError, TimeoutError):
            for future in futures:
                future.cancel()


def main(input_dir: str, output_path: str) -> int:
    pdfs = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() == ".pdf" and _CASE_STEM_RE.match(p.stem)
    )
    if not pdfs:
        print(f"no case PDFs found in {input_dir}", file=sys.stderr)
        return 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with open(output_path, "w") as out:
        pending = pdfs
        for _ in range(_MAX_ROUNDS):
            if not pending:
                break
            try:
                _run_round(pending, out, written)
            except Exception as exc:
                print(f"pool round aborted: {exc!r}", file=sys.stderr)
            pending = [p for p in pending if p.stem not in written]
            if pending:
                print(f"retrying {len(pending)} unfinished packets",
                      file=sys.stderr)
        for p in pending:
            out.write(json.dumps(_fallback(p.stem), sort_keys=True) + "\n")
            out.flush()
            written.add(p.stem)
    print(f"wrote {len(written)} predictions to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
