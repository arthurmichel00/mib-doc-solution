# MIB Document Challenge — Submission Pipeline (arthurmichel00)

An offline, deterministic Docker pipeline for the 8090 MIB Doc Challenge: it extracts fields from
damaged/scanned application PDFs and adjudicates each case (APPROVED / DENIED / NEEDS_REVIEW) with a
calibrated confidence. No runtime LLMs or VLMs, no network access, no per-case hardcoding. Hidden PDF
spans are redacted before rasterization, pages are typed digital-vs-scan, scans go through a measured
OCR ensemble (multi-pass Tesseract ladder, bundled RapidOCR ONNX fallback, bounded preprocessing
escalation, a small candidate-trained CRNN line reader, and a constrained-candidate CTC scoring pass
over the closed-menu fields), and a rule-based policy engine with positive-evidence gates makes the
final call. The whole thing runs inside the official scoring contract: 4 vCPUs, 8 GiB RAM,
`--network none`, ≤ 6 s per PDF.

## How to run

```bash
docker build -t mib-submission .

mkdir -p /tmp/mib-output
docker run --rm --network none \
  --mount type=bind,src="/absolute/path/to/input-pdfs",dst=/input,readonly \
  --mount type=bind,src="/tmp/mib-output",dst=/output \
  mib-submission /input /output/predictions.jsonl
```

The image accepts exactly two arguments (`<input_pdf_dir> <output_predictions_path>`); `run.sh` is the
entrypoint and follows the challenge README spec. The container satisfies the full
`DOCKER_SUBMISSION.md` contract (`--read-only`, `--network none`, `--cpus 4`, `--memory 8g`,
tmpfs `/tmp`) and performs no network fetches at scoring time.

## Results

| Metric | Value |
|---|---:|
| Full train (1,000 cases), official Docker scoring | **129.05 / 150** |
| Fixed 200-case holdout (generalization check) | **130.50 / 150** |
| — Field extraction | 45.42 / 50 |
| — Classification | 66.67 / 80 |
| — Calibration | 16.96 / 20 (Brier 0.0760) |
| Runtime | 5.81 s/PDF at 4 vCPUs (budget 6.0), 0.36 GiB image |

Holdout ≥ train at every measured milestone; the single catastrophic false approval is a documented
designed trap (MIB-000865) at every milestone. The runtime figure is the quiet-box `docker_check`
measurement: the previous build ran at 5.64 s/PDF, and the constrained-candidate pass added on this
build costs ~0.3 s on each of the ~20% of cases that trigger it — disclosed rather than hidden. Full
methodology, per-change measurements, failure modes, and red-team hardening are in `MEMO.md`.

## How this was built

Flag-gated AI-agent loops, with verification in code rather than review. Every candidate lever was built
behind a feature flag and A/B-measured on the full training set against five gates: score ≥ baseline,
holdout ≥ train, a single designed-trap catastrophic false approval, zero fallback rows, and a green
red-team corpus. More than thirty levers were built and measured; fourteen shipped enabled, and the kill
reason for every rejected one is recorded in [LEVERS.md](LEVERS.md). Humans entered at exactly three decision
points: goal-setting, trust-boundary extensions, and ship calls. Evidence detail: [APPENDIX.md](APPENDIX.md).

## Honesty statement

- The corpus's hidden "answer key" text is adversarial and is **never read**: hidden and injected PDF
  content is stripped before rasterization, so no downstream step can resurrect it.
- No answer-key or gold-label lookups exist anywhere in the runtime path; adjudication is driven only
  by affirmatively-read document evidence.
- Under-determined cases — silent flags, unreadable fees, missing outcome-determinative fields — are
  deliberately abstained to NEEDS_REVIEW rather than guessed.

## License

The repository's own code is **MIT-licensed** (see `LICENSE`). PyMuPDF is AGPL-3.0 and governs the
combined work — see `THIRD_PARTY_NOTICES.md` for the full third-party license table.
