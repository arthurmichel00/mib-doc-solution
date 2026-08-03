# MIB Document Challenge — Submission Pipeline (arthurmichel00)

An offline, deterministic Docker pipeline for the 8090 MIB Doc Challenge: it extracts fields from
damaged/scanned application PDFs and adjudicates each case (APPROVED / DENIED / NEEDS_REVIEW) with a
calibrated confidence. No runtime LLMs or VLMs, no network access, no per-case hardcoding. Hidden PDF
spans are redacted before rasterization, pages are typed digital-vs-scan, scans go through a measured
OCR ensemble (multi-pass Tesseract ladder, bundled RapidOCR ONNX fallback, bounded preprocessing
escalation, a strip-scoped fine-tuned Tesseract LSTM, a small candidate-trained CRNN line reader, and
two candidate-scoring channels over the closed-menu fields — CTC scoring of every legal value against
the recognizer's frame posteriors, fused across views and pages, plus a generator-inversion reader that
renders each candidate through a recovered degradation kernel), and a rule-based policy engine with
positive-evidence gates makes the final call. The whole thing runs inside the official scoring contract: 4 vCPUs, 8 GiB RAM,
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
| Full train (1,000 cases), official Docker scoring | **129.14 / 150** |
| Fixed 200-case holdout (generalization check) | **130.57 / 150** |
| — Field extraction | 45.51 / 50 |
| — Classification | 66.67 / 80 |
| — Calibration | 16.96 / 20 (Brier 0.0760) |
| Runtime | 5.87 s/PDF at 4 vCPUs (budget 6.0), 0.36 GiB image |

Holdout ≥ train at every measured milestone; the single catastrophic false approval is a documented
designed trap (MIB-000865) at every milestone. The runtime figure is the quiet-box `docker_check`
measurement on its fixed 100-PDF subset: the previous build ran at 5.81 s/PDF, and the two reading
channels added on this build cost ~0.06 s/PDF between them — disclosed rather than hidden. End-to-end
over the full 1,000-case corpus the same image runs 5.01 s/PDF; the stricter figure is the one quoted.
Full methodology, per-change measurements, failure modes, and red-team hardening are in
[APPENDIX.md](APPENDIX.md); the per-lever record and every kill reason are in [LEVERS.md](LEVERS.md).

## How this was built

Flag-gated AI-agent loops, with verification in code rather than review. Every candidate lever was built
behind a feature flag and A/B-measured on the full training set against five gates: score ≥ baseline,
holdout ≥ train, a single designed-trap catastrophic false approval, zero fallback rows, and a green
red-team corpus. More than forty levers were built and measured; seventeen shipped enabled, and the kill
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
