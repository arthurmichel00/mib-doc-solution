#!/usr/bin/env bash
# End-to-end Docker contract check: rebuild the image, run the EXACT scoring
# invocation from DOCKER_SUBMISSION.md on a 100-PDF train subset, validate
# the predictions, and report image size + per-PDF wall time.
#
# Usage: tools/docker_check.sh [n_pdfs]   (default 100)
# Re-run this after any pipeline change before considering it Docker-safe.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHALLENGE_DIR="$(cd "$SOLUTION_DIR/../mib-doc-challenge" && pwd)"
TRAIN_DIR="$CHALLENGE_DIR/data/train"
N="${1:-100}"
IMAGE="${MIB_IMAGE:-mib-submission}"
WORK="$(mktemp -d /tmp/mib-docker-check.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
SUBSET="$WORK/input"
OUTDIR="$WORK/output"
mkdir -p "$SUBSET" "$OUTDIR"

echo "== build =="
docker build -t "$IMAGE" "$SOLUTION_DIR"

SIZE_BYTES=$(docker image inspect "$IMAGE" --format '{{.Size}}')
SIZE_GIB=$(python3 -c "print(f'{$SIZE_BYTES/2**30:.2f}')")
echo "image size: ${SIZE_GIB} GiB (limit 4.00)"
python3 -c "import sys; sys.exit(0 if $SIZE_BYTES <= 4*2**30 else 1)" \
    || { echo "FAIL: image exceeds 4 GiB"; exit 1; }

echo "== subset: first $N train PDFs =="
# NB: `ls | head` under pipefail dies of SIGPIPE; printf's output fits the
# pipe buffer, and the shell glob is already sorted.
while IFS= read -r f; do cp "$f" "$SUBSET/"; done \
    < <(printf '%s\n' "$TRAIN_DIR"/*.pdf | head -"$N")
COPIED=$(ls "$SUBSET" | wc -l | tr -d ' ')
[ "$COPIED" -eq "$N" ] || { echo "FAIL: subset has $COPIED PDFs, wanted $N"; exit 1; }

echo "== offline fallback-OCR init check (--network none) =="
docker run --rm --network none --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=2g \
    --entrypoint python "$IMAGE" /app/tools/offline_ocr_check.py \
    || { echo "FAIL: fallback OCR needs network or cannot read"; exit 1; }

echo "== scoring invocation (exact contract flags) =="
START=$(date +%s)
docker run --rm --network none --cpus 4 --memory 8g --pids-limit 512 \
    --read-only --tmpfs /tmp:rw,nosuid,nodev,size=2g \
    --mount "type=bind,src=$SUBSET,dst=/input,readonly" \
    --mount "type=bind,src=$OUTDIR,dst=/output" \
    "$IMAGE" /input /output/predictions.jsonl
ELAPSED=$(( $(date +%s) - START ))

LINES=$(wc -l < "$OUTDIR/predictions.jsonl" | tr -d ' ')
echo "wall time: ${ELAPSED}s for $N PDFs => $(python3 -c "print(f'{$ELAPSED/$N:.2f}')")s/PDF (budget 6.0)"
[ "$LINES" -eq "$N" ] || { echo "FAIL: $LINES predictions for $N PDFs"; exit 1; }

echo "== validate_submission.py =="
python3 "$CHALLENGE_DIR/scripts/validate_submission.py" \
    --submission "$OUTDIR/predictions.jsonl" \
    --pdf-dir "$SUBSET" --require-complete

echo "OK: docker contract check passed ($N PDFs, ${ELAPSED}s, ${SIZE_GIB} GiB image)"
