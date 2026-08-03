# Builder: compile tesserocr against Debian bookworm's tesseract 5.3
# (no manylinux wheels exist on PyPI). The runtime stage installs the
# matching tesseract-ocr runtime packages from the same distro.
FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS tesserocr-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ pkg-config libtesseract-dev libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir --wheel-dir /wheels tesserocr==2.8.0

FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
# libgl1 + libglib2.0-0: rapidocr-onnxruntime pulls in non-headless opencv-python,
# whose cv2 needs libGL.so.1 / libgthread-2.0 — absent from slim. Without these the
# fallback OCR ImportErrors under --network none (offline_ocr_check.py catches it).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY --from=tesserocr-build /wheels /tmp/wheels
RUN pip install --no-cache-dir -r /app/requirements.txt /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Scoring runs with --read-only and a 2 GiB tmpfs on /tmp: every cache and
# temp path must resolve under /tmp, and imports must never try to write
# bytecode into /app.
ENV HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    OMP_THREAD_LIMIT=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MIB_RAPID_MODEL=v6 \
    MIB_REASON_ADJ=1 \
    MIB_STAMP_RESCUE=1 \
    MIB_SNAPFIX=1 \
    MIB_CTCFILL=1 \
    MIB_ABSYNTH=1 \
    MIB_CTCFILL_FUSION=1 \
    MIB_TESSFT=1

COPY run.sh solution.py THIRD_PARTY_NOTICES.md /app/
COPY mib_pipeline /app/mib_pipeline
COPY models /app/models
COPY tools/offline_ocr_check.py /app/tools/offline_ocr_check.py
RUN chmod +x /app/run.sh && python -m compileall -q /app

WORKDIR /app
ENTRYPOINT ["/app/run.sh"]
