"""Candidate-trained CRNN line recognizer — last-resort escalation engine.

A ~6M-parameter CRNN+CTC trained at dev time purely on synthetic renders of
the corpus's base-14 fonts under measured damage profiles (q~58 JPEG at
144 DPI, bold stroke-fusion, 2D multi-ghost) plus weak-labeled real lines
where Tesseract and PP-OCR independently agree. Reserved-crop gate accuracy
of the shipped checkpoint: 24/53 case-fields (45.3%) on lines BOTH shipped
engines misread — it only ever runs after them, so a hit is additive.

Trust boundary, measured then enforced:
- Lines carry tier1_ok=False — field reads only, never a Finding or stamp
  (a 45%-accurate engine must not fabricate tier-1 evidence).
- sponsor_id is muzzled entirely at the pipeline layer: planted revoked-ID
  decoys make sponsor the one field where a sub-trusted read could flip an
  adjudication for the wrong reason.
- The pre-known restore guard limits fills to fields no other engine read;
  the one exception is the narrowly-scoped name-challenge (pipeline.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from .model import Line, Source
from . import ocr

# SPONSOR MUZZLE, applied at the source: CRNN never emits a sponsor-bearing
# line, so no later reconcile pass (weld, re-collects) can pick a CRNN
# sponsor up — planted revoked-ID decoys make sponsor the one field where a
# sub-trusted misread could flip an adjudication for the wrong reason.
_SPONSOR_RE = re.compile(r"sponsor|spn[\s\-–—._:]*[0-9]", re.I)

_SESSION = None
_META = None
_CONF_SCALE = 0.85  # cross-engine normalization, same axis as PP-OCR's


def _load():
    global _SESSION, _META
    if _SESSION is None:
        import onnxruntime as ort

        base = ocr._models_dir() / "crnn"
        _META = json.loads((base / "meta.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _SESSION = ort.InferenceSession(str(base / "crnn.onnx"),
                                        sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    return _SESSION, _META


def _decode(logits: np.ndarray, charset: str, blank: int) -> tuple[str, float]:
    """CTC greedy decode with a mean-probability confidence."""
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    probs = exp / exp.sum(-1, keepdims=True)
    ids = probs.argmax(-1)
    chars, confs, prev = [], [], -1
    for t, i in enumerate(ids):
        if i != prev and i != blank:
            chars.append(charset[i - 1] if 0 < i <= len(charset) else "")
            confs.append(float(probs[t, i]))
        prev = i
    if not chars:
        return "", 0.0
    return "".join(chars), float(np.mean(confs))


def _read_crop(gray: np.ndarray) -> tuple[str, float]:
    session, meta = _load()
    h = meta["height"]
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        return "", 0.0
    w = max(32, int(gray.shape[1] * h / gray.shape[0]))
    img = cv2.resize(gray, (w, h))
    x = (1.0 - img.astype(np.float32) / 255.0)[None, None]
    logits = session.run(["logits"], {"image": x})[0][0]
    return _decode(logits, meta["charset"], meta["blank"])


def _tess_line_rows(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Line crops from Tesseract layout boxes — the gate-crop geometry.

    Tesseract's layout analysis boxes rows its recognizer garbles AND rows
    PP-OCR's detector misses outright (fused-bold pages box zero det rows);
    the reserved gate set was cut from exactly these boxes, so this source
    restores gate/production crop parity (measured: det-only crops lost 3
    of the gate's recoveries to never-boxed rows). tight-box: contiguous
    words until a gap > 3 line-heights, then extend right ~26 heights so
    washed value pixels stay in-crop.
    """
    import pytesseract

    out = []
    for psm in (6, 11):
        try:
            d = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DICT,
                config=f"--oem 1 --psm {psm}")
        except Exception:
            continue
        lines: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
        for i, txt in enumerate(d["text"]):
            if not txt.strip():
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            x, y, w, h = (d["left"][i], d["top"][i],
                          d["width"][i], d["height"][i])
            lines.setdefault(key, []).append((x, y, x + w, y + h))
        for ws in lines.values():
            ws = sorted(ws)
            h = max(4, max(y1 - y0 for _, y0, _, y1 in ws))
            kept = [ws[0]]
            for w in ws[1:]:
                if w[0] - kept[-1][2] > 3 * h:
                    break
                kept.append(w)
            pad = 6
            out.append((max(0, kept[0][0] - pad),
                        max(0, min(w[1] for w in kept) - pad),
                        min(gray.shape[1], max(w[2] for w in kept) + 26 * h),
                        min(gray.shape[0], max(w[3] for w in kept) + pad)))
    return out


def crnn_lines(gray: np.ndarray, page_index: int) -> list[Line]:
    """Read detected text rows with the candidate-trained recognizer.

    Row boxes come from BOTH the bundled PP-OCR detector and Tesseract's
    layout boxes (each survives damage the other misses); each crop is
    preprocessed exactly like the training/gate pipeline (grayscale,
    aspect-preserving resize, inversion) — preprocessing drift between the
    gate and production was a measured failure class.
    """
    boxes: list[tuple[int, int, int, int]] = []
    try:
        det, _ = ocr._rapid_reader()(gray, use_det=True, use_cls=False,
                                     use_rec=False)
        for box in det or []:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            boxes.append((max(0, min(xs) - 2), max(0, min(ys) - 2),
                          min(gray.shape[1], max(xs) + 2),
                          min(gray.shape[0], max(ys) + 2)))
    except Exception:
        pass
    boxes.extend(_tess_line_rows(gray))

    rows: list[tuple[float, float, str, float]] = []
    seen: set[tuple[int, int]] = set()
    for x0, y0, x1, y1 in boxes:
        key = (round((y0 + y1) / 18), round(x0 / 30))
        if key in seen:
            continue
        seen.add(key)
        text, conf = _read_crop(gray[y0:y1, x0:x1])
        if not text.strip() or _SPONSOR_RE.search(text):
            continue
        rows.append(((y0 + y1) / 2.0, float(x0), text.strip(),
                     conf * _CONF_SCALE))
    rows.sort(key=lambda r: (r[0], r[1]))
    lines: list[Line] = []
    current: list[tuple[float, float, str, float]] = []
    for row in rows:
        if current and abs(row[0] - current[0][0]) > 18:
            lines.append(_merge_row(current, page_index))
            current = []
        current.append(row)
    if current:
        lines.append(_merge_row(current, page_index))
    return lines


def _merge_row(row: list[tuple[float, float, str, float]],
               page_index: int) -> Line:
    row.sort(key=lambda r: r[1])
    text = " ".join(r[2] for r in row)
    conf = float(np.mean([r[3] for r in row]))
    return Line(text=text, page_index=page_index, source=Source.OCR,
                conf=conf, tier1_ok=False)
