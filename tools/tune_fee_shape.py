#!/usr/bin/env python3
"""Synthetic threshold sweep for the fee-value shape classifier (spec §9.2).

Renders every vocabulary word through the corpus receipt geometry, applies a
degradation grid spanning the measured corpus profile (JPEG q~58, blur, skew,
sensor noise) and beyond it, plus adversarial probes (cross-vocabulary words,
forged amounts, junk strings, noise, blank), and reports the score/margin
distributions of correct vs. wrong candidate reads. Legality: no corpus gold
is read here — tuning data is 100% synthetic (spec §9.2).

Usage: .venv/bin/python tools/tune_fee_shape.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib_pipeline import fee_shape  # noqa: E402

RNG = np.random.default_rng(8090)


def render_page(status, amount, waiver, dpi=576):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 48), "MIB Fee Receipt", fontsize=14, fontname="hebo")
    for label, value, y in (("Case ID", "MIB-000123", 133),
                            ("Fee Status", status, 157),
                            ("Amount", amount, 181),
                            ("Waiver Code", waiver, 205)):
        page.insert_text((88, y - 2), label, fontsize=8, fontname="hebo")
        page.insert_text((238, y), value, fontsize=9, fontname="helv")
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width).copy()
    doc.close()
    return gray


def degrade(gray, jpeg_q, blur, skew, noise):
    img = gray.copy()
    if skew:
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        img = np.clip(img.astype(np.float32)
                      + RNG.normal(0, noise, img.shape), 0, 255
                      ).astype(np.uint8)
    if jpeg_q:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


GRID = list(itertools.product((None, 58, 40), (0.0, 1.5, 3.0, 4.5),
                              (0.0, 0.8), (0.0, 12.0)))

# Adversarial value strings the classifier must never read as vocabulary.
STATUS_PROBES = ("$809.00", "void", "PAID STAMP COPY", "unknwn pald",
                 "waived unknown")
AMOUNT_PROBES = ("$500.00", "$909.00", "$809.99", "$80.00", "$8090.00",
                 "$0.01", "809.00")


def sweep():
    correct, wrong = [], []
    reject_kinds = {"abstain_genuine": 0, "abstain_probe": 0, "wrong": 0,
                    "ok": 0}
    for (q, blur, skew, noise) in GRID:
        kw = dict(jpeg_q=q, blur=blur, skew=skew, noise=noise)
        for word in fee_shape.STATUS_VOCAB:
            crops = fee_shape.crop_regions(degrade(
                render_page(word, "$809.00", "N/A"), **kw))
            read = fee_shape.classify_status(crops["fee_status"])
            _tally(read, word, correct, wrong, reject_kinds, kw, "status")
        for word in fee_shape.AMOUNT_VOCAB:
            crops = fee_shape.crop_regions(degrade(
                render_page("paid", word, "N/A"), **kw))
            read = fee_shape.classify_amount(crops["fee_amount"])
            _tally(read, word, correct, wrong, reject_kinds, kw, "amount")
        for word in fee_shape.WAIVER_VOCAB:
            crops = fee_shape.crop_regions(degrade(
                render_page("waived", "$0.00", word), **kw))
            read = fee_shape.classify_waiver(crops["waiver_code"])
            _tally(read, word, correct, wrong, reject_kinds, kw, "waiver")
        for probe in STATUS_PROBES:
            crops = fee_shape.crop_regions(degrade(
                render_page(probe, "$809.00", "N/A"), **kw))
            read = fee_shape.classify_status(crops["fee_status"])
            _tally(read, None, correct, wrong, reject_kinds, kw,
                   f"status-probe:{probe}")
        for probe in AMOUNT_PROBES:
            crops = fee_shape.crop_regions(degrade(
                render_page("paid", probe, "N/A"), **kw))
            read = fee_shape.classify_amount(crops["fee_amount"])
            _tally(read, None, correct, wrong, reject_kinds, kw,
                   f"amount-probe:{probe}")
    # unstructured probes
    for _ in range(20):
        crop = RNG.integers(0, 255, (168, 900), dtype=np.uint8)
        read = fee_shape.classify_status(crop)
        _tally(read, None, correct, wrong, reject_kinds, {}, "noise")
    return correct, wrong, reject_kinds


def _tally(read, expected, correct, wrong, kinds, kw, tag):
    if read is None:
        kinds["abstain_genuine" if expected else "abstain_probe"] += 1
        return
    if expected is not None and read.value == expected:
        kinds["ok"] += 1
        correct.append((read.score, read.margin, tag, kw))
    else:
        kinds["wrong"] += 1
        wrong.append((read.score, read.margin, read.value, tag, kw))


def main():
    correct, wrong, kinds = sweep()
    print(f"tally: {kinds}")
    total_genuine = kinds["ok"] + kinds["abstain_genuine"]
    print(f"genuine accept rate at current bars "
          f"(SCORE_MIN={fee_shape.SCORE_MIN}, MARGIN_MIN={fee_shape.MARGIN_MIN}): "
          f"{kinds['ok']}/{total_genuine}")
    if correct:
        cs = sorted(c[0] for c in correct)
        cm = sorted(c[1] for c in correct)
        print(f"correct accepts: score min/p5 {cs[0]:.3f}/"
              f"{cs[len(cs)//20]:.3f}  margin min/p5 {cm[0]:.3f}/"
              f"{cm[len(cm)//20]:.3f}")
    if wrong:
        print("WRONG ACCEPTS (must be empty):")
        for w in wrong:
            print("   ", w)
    else:
        print("wrong accepts: NONE across the full grid")


if __name__ == "__main__":
    main()
