#!/usr/bin/env python3
"""Prove the fallback recognizer initializes and reads with NO network.

Run inside the container under --network none (docker_check.sh does this):
it draws a synthetic text image, initializes RapidOCR, and asserts a read.
Any attempt to download models would crash here instead of during scoring.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

sys.path.insert(0, "/app")
sys.path.insert(0, ".")

from mib_pipeline import ocr  # noqa: E402


def main() -> int:
    img = np.full((120, 640), 255, dtype=np.uint8)
    cv2.putText(img, "Sponsor ID: SPN-1234", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    lines = ocr.rapid_lines(img, page_index=0)
    text = " ".join(l.text for l in lines)
    if "1234" not in text:
        print(f"FAIL: fallback OCR read {text!r}", file=sys.stderr)
        return 1
    print(f"offline fallback OCR OK: {text!r}")

    from mib_pipeline import crnn  # noqa: E402

    crnn_out = crnn.crnn_lines(img, page_index=0)
    if not crnn_out:
        print("FAIL: CRNN produced no lines on the synthetic probe",
              file=sys.stderr)
        return 1
    if any(l.tier1_ok for l in crnn_out):
        print("FAIL: CRNN lines must carry tier1_ok=False", file=sys.stderr)
        return 1
    print(f"offline CRNN OK: {crnn_out[0].text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
