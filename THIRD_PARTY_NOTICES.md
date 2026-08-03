# Third-Party Notices

This repository's own code is MIT-licensed (see LICENSE). It depends on
third-party components with their own licenses, installed at Docker build time:

| Component | License | Note |
|---|---|---|
| PyMuPDF (fitz) | **AGPL-3.0** | PDF rendering. AGPL-3.0 governs the combined work when this software is distributed or run as a network service. Users requiring different terms need a commercial PyMuPDF license from Artifex. |
| Tesseract OCR / tesserocr | Apache-2.0 / MIT | OCR engine + binding |
| RapidOCR (rapidocr-onnxruntime) | Apache-2.0 | PP-OCR ONNX runtime reader |
| onnxruntime | MIT | inference runtime |
| OpenCV (opencv-python) | Apache-2.0 | image processing |
| NumPy | BSD-3-Clause | arrays |
| scikit-learn | BSD-3-Clause | calibration fitting (dev-time) |

Two pieces of third-party material are redistributed with the submission
and carry their own sections below: `mib_pipeline/absynth.py`, a port, and
`models/tessdata/mib.traineddata`, a fine-tuned Tesseract LSTM committed
under `models/` so a fresh clone builds the submitted image. Apart from
those, no third-party source
is vendored in this repository; all components are fetched from PyPI/apt
at image build, and the remaining model files under models/ are
project-generated artifacts. Design ideas adapted from other public
solutions, where no third-party material is redistributed, are recorded
separately in `ATTRIBUTION.md`.

## Adapted mechanisms

Several mechanisms in `mib_pipeline/` are our own implementations of ideas
and algorithms taken from MIT-licensed public solutions to the same
challenge. No source file is copied; `ATTRIBUTION.md` records, per
mechanism, what was taken and where our variant deviates.

`mib_pipeline/row_restore.py` carried an entry here while it was a port.
It no longer is: the module was reimplemented from our own two-rail
specification and shares no expression with any third-party source, so no
third-party licence governs it. `ATTRIBUTION.md` keeps a courtesy note
recording the public solutions whose corpus evidence informed the decision
to build it.

### Other adapted solutions

The remaining adapted mechanisms (closed-menu CTC scoring and joint-name
decode from the "moonshots" solution; fuzzy Finding recovery and the
rotation probe from tylergibbs1; reason templates from kirtandesai;
fusion edit costs from mib-intake and balawal; the sponsor digit vote from
handemanai) are likewise MIT-licensed and credited per mechanism in
`ATTRIBUTION.md`.

---

The two sections that follow are self-contained: each names the artifact,
where it sits in the image, where it came from, and reproduces that work's
license. The bundled runtime dependencies close the document.

## Adapted MIT-licensed solution source

`mib_pipeline/absynth.py` is a port. Its layout constants, the
label-anchored analysis-by-synthesis method, the self-calibrating
per-row degradation fit, the two-anchor registration cross-check, and the
common-canvas NCC decode are all derived from:

- **luke-harriman / mib-doc-challenge-solution**
  (<https://github.com/luke-harriman/mib-doc-challenge-solution>), files
  `lib/absynth2.py` and `lib/mfr_reader.py`, at commit `820b0cd`.

The MIT licence requires the copyright notice to travel with substantial
portions of the work, so it is reproduced in full:

```
MIT License

Copyright (c) 2026 Luke Harriman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Other public solutions listed in `ATTRIBUTION.md` (moonshots, tylergibbs1,
kirtandesai, balawal, handemanai, mib-intake) are all MIT-licensed and were
adapted at the level of ideas and thresholds rather than source, with our
own implementations; `ATTRIBUTION.md` records what came from where.

## `models/tessdata/mib.traineddata` — fine-tuned Tesseract LSTM

- **Upstream:** Shrey Shingala,
  `https://github.com/ShreyShingala/ocr-document-pipeline-challenge`
- **Upstream path:** `tessdata/mib.traineddata`
- **License:** MIT
- **Used:** verbatim and unmodified. Not retrained — the upstream project's
  own measurements show further fine-tuning on a guessed noise model made
  results worse, so the artifact is consumed exactly as published.
- **In the image:** `/app/models/tessdata/mib.traineddata`, loaded by
  `mib_pipeline/ocr.py: tessft_engine()` as tesseract language `mib`,
  behind the `MIB_TESSFT` flag.
- **SHA-256:**
  `2e75e40c35abb10c1a7126f509025a4de7dcbebd76049a334e43c69233d6f9ac`
- **Size:** 15,400,601 bytes
- **Base model, from the artifact's own version string:**
  `4.00.00alpha:eng:synth20170629:[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`
  — upstream `tessdata_best` English (float, `Lfx512`), fine-tuned. It
  carries the inherited `lstm-word-dawg` / `lstm-punc-dawg` /
  `lstm-number-dawg`. The engine our image ships as stock is the same
  `eng:synth20170629` lineage but the int-quantized `Lfx192` build, which
  is why the fine-tuned pass measures ~2x the stock pass's time.
  Tesseract version used for the fine-tune is not documented upstream; the
  file is standard traineddata and loads on 5.x (verified on 5.5.1 here,
  and their own image runs Debian slim's 5.3.x, the same family as ours).
- **Provenance note:** this binary is committed to this repository under
  `models/tessdata/`, exactly like the PP-OCR ONNX weights beside it, so
  `docker build` on a fresh clone reproduces the submitted image with no
  out-of-band download step. (In our private development workspace the
  `models/` tree is gitignored and the artifact is vendored onto disk before
  the build; here it is version-controlled.) The Dockerfile's existing
  `COPY models /app/models` picks it up, and the SHA-256 above identifies
  the correct file.

```
MIT License

Copyright (c) 2026 Shrey Shingala

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Bundled runtime dependencies

| Component | Licence | Use |
| --- | --- | --- |
| PyMuPDF | AGPL-3.0 | PDF parsing, span visibility filtering, page rendering, glyph rendering for synthesis |
| RapidOCR (PP-OCR ONNX) | Apache-2.0 | OCR on raster pages |
| Tesseract (via pytesseract) | Apache-2.0 | Second OCR engine |
| OpenCV (headless) | Apache-2.0 | Image restoration, template matching, correlation |
| ONNX Runtime | MIT | Inference backend for the bundled recognizers |
| NumPy | BSD-3-Clause | Numerics |
| Pillow | HPND | Image IO (transitive) |
| pyclipper, Shapely, six | MIT / BSD-3-Clause / MIT | RapidOCR transitive dependencies |

**AGPL note.** PyMuPDF is AGPL-3.0. Conveying a combined work that links it
obliges the conveyor to publish the complete corresponding source of that
work; this repository's submission does so.
