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

No third-party source is vendored in this repository; all components are
fetched from PyPI/apt at image build. Model files under models/ are
project-generated artifacts.
