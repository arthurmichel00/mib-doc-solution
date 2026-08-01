"""Diagnostic-only green-APPROVED-stamp counter (A7 follow-up).

The counter must (1) detect the green stamp's saturated-green hollow
geometry and nothing else, (2) render in RGB only digital note pages,
(3) log to stderr + the diagnostics file, never into row content, and
(4) be adjudication-inert by construction: returns None, mutates neither
the document nor the loaded pages, and swallows every error except the
case-deadline TimeoutError. Corpus-level byte-identity of emitted rows is
verified operationally (research/17-a7-stamp-verify.md follow-up run);
these tests pin the mechanism.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from mib_pipeline import diagnostics
from mib_pipeline.model import Line, Page, PageKind, Source
from mib_pipeline.pdf_loader import load_pages

# The stamp's ink and the two component geometries the detector was
# calibrated on (100 dpi): box outline 227x97 fill .08, word 154x47 fill .24.
_GREEN = (19 / 255, 137 / 255, 19 / 255)
_RED = (200 / 255, 30 / 255, 30 / 255)
_PALE_GREEN = (120 / 255, 160 / 255, 120 / 255)   # sat ~64, below the gate


# --------------------------------------------------------------------------
# helpers

def _blank_rgb(rows: int = 1100, cols: int = 850) -> np.ndarray:
    return np.full((rows, cols, 3), 255, dtype=np.uint8)


def _paint_stamp(img: np.ndarray, rgb=(19, 137, 19), x0: int = 500,
                 y0: int = 150) -> None:
    """Hollow 227x97 box (3px stroke) + a 154x47 striped word block,
    matching the calibrated component geometry and fill windows."""
    x1, y1 = x0 + 227, y0 + 97
    img[y0:y0 + 3, x0:x1] = rgb
    img[y1 - 3:y1, x0:x1] = rgb
    img[y0:y1, x0:x0 + 3] = rgb
    img[y0:y1, x1 - 3:x1] = rgb
    # word block: vertical strokes 4px on / 8px off -> the 9x9 close merges
    # them into one component (bbox 148x47, fill ~0.35) the way it merges
    # the real stamp's glyphs, inside the calibrated 0.04-0.40 fill window
    wx, wy = x0 + 37, y0 + 25
    for sx in range(wx, wx + 154, 12):
        img[wy:wy + 47, sx:sx + 4] = rgb


def _note_pdf(stamp_color=None, header: str = "Manual Adjudicator Note"):
    """One-page digital PDF with a note header and an optional stamp."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), header, fontsize=14, color=(0, 0, 0))
    page.insert_text((72, 100), "Finding: APPROVED. Reason: Clean packet.",
                     fontsize=10, color=(0, 0, 0))
    if stamp_color is not None:
        rect = pymupdf.Rect(360, 108, 360 + 163, 108 + 70)
        page.draw_rect(rect, color=stamp_color, width=2.5)
        page.insert_text((rect.x0 + 26, rect.y0 + 45), "APPROVED",
                         fontsize=26, color=stamp_color)
    return doc


def _zero_counts() -> dict:
    return {"cases": 0, "note_pages": 0, "review_cases": 0,
            "review_pages": 0, "stamped_cases": 0}


@pytest.fixture(autouse=True)
def _isolated_diag(tmp_path, monkeypatch):
    """Route the diagnostics file to tmp and zero the per-process tallies."""
    monkeypatch.setenv("MIB_STAMP_DIAG_FILE", str(tmp_path / "diag.log"))
    monkeypatch.setattr(diagnostics, "_counts", _zero_counts())
    yield tmp_path / "diag.log"


def _scan_pdf(stamp_rgb=None):
    """One-page PDF whose page is a full-bleed JPEG raster (PageKind.SCAN),
    mirroring the corpus encoding (/DeviceRGB DCTDecode)."""
    import cv2
    img = _blank_rgb()
    if stamp_rgb is not None:
        _paint_stamp(img, rgb=stamp_rgb)
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=buf.tobytes())
    return doc


# --------------------------------------------------------------------------
# detector: color + geometry gates

class TestStampComponents:
    def test_calibrated_stamp_geometry_is_detected(self):
        img = _blank_rgb()
        _paint_stamp(img)
        hits = diagnostics.stamp_components(img)
        assert len(hits) == 2  # box outline + word block, never merged
        assert {h["bbox"][2] for h in hits} == {227, 148}

    def test_red_decoy_same_geometry_is_ignored(self):
        img = _blank_rgb()
        _paint_stamp(img, rgb=(200, 30, 30))
        assert diagnostics.stamp_components(img) == []

    def test_desaturated_green_smudge_is_ignored(self):
        img = _blank_rgb()
        _paint_stamp(img, rgb=(120, 160, 120))   # sat ~64 < 150 gate
        assert diagnostics.stamp_components(img) == []

    def test_solid_green_block_fails_the_hollow_fill_gate(self):
        img = _blank_rgb()
        img[150:247, 500:727] = (19, 137, 19)    # fill 1.0
        assert diagnostics.stamp_components(img) == []

    def test_sub_threshold_speck_is_ignored(self):
        img = _blank_rgb()
        img[150:160, 500:530] = (19, 137, 19)    # 300 px < 400 floor
        assert diagnostics.stamp_components(img) == []

    def test_blank_page_short_circuits(self):
        assert diagnostics.stamp_components(_blank_rgb()) == []


# --------------------------------------------------------------------------
# page targeting: RGB rendering is limited to digital note pages

class TestDigitalNoteIndexes:
    def _page(self, kind: PageKind, header: str, index: int = 0) -> Page:
        page = Page(index=index, kind=kind)
        page.lines = [Line(text=header, page_index=index,
                           source=Source.DIGITAL, conf=0.99)]
        return page

    def test_digital_note_page_is_selected(self):
        page = self._page(PageKind.DIGITAL, "Manual Adjudicator Note")
        assert diagnostics._digital_note_indexes([page]) == [0]

    def test_scan_page_is_never_rendered_even_if_note_like(self):
        page = self._page(PageKind.SCAN, "Manual Adjudicator Note")
        assert diagnostics._digital_note_indexes([page]) == []

    def test_other_digital_doc_types_are_skipped(self):
        page = self._page(
            PageKind.DIGITAL,
            "FORM I-8090: Extraterrestrial Work Authorization Intake")
        assert diagnostics._digital_note_indexes([page]) == []

    def test_classification_probe_does_not_set_doc_type(self):
        page = self._page(PageKind.DIGITAL, "Manual Adjudicator Note")
        diagnostics._digital_note_indexes([page])
        assert page.doc_type is None


# --------------------------------------------------------------------------
# the hook: logging, counters, and inertness

class TestLogStampScan:
    def test_stamped_note_logs_case_line_and_counts(self, _isolated_diag,
                                                    capsys):
        doc = _note_pdf(stamp_color=_GREEN)
        pages = load_pages(doc)
        assert diagnostics.log_stamp_scan(doc, pages, "MIB-000005") is None
        err = capsys.readouterr().err
        assert "[stamp-diag] MIB-000005 green-APPROVED stamp detected" in err
        payload = json.loads(err.split("detected: ", 1)[1])
        assert payload["pages"] == [0] and payload["components"] >= 1
        assert payload["source"] == "note_page"
        assert diagnostics._counts == dict(
            _zero_counts(), cases=1, note_pages=1, stamped_cases=1)
        assert "MIB-000005" in _isolated_diag.read_text()

    def test_unstamped_note_is_rendered_but_silent(self, _isolated_diag,
                                                   capsys):
        doc = _note_pdf(stamp_color=None)
        diagnostics.log_stamp_scan(doc, load_pages(doc), "MIB-000024")
        assert "stamp detected" not in capsys.readouterr().err
        assert diagnostics._counts == dict(
            _zero_counts(), cases=1, note_pages=1)
        assert not _isolated_diag.exists()

    def test_red_decoy_note_is_silent(self, capsys):
        doc = _note_pdf(stamp_color=_RED)
        diagnostics.log_stamp_scan(doc, load_pages(doc), "MIB-000099")
        assert "stamp detected" not in capsys.readouterr().err
        assert diagnostics._counts["stamped_cases"] == 0

    def test_pale_green_forgery_is_silent(self, capsys):
        doc = _note_pdf(stamp_color=_PALE_GREEN)
        diagnostics.log_stamp_scan(doc, load_pages(doc), "MIB-000098")
        assert diagnostics._counts["stamped_cases"] == 0

    def test_non_note_case_renders_nothing(self):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Sponsor Attestation Letter",
                         fontsize=14, color=(0, 0, 0))
        diagnostics.log_stamp_scan(doc, load_pages(doc), "MIB-000097")
        assert diagnostics._counts == dict(_zero_counts(), cases=1)

    def test_hook_mutates_neither_doc_nor_pages(self):
        doc = _note_pdf(stamp_color=_GREEN)
        before_pix = doc[0].get_pixmap(dpi=72).samples
        pages = load_pages(doc)
        lines_before = [(l.text, l.conf) for l in pages[0].lines]
        diagnostics.log_stamp_scan(doc, pages, "MIB-000005")
        assert pages[0].doc_type is None
        assert [(l.text, l.conf) for l in pages[0].lines] == lines_before
        assert doc[0].get_pixmap(dpi=72).samples == before_pix

    def test_errors_are_swallowed_and_logged(self, capsys):
        doc = _note_pdf(stamp_color=_GREEN)
        pages = load_pages(doc)
        doc.close()   # renders will now raise inside the hook
        diagnostics.log_stamp_scan(doc, pages, "MIB-000096")
        assert "scan error (ignored)" in capsys.readouterr().err

    def test_case_deadline_timeout_is_never_swallowed(self, monkeypatch):
        doc = _note_pdf(stamp_color=None)
        pages = load_pages(doc)

        def _boom(_pages):
            raise TimeoutError("case exceeded 180s deadline")

        monkeypatch.setattr(diagnostics, "_digital_note_indexes", _boom)
        with pytest.raises(TimeoutError):
            diagnostics.log_stamp_scan(doc, pages, "MIB-000095")

    def test_unwritable_diag_file_still_logs_to_stderr(self, monkeypatch,
                                                       capsys):
        monkeypatch.setenv("MIB_STAMP_DIAG_FILE",
                           "/nonexistent-dir/diag.log")
        doc = _note_pdf(stamp_color=_GREEN)
        diagnostics.log_stamp_scan(doc, load_pages(doc), "MIB-000094")
        assert "stamp detected" in capsys.readouterr().err


# --------------------------------------------------------------------------
# review sweep: NEEDS_REVIEW cases re-render every page in color, so a
# stamp on a SCANNED note (color-capable: corpus scans are /DeviceRGB
# JPEGs) is still counted

class TestLogStampScanReview:
    def _run(self, doc, case_id, tmp_path):
        path = tmp_path / f"{case_id}.pdf"
        doc.save(str(path))
        diagnostics.log_stamp_scan_review(str(path), case_id)
        return path

    def test_green_stamp_on_jpeg_scan_page_is_detected(self, tmp_path,
                                                       capsys):
        self._run(_scan_pdf(stamp_rgb=(19, 137, 19)), "MIB-000093", tmp_path)
        err = capsys.readouterr().err
        assert "MIB-000093 green-APPROVED stamp detected" in err
        assert json.loads(err.split("detected: ", 1)[1])["source"] == \
            "review_scan"
        assert diagnostics._counts == dict(
            _zero_counts(), review_cases=1, review_pages=1, stamped_cases=1)

    def test_stampless_scan_page_is_silent(self, tmp_path, capsys):
        self._run(_scan_pdf(stamp_rgb=None), "MIB-000092", tmp_path)
        assert "stamp detected" not in capsys.readouterr().err
        assert diagnostics._counts == dict(
            _zero_counts(), review_cases=1, review_pages=1)

    def test_red_decoy_on_scan_page_is_silent(self, tmp_path, capsys):
        self._run(_scan_pdf(stamp_rgb=(200, 30, 30)), "MIB-000091", tmp_path)
        assert "stamp detected" not in capsys.readouterr().err

    def test_unreadable_path_is_swallowed_and_logged(self, capsys):
        diagnostics.log_stamp_scan_review("/nonexistent/case.pdf",
                                          "MIB-000090")
        assert "review-scan error (ignored)" in capsys.readouterr().err

    def test_case_deadline_timeout_is_never_swallowed(self, tmp_path,
                                                      monkeypatch):
        path = tmp_path / "MIB-000089.pdf"
        _scan_pdf(stamp_rgb=None).save(str(path))

        def _boom(_doc, _index):
            raise TimeoutError("case exceeded 180s deadline")

        monkeypatch.setattr(diagnostics, "_render_rgb", _boom)
        with pytest.raises(TimeoutError):
            diagnostics.log_stamp_scan_review(str(path), "MIB-000089")

    def test_pathological_page_count_is_capped(self, tmp_path):
        doc = pymupdf.open()
        for _ in range(diagnostics._REVIEW_MAX_PAGES + 5):
            doc.new_page(width=612, height=792)
        self._run(doc, "MIB-000088", tmp_path)
        assert diagnostics._counts["review_pages"] == \
            diagnostics._REVIEW_MAX_PAGES


# --------------------------------------------------------------------------
# corpus spot checks (skipped where the corpora are not mounted)

_TRAIN = Path(__file__).resolve().parents[2] / \
    "mib-doc-challenge" / "data" / "train"


@pytest.mark.skipif(not _TRAIN.is_dir(), reason="train corpus not mounted")
class TestRealCorpus:
    def _scan(self, pdf: Path) -> int:
        with pymupdf.open(str(pdf)) as doc:
            diagnostics.log_stamp_scan(doc, load_pages(doc), pdf.stem)
        return diagnostics._counts["stamped_cases"]

    def test_known_stamped_case_detects(self):
        assert self._scan(_TRAIN / "MIB-000005.pdf") == 1

    def test_known_unstamped_case_is_silent(self):
        assert self._scan(_TRAIN / "MIB-000024.pdf") == 0

    def test_red_stamp_decoy_case_is_silent(self):
        # MIB-000002 carries red decoy artifacts and no green stamp
        assert self._scan(_TRAIN / "MIB-000002.pdf") == 0
