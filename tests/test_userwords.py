"""Vocabulary-constrained escalation OCR behind MIB_USERWORDS=1 (Sunday).

The decision inputs come from closed vocabularies (vocab.py) printed after
a fixed set of form labels. ocr.userwords_lines runs ONE extra tesseract
pass per escalation view with those strings loaded as user-words (plus
SPN-####/ISO-date user-patterns), biasing the LSTM beam search on damaged
rows toward tokens the downstream matchers can consume. The pass lives in
the lazily-escalated ladder tail and inside the SAME restore guard as
every escalation variant: it fills still-unread inputs only and can never
out-vote an affirmative pre-escalation read. With the flag unset the block
is dead code: no engine init, no OCR pass, no behavior change.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from mib_pipeline import ocr, pipeline, vocab
from mib_pipeline.model import Line, Source

CASE_ID = "MIB-000124"


def _lines(texts, index=0, conf=0.9):
    return [Line(text=t, page_index=index, source=Source.OCR, conf=conf)
            for t in texts]


class RecordingEngine:
    def __init__(self, canned=None):
        self.calls = []
        self._canned = list(canned or [])

    def words(self, image, sparse=False):
        self.calls.append((image.shape, sparse))
        return self._canned.pop(0) if self._canned else []


# ---------------------------------------------------- word-list builder


class TestUserwordsBuilder:
    def test_species_codes_keep_underscore_and_space_variants(self):
        for code in vocab.SPECIES_CODES:
            assert code in ocr.USERWORDS                    # ALPHA_DRACONIAN
            for token in code.split("_"):
                assert token in ocr.USERWORDS               # ALPHA, DRACONIAN

    def test_home_world_tokens(self):
        for world in vocab.HOME_WORLDS:
            for token in world.split():
                assert token in ocr.USERWORDS   # Mars, Dome-7, TRAPPIST-1e...

    def test_visa_classes_verbatim(self):
        for cls in vocab.VISA_CLASSES:
            assert cls in ocr.USERWORDS

    def test_purpose_tokens(self):
        for purpose in vocab.PURPOSES:
            for token in purpose.split():
                assert token in ocr.USERWORDS   # archive, audit, research...

    def test_fee_statuses(self):
        for status in vocab.FEE_STATUSES:
            assert status in ocr.USERWORDS

    def test_form_label_tokens(self):
        for label in ("Home World", "Visa Class", "Sponsor ID",
                      "Arrival Date", "Declared Purpose", "Species Code",
                      "Fee Status", "Observed Flags", "FINDING", "Reason"):
            for token in label.split():
                assert token in ocr.USERWORDS

    def test_entries_are_single_nonempty_tokens(self):
        assert ocr.USERWORDS
        assert len(set(ocr.USERWORDS)) == len(ocr.USERWORDS)   # de-duped
        for word in ocr.USERWORDS:
            assert word and not any(c.isspace() for c in word)

    def test_patterns_cover_sponsor_and_iso_date(self):
        assert ocr.USERPATTERNS == (r"SPN-\d\d\d\d", r"\d\d\d\d-\d\d-\d\d")


# ------------------------------------------------ tmpfiles (once/process)


class TestDictionaryFiles:
    def test_userwords_file_written_once_one_word_per_line(self, monkeypatch,
                                                           tmp_path):
        monkeypatch.setattr(ocr, "_USERWORDS_PATH", None)
        monkeypatch.setattr(ocr.tempfile, "tempdir", str(tmp_path))
        path = ocr.userwords_path()
        assert ocr.userwords_path() == path            # cached, not rewritten
        with open(path, encoding="utf-8") as fh:
            assert fh.read().splitlines() == list(ocr.USERWORDS)

    def test_userpatterns_file_written_once(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ocr, "_USERPATTERNS_PATH", None)
        monkeypatch.setattr(ocr.tempfile, "tempdir", str(tmp_path))
        path = ocr.userpatterns_path()
        assert ocr.userpatterns_path() == path
        with open(path, encoding="utf-8") as fh:
            assert fh.read().splitlines() == list(ocr.USERPATTERNS)


# ------------------------------------------------------- engine plumbing


class _StubPytesseract:
    class Output:
        DICT = "dict"

    def __init__(self):
        self.configs = []

    def image_to_data(self, image, config="", output_type=None):
        self.configs.append(config)
        return {"text": [], "conf": [], "block_num": [], "par_num": [],
                "line_num": [], "left": [], "top": [], "width": [],
                "height": []}


class TestEnginePlumbing:
    def test_config_extra_reaches_every_tesseract_call(self, monkeypatch):
        stub = _StubPytesseract()
        monkeypatch.setitem(sys.modules, "pytesseract", stub)
        engine = ocr.PytesseractEngine(config_extra="--user-words /tmp/w.txt")
        img = np.full((8, 8), 255, np.uint8)
        engine.words(img)
        engine.words(img, sparse=True)
        assert stub.configs == ["--oem 1 --psm 6 --user-words /tmp/w.txt",
                                "--oem 1 --psm 11 --user-words /tmp/w.txt"]

    def test_no_config_extra_is_byte_identical_to_before(self, monkeypatch):
        stub = _StubPytesseract()
        monkeypatch.setitem(sys.modules, "pytesseract", stub)
        ocr.PytesseractEngine().words(np.full((8, 8), 255, np.uint8))
        assert stub.configs == ["--oem 1 --psm 6"]

    def test_userwords_engine_loads_both_dictionaries(self, monkeypatch):
        monkeypatch.setenv("MIB_OCR_ENGINE", "pytesseract")
        monkeypatch.setattr(ocr, "_USERWORDS_ENGINE", None)
        engine = ocr.userwords_engine()
        assert isinstance(engine, ocr.PytesseractEngine)
        assert f"--user-words {ocr.userwords_path()}" in engine._config_extra
        assert f"--user-patterns {ocr.userpatterns_path()}" \
            in engine._config_extra
        assert ocr.userwords_engine() is engine       # per-process singleton

    def test_userwords_lines_is_one_pass(self, monkeypatch):
        engine = RecordingEngine()
        monkeypatch.setattr(ocr, "userwords_engine", lambda: engine)
        got = ocr.userwords_lines(np.full((6, 9), 255, np.uint8), 3)
        assert engine.calls == [((6, 9), False)]      # ONE pass, non-sparse
        assert got == []

    def test_userwords_lines_emit_ordinary_ocr_lines(self, monkeypatch):
        words = [ocr.OcrWord("Fee", 0.8, (1, 1, 1), 1.0),
                 ocr.OcrWord("Status:", 0.8, (1, 1, 1), 30.0),
                 ocr.OcrWord("paid", 0.8, (1, 1, 1), 80.0)]
        engine = RecordingEngine(canned=[words])
        monkeypatch.setattr(ocr, "userwords_engine", lambda: engine)
        got = ocr.userwords_lines(np.full((6, 9), 255, np.uint8), 3)
        assert [l.text for l in got] == ["Fee Status: paid"]
        assert got[0].page_index == 3
        assert got[0].source == Source.OCR
        assert got[0].tier1_ok is True                # ordinary trusted lines


# ------------------------------------------------------------ flag gate


class TestFlagGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_USERWORDS", raising=False)
        assert pipeline.USERWORDS_DEFAULT is False
        assert pipeline._userwords_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_USERWORDS", "1")
        assert pipeline._userwords_enabled() is True
        monkeypatch.setenv("MIB_USERWORDS", "0")
        assert pipeline._userwords_enabled() is False


# --------------------------------------- pipeline wiring (real PDF, e2e)


def _make_scan_pdf(tmp_path, page_specs, case_id=CASE_ID):
    """Synthetic packet whose pages are full-page raster images (SCAN kind)
    with the genuine vector footer (pattern from test_note_rescue)."""
    import pymupdf

    doc = pymupdf.open()
    for i, lines in enumerate(page_specs):
        src = pymupdf.open()
        sp = src.new_page(width=612, height=792)
        y = 70
        for line in lines:
            sp.insert_text((60, y), line, fontsize=16, fontname="helv")
            y += 30
        pix = sp.get_pixmap(dpi=180)
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, pixmap=pix)
        page.insert_text((40, 782), f"Packet {case_id} / page {i + 1}",
                         fontsize=6, fontname="helv")
        src.close()
    path = tmp_path / f"{case_id}.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


# Fee row and flags row missing: the case is under-determined, so the
# escalation tail (and with it the userwords block) runs.
INTAKE_SPEC = [
    "FORM I-8090: Extraterrestrial Work Authorization Intake",
    f"Case ID: {CASE_ID}",
    "Applicant: Solul Zamora",
    "Species Code: ORION_GRAYS",
    "Home World: Proxima-b",
    "Visa Class: DIP-1",
    "Arrival Date: 2026-05-04",
    "Declared Purpose: research",
]
# Same, plus an affirmatively readable fee row (for the restore-guard test;
# the missing flags row keeps the case under-determined).
INTAKE_SPEC_FEE_PAID = INTAKE_SPEC + ["Fee Status: paid"]


def _quiet_escalation(monkeypatch):
    """Silence the heavy under-determined engines irrelevant to the block
    under test (their own suites cover them)."""
    from mib_pipeline import crnn

    monkeypatch.setattr(ocr, "rapid_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "escalation_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "weld_sponsor_lines", lambda *a, **k: [])
    monkeypatch.setattr(crnn, "crnn_lines", lambda *a, **k: [])


class TestIntegrationPipeline:
    def test_flag_off_never_runs_the_pass(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIB_USERWORDS", raising=False)
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC])
        calls = []
        monkeypatch.setattr(ocr, "userwords_lines",
                            lambda *a, **k: calls.append(1) or [])
        row = pipeline.process_pdf(pdf)
        assert calls == []                    # inert: zero OCR passes
        assert row["_path"] == "R8_fee_unread"
        assert row["adjudication"] == "NEEDS_REVIEW"

    def test_flag_on_fills_an_unread_decision_input(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.setenv("MIB_USERWORDS", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC])
        queried = []

        def fake_userwords(gray, page_index):
            queried.append(page_index)
            return _lines(["Fee Status: paid"], index=page_index)

        monkeypatch.setattr(ocr, "userwords_lines", fake_userwords)
        row = pipeline.process_pdf(pdf)
        assert queried == [0]                 # every scan page, once
        assert row["fee_status"] == "paid"    # the fill went through
        assert row["_path"] != "R8_fee_unread"

    def test_flag_on_never_outvotes_an_affirmative_read(self, tmp_path,
                                                        monkeypatch):
        """Restore guard: the pass may FILL, never flip. A userwords line
        contradicting the ladder's affirmative fee read is discarded by
        the escalation restore guard."""
        monkeypatch.setenv("MIB_USERWORDS", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC_FEE_PAID])
        monkeypatch.setattr(
            ocr, "userwords_lines",
            lambda gray, page_index: _lines(["Fee Status: unpaid"],
                                            index=page_index, conf=0.99))
        row = pipeline.process_pdf(pdf)
        assert row["fee_status"] == "paid"    # affirmative read restored
        assert row["adjudication"] != "DENIED"

    def test_flag_on_cannot_mint_finding_over_existing_evidence(
            self, tmp_path, monkeypatch):
        """A userwords line naming a Finding must not displace a
        pre-existing finding: the restore guard puts the ladder's finding
        back (same containment as every escalation variant)."""
        monkeypatch.setenv("MIB_USERWORDS", "1")
        _quiet_escalation(monkeypatch)
        # The finding sits on a finding-eligible note page (intake labels
        # would make their page ineligible, exactly like the D2 fixtures).
        pdf = _make_scan_pdf(
            tmp_path,
            [INTAKE_SPEC,
             ["Manual Adjudicator Note", "Finding: APPROVED."]])
        monkeypatch.setattr(
            ocr, "userwords_lines",
            lambda gray, page_index: _lines(["Finding: DENIED."],
                                            index=page_index, conf=0.99))
        row = pipeline.process_pdf(pdf)
        assert row["adjudication"] == "APPROVED"
        assert row["_path"] == "N0_note_approved"
