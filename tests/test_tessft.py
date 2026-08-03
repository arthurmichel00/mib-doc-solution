"""Fine-tuned-font escalation OCR behind MIB_TESSFT=1.

A second tesseract LSTM whose recognizer was fine-tuned on the challenge
generator's font (MIT artifact by Shrey Shingala; see ATTRIBUTION.md).
It is NOT a corpus-wide second engine: it runs from the lazily-escalated
tail, and even there it reads only label-anchored VALUE STRIPS for the
fields still unread, located from word-box geometry the ladder already
produced. Its lines pool additively and sit inside the same escalation
restore guard as every variant. Flag unset = dead code: no engine init,
no OCR pass, no behavior change.
"""
from __future__ import annotations

import difflib

import cv2
import numpy as np
import pymupdf
import pytest

from mib_pipeline import ocr, pipeline
from mib_pipeline.fields import CaseEvidence
from mib_pipeline.model import Line, Source

CASE_ID = "MIB-000124"

MODEL_PRESENT = (ocr.tessft_dir() / f"{ocr.TESSFT_LANG}.traineddata").is_file()
needs_model = pytest.mark.skipif(
    not MODEL_PRESENT,
    reason="mib.traineddata not vendored in solution/models/tessdata "
           "(gitignored; see ATTRIBUTION.md for provenance)")


def _lines(texts, index=0, conf=0.9):
    return [Line(text=t, page_index=index, source=Source.OCR, conf=conf)
            for t in texts]


def _word(text, conf, x, y, w=60.0, h=20.0, key=(1, 1, 1)):
    return ocr.OcrWord(text, conf, key, x, y, w, h)


def _result(words, gray=None):
    if gray is None:
        gray = np.full((900, 1600), 255, np.uint8)
    return ocr.ScanOcrResult(lines=[], gray=gray, upright=True, words=words)


def _helv_strip(text: str, dpi: int = 288) -> np.ndarray:
    """One line of the generator's font rendered clean, cropped to a strip."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((40, 60), text, fontsize=16, fontname="helv")
    pix = page.get_pixmap(dpi=dpi)
    buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    doc.close()
    gray = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(gray[100:320, 100:1400])


def _read(engine, img) -> str:
    return " ".join(w.text for w in engine.words(ocr.pad_for_ocr(img)))


# --------------------------------------------------------- the artifact


class TestArtifact:
    def test_model_is_vendored_where_the_image_copies_from(self):
        """models/ is COPYed to /app/models, so the traineddata rides in
        with the ONNX weights — no Dockerfile change needed."""
        assert MODEL_PRESENT, (
            f"expected {ocr.tessft_dir()}/{ocr.TESSFT_LANG}.traineddata")

    def test_dir_defaults_under_models_and_honors_override(self, monkeypatch,
                                                           tmp_path):
        monkeypatch.delenv("MIB_TESSFT_DIR", raising=False)
        assert ocr.tessft_dir() == ocr._models_dir() / "tessdata"
        monkeypatch.setenv("MIB_TESSFT_DIR", str(tmp_path))
        assert ocr.tessft_dir() == tmp_path


# ------------------------------------------------------- engine loading


class TestEngineLoading:
    @needs_model
    def test_engine_loads_and_is_a_process_singleton(self, monkeypatch):
        monkeypatch.setattr(ocr, "_TESSFT_ENGINE", None)
        monkeypatch.setattr(ocr, "_TESSFT_UNAVAILABLE", False)
        engine = ocr.tessft_engine()
        assert engine is not None
        assert ocr.tessft_engine() is engine

    @needs_model
    def test_tesserocr_loads_the_custom_traineddata_in_process(self):
        """The image's backend. tesserocr resolves <lang>.traineddata under
        the `path` Init param, which is how a non-eng model gets loaded
        without touching TESSDATA_PREFIX (that env var still points the
        stock engine at the distro tessdata)."""
        tesserocr = pytest.importorskip(
            "tesserocr", reason="dev venv uses the pytesseract backend; "
                                "the image builds tesserocr 2.8.0")
        api = tesserocr.PyTessBaseAPI(
            path=str(ocr.tessft_dir()), lang=ocr.TESSFT_LANG,
            oem=tesserocr.OEM.LSTM_ONLY, psm=6)
        try:
            assert api.GetInitLanguagesAsString() == ocr.TESSFT_LANG
        finally:
            api.End()

    def test_missing_model_disables_the_pass_instead_of_raising(
            self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(ocr, "_TESSFT_ENGINE", None)
        monkeypatch.setattr(ocr, "_TESSFT_UNAVAILABLE", False)
        monkeypatch.setenv("MIB_TESSFT_DIR", str(tmp_path))    # empty dir
        assert ocr.tessft_engine() is None
        assert "MIB_TESSFT disabled" in capsys.readouterr().err
        # and it stays disabled without re-probing the filesystem
        assert ocr.tessft_engine() is None

    def test_missing_model_makes_the_pass_emit_nothing(self, monkeypatch,
                                                       tmp_path):
        monkeypatch.setattr(ocr, "_TESSFT_ENGINE", None)
        monkeypatch.setattr(ocr, "_TESSFT_UNAVAILABLE", False)
        monkeypatch.setenv("MIB_TESSFT_DIR", str(tmp_path))
        result = _result([_word("Fee", 0.9, 10, 10), _word("Status:", 0.9, 80, 10)])
        assert ocr.tessft_lines(result, 0, ("fee_status",)) == []


# ----------------------------------------- read quality on the font (7b)


@needs_model
class TestReadsAtLeastAsWellAsStock:
    """The artifact's whole claim is this font: held-out CER 9.59% -> 0.19%.
    On clean renders the stock engine is already good, so the bar is
    'at least as well', asserted per string against the ground truth."""

    STRINGS = (
        "Fee Status: paid",
        "Visa Class: DIP-1",
        "Home World: Proxima-b",
        "Sponsor ID: SPN-1345",
        "Arrival Date: 2026-05-24",
        "Observed flags: none",
    )

    @pytest.mark.parametrize("text", STRINGS)
    def test_fine_tuned_matches_or_beats_stock(self, text):
        strip = _helv_strip(text)
        ft = _read(ocr.tessft_engine(), strip)
        stock = _read(ocr.default_engine(), strip)
        score = lambda s: difflib.SequenceMatcher(None, text, s).ratio()
        assert score(ft) >= score(stock), (
            f"fine-tuned regressed: {ft!r} vs stock {stock!r}")

    def test_fine_tuned_reads_the_font_exactly(self):
        strip = _helv_strip("Fee Status: paid")
        assert _read(ocr.tessft_engine(), strip) == "Fee Status: paid"


# -------------------------------------------------------- strip locator


class TestStripLocator:
    def test_locates_the_value_strip_right_of_a_confident_label(self):
        words = [_word("Fee", 0.9, 100, 200), _word("Status:", 0.9, 170, 200),
                 _word("paid", 0.9, 260, 200)]
        strips = ocr.tessft_strips(_result(words), ("fee_status",))
        assert len(strips) == 1
        h, w = strips[0].shape
        # starts 2px left of the label, runs _TESSFT_STRIP_W right of it, so
        # the crop carries "Fee Status:" AND the value that follows
        assert w == ocr._TESSFT_STRIP_W + 2
        assert h >= 20                      # label height + padding

    def test_only_requested_fields_are_located(self):
        words = [_word("Fee", 0.9, 100, 200), _word("Status:", 0.9, 170, 200),
                 _word("Visa", 0.9, 100, 300), _word("Class:", 0.9, 170, 300)]
        assert len(ocr.tessft_strips(_result(words), ("fee_status",))) == 1
        assert len(ocr.tessft_strips(
            _result(words), ("fee_status", "visa_class"))) == 2
        assert ocr.tessft_strips(_result(words), ()) == []

    def test_low_confidence_label_is_not_an_anchor(self):
        """The locator gate is what guarantees the crop holds real text —
        it is the reason the model is never handed blank paper."""
        conf = ocr._TESSFT_LOC_MIN_CONF - 0.01
        words = [_word("Fee", conf, 100, 200), _word("Status:", 0.9, 170, 200)]
        assert ocr.tessft_strips(_result(words), ("fee_status",)) == []

    def test_tokens_on_different_rows_are_not_a_label(self):
        words = [_word("Fee", 0.9, 100, 200), _word("Status:", 0.9, 170, 400)]
        assert ocr.tessft_strips(_result(words), ("fee_status",)) == []

    def test_repeated_reads_of_one_row_yield_one_strip(self):
        words = [_word("Fee", 0.9, 100, 200), _word("Status:", 0.9, 170, 200),
                 _word("Fee", 0.8, 100, 202), _word("Status:", 0.8, 170, 202)]
        assert len(ocr.tessft_strips(_result(words), ("fee_status",))) == 1

    def test_strip_count_is_capped(self):
        words = []
        for i in range(20):
            y = 40 * i
            words += [_word("Home", 0.9, 100, y), _word("World:", 0.9, 170, y)]
        strips = ocr.tessft_strips(_result(words), ("home_world",))
        assert len(strips) == ocr._TESSFT_MAX_STRIPS

    def test_abstains_without_a_word_stash(self):
        """Legacy ScanOcrResults carry words=None; no full-page fallback."""
        assert ocr.tessft_strips(_result(None), ("fee_status",)) == []
        assert ocr.tessft_strips(_result([]), ("fee_status",)) == []

    def test_stash_without_geometry_is_ignored(self):
        words = [ocr.OcrWord("Fee", 0.9, (1, 1, 1), 100.0),
                 ocr.OcrWord("Status:", 0.9, (1, 1, 1), 170.0)]
        assert ocr.tessft_strips(_result(words), ("fee_status",)) == []

    def test_every_gate_field_has_a_label_pattern(self):
        ev = CaseEvidence()
        for field in pipeline._unread_inputs(ev):
            assert field in ocr.TESSFT_LABELS


# ---------------------------------------------------------- conf scale


class TestConfScale:
    def test_emitted_lines_carry_the_scale(self, monkeypatch):
        words = [_word("Fee", 0.80, 1, 1), _word("Status:", 0.80, 70, 1)]
        monkeypatch.setattr(ocr, "tessft_strips",
                            lambda *a: [np.full((30, 300), 255, np.uint8)])
        monkeypatch.setattr(ocr, "tessft_engine",
                            lambda: type("E", (), {"words": lambda s, i, sparse=False: words})())
        got = ocr.tessft_lines(_result(words), 4, ("fee_status",))
        assert [round(l.conf, 6) for l in got] == [
            round(0.80 * ocr._TESSFT_CONF_SCALE, 6)]
        assert got[0].page_index == 4
        assert got[0].source == Source.OCR
        assert got[0].tier1_ok is True      # a real recognizer, not a repair

    def test_scale_keeps_the_measured_hallucination_ceiling_sub_affirmative(self):
        """Calibration record. On textureless noise the fine-tuned model
        invented words up to conf 0.7320 where stock peaked at 0.5055; the
        scale exists to keep that ceiling under the affirmative-read line
        so an invented read can never set known=True."""
        from mib_pipeline.fields import _KNOWN_MIN_OCR_CONF

        assert 0.7320 * ocr._TESSFT_CONF_SCALE < _KNOWN_MIN_OCR_CONF
        # ... while a typical genuine read (matched-word mean 0.92) survives
        assert 0.92 * ocr._TESSFT_CONF_SCALE > _KNOWN_MIN_OCR_CONF


# ------------------------------------------------------------ flag gate


class TestFlagGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_TESSFT", raising=False)
        assert pipeline.TESSFT_DEFAULT is False
        assert pipeline._tessft_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_TESSFT", "1")
        assert pipeline._tessft_enabled() is True
        monkeypatch.setenv("MIB_TESSFT", "0")
        assert pipeline._tessft_enabled() is False


# ------------------------------------------------- unread-input derivation


class TestUnreadInputs:
    def test_agrees_with_the_shipped_gate(self):
        """_under_determined is now derived from _unread_inputs; the two
        must stay the same predicate for every combination of gate fields."""
        import itertools

        for bits in itertools.product((None, "x"), repeat=4):
            ev = CaseEvidence()
            ev.values["fee_status"] = bits[0]
            ev.values["visa_class"] = bits[1]
            ev.values["home_world"] = bits[2]
            ev.values["sponsor_id"] = bits[3]
            for flags_known in (False, True):
                for arrival in (False, True):
                    ev.flags_known = flags_known
                    ev.arrival_on_intake = arrival
                    expected = (
                        bits[0] is None or not flags_known or not arrival
                        or bits[1] is None or bits[2] is None
                        or (bits[1] != "DIP-1" and bits[3] is None))
                    assert pipeline._under_determined(ev) is expected
                    assert bool(pipeline._unread_inputs(ev)) is expected

    def test_dip1_does_not_ask_for_a_sponsor_strip(self):
        ev = CaseEvidence()
        ev.values.update({"fee_status": "paid", "visa_class": "DIP-1",
                          "home_world": "Mars Dome-7", "sponsor_id": None})
        ev.flags_known = True
        ev.arrival_on_intake = True
        assert pipeline._unread_inputs(ev) == ()


# --------------------------------------- pipeline wiring (real PDF, e2e)


def _make_scan_pdf(tmp_path, page_specs, case_id=CASE_ID):
    """Synthetic packet whose pages are full-page raster images (SCAN kind)
    with the genuine vector footer (pattern from test_userwords)."""
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
INTAKE_SPEC_FEE_PAID = INTAKE_SPEC + ["Fee Status: paid"]


def _quiet_escalation(monkeypatch):
    """Silence the heavy under-determined engines irrelevant to this block
    (their own suites cover them; the worktree also has no ONNX weights)."""
    from mib_pipeline import crnn

    monkeypatch.setattr(ocr, "rapid_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "escalation_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "weld_sponsor_lines", lambda *a, **k: [])
    monkeypatch.setattr(crnn, "crnn_lines", lambda *a, **k: [])


class TestIntegrationPipeline:
    def test_flag_off_never_runs_the_pass(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIB_TESSFT", raising=False)
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC])
        calls = []
        monkeypatch.setattr(ocr, "tessft_lines",
                            lambda *a, **k: calls.append(1) or [])
        monkeypatch.setattr(ocr, "tessft_engine",
                            lambda: pytest.fail("engine must not init"))
        row = pipeline.process_pdf(pdf)
        assert calls == []                    # inert: zero OCR passes
        assert row["_path"] == "R8_fee_unread"
        assert row["adjudication"] == "NEEDS_REVIEW"

    def test_flag_on_fills_an_unread_decision_input(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC])
        asked = []

        def fake(result, page_index, wanted):
            asked.append((page_index, wanted))
            return _lines(["Fee Status: paid"], index=page_index)

        monkeypatch.setattr(ocr, "tessft_lines", fake)
        row = pipeline.process_pdf(pdf)
        assert [p for p, _ in asked] == [0]          # every scan page, once
        assert "fee_status" in asked[0][1]           # asked for what is unread
        assert row["fee_status"] == "paid"
        assert row["_path"] != "R8_fee_unread"

    def test_only_unread_fields_are_requested(self, tmp_path, monkeypatch):
        """The cost gate: the pass never asks for strips of fields the
        ladder already read."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC])
        asked = []
        monkeypatch.setattr(
            ocr, "tessft_lines",
            lambda result, i, wanted: asked.append(wanted) or [])
        pipeline.process_pdf(pdf)
        assert asked
        # visa/home world/arrival print cleanly on this packet
        assert "visa_class" not in asked[0]
        assert "home_world" not in asked[0]
        # DIP-1 waives the sponsor requirement
        assert "sponsor_id" not in asked[0]

    def test_flag_on_never_outvotes_an_affirmative_read(self, tmp_path,
                                                        monkeypatch):
        """Restore guard: the pass may FILL, never flip."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC_FEE_PAID])
        monkeypatch.setattr(
            ocr, "tessft_lines",
            lambda result, i, wanted: _lines(["Fee Status: unpaid"],
                                             index=i, conf=0.99))
        row = pipeline.process_pdf(pdf)
        assert row["fee_status"] == "paid"
        assert row["adjudication"] != "DENIED"

    def test_pooling_is_additive_and_keeps_the_stock_read(self, tmp_path,
                                                          monkeypatch):
        """A stock-engine line is never replaced — the fine-tuned line joins
        the pool beside it, and a duplicate at lower confidence does not
        pull the stock line's confidence down."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC_FEE_PAID])
        monkeypatch.setattr(
            ocr, "tessft_lines",
            lambda result, i, wanted: _lines(["Fee Status: paid"], index=i,
                                             conf=0.11)
            + _lines(["Observed flags: none"], index=i, conf=0.80))

        seen = {}
        real = pipeline.fields.collect_candidates

        def spy(pages, case_id):
            seen["lines"] = [l for p in pages for l in p.lines]
            return real(pages, case_id)

        monkeypatch.setattr(pipeline.fields, "collect_candidates", spy)
        pipeline.process_pdf(pdf)
        pool = seen["lines"]
        fee = [l for l in pool
               if " ".join(l.text.lower().split()) == "fee status: paid"]
        assert fee, "the stock read vanished from the pool"
        assert max(l.conf for l in fee) > 0.11    # stock conf kept, not 0.11
        # and the additive line is there too
        assert any("observed flags" in l.text.lower() for l in pool)

    @needs_model
    def test_real_pass_runs_end_to_end_and_invents_nothing(self, tmp_path,
                                                           monkeypatch):
        """No stub: the genuine engine, locator and pooling inside
        process_pdf. The packet prints a bare 'Fee Status:' label with no
        value — the damage shape the lever exists for — so the locator
        anchors a real strip and the fine-tuned model reads it. The field
        must stay unread: there is no value printed, and a pass that
        invented one here is the exact catastrophic-false-approval risk."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC + ["Fee Status:"]])

        seen = []
        real = ocr.tessft_lines
        monkeypatch.setattr(
            ocr, "tessft_lines",
            lambda result, i, wanted: seen.append(real(result, i, wanted))
            or seen[-1])

        row = pipeline.process_pdf(pdf)
        assert seen, "the pass never ran"
        reads = [l for batch in seen for l in batch]
        assert any("fee status" in l.text.lower() for l in reads), \
            f"locator/engine produced nothing usable: {[l.text for l in reads]}"
        assert all(l.conf <= ocr._TESSFT_CONF_SCALE for l in reads)
        # The fee is not affirmatively read, so adjudication still routes to
        # the unread path. (The row carries a non-authoritative extracted
        # value; `known` is what policy consults, and it stays False.)
        assert row["_path"] == "R8_fee_unread"

        # Strongest statement available: on a label with no value printed,
        # turning the lever on changes the emitted row not at all.
        monkeypatch.setenv("MIB_TESSFT", "0")
        assert pipeline.process_pdf(pdf) == row

    def test_packet_with_no_scan_page_never_fires(self, tmp_path,
                                                  monkeypatch):
        """149 corpus packets are all digital text; they must not pay.
        The enclosing escalation block gates on `scans`, so the pass is
        unreachable for them — pinned here so a future reorder cannot
        quietly move this block outside that gate."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        y = 70
        for line in INTAKE_SPEC:                 # vector text, no raster
            page.insert_text((60, y), line, fontsize=11, fontname="helv")
            y += 24
        page.insert_text((40, 782), f"Packet {CASE_ID} / page 1",
                         fontsize=6, fontname="helv")
        pdf = str(tmp_path / f"{CASE_ID}.pdf")
        doc.save(pdf)
        doc.close()

        monkeypatch.setattr(ocr, "tessft_lines",
                            lambda *a, **k: pytest.fail("no scan page"))
        monkeypatch.setattr(ocr, "tessft_engine",
                            lambda: pytest.fail("engine must not init"))
        pipeline.process_pdf(pdf)

    def test_clean_row_read_is_not_displaced(self, tmp_path, monkeypatch):
        """The MIB_USERWORDS anti-repeat (ledgered -0.22, 'vocab-biased
        decoding corrupts reads'). This lever is model-level rather than
        decoder-bias, but the failure mode it must not share is displacing
        a correct read on a CLEAN row. Here every decision field prints
        cleanly except the fee, and the pass claims a wrong value for one
        of the clean ones at maximum confidence."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC])
        monkeypatch.setattr(
            ocr, "tessft_lines",
            lambda result, i, wanted: _lines(
                ["Home World: Kepler-442b", "Visa Class: XW-1",
                 "Arrival Date: 2019-01-01"], index=i, conf=1.0))
        row = pipeline.process_pdf(pdf)
        assert row["home_world"] == "Proxima-b"
        assert row["visa_class"] == "DIP-1"
        assert row["arrival_date"] == "2026-05-04"

    def test_engine_never_initialises_when_the_case_is_determined(
            self, tmp_path, monkeypatch):
        """Nothing unread -> no strips, no engine, even with the flag on."""
        monkeypatch.setenv("MIB_TESSFT", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_SPEC_FEE_PAID
                                        + ["Observed flags: none"]])
        monkeypatch.setattr(ocr, "tessft_lines",
                            lambda *a, **k: pytest.fail("pass must not run"))
        pipeline.process_pdf(pdf)
