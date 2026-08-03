"""Closed-menu CTC fill behind MIB_CTCFILL=1 (ctc-branch).

The lever scores every legal menu candidate against the PP-OCRv6 rec
model's frame posteriors with the exact CTC forward algorithm, on value
strips anchored by the OCR ladder's own word-box geometry. It fires only
on decision-relevant menu fields still unread after every engine and the
escalation restore guard, and its accepted values are extraction-only
fills: confidence capped below the affirmative-read threshold, `known`
never set, policy blind to them. With the flag unset the block is dead
code. Model-dependent tests skip when the rec bundle is absent
(MIB_MODELS_DIR points at it in the battery environment).
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ctcfill, fields, ocr, pipeline, policy
from mib_pipeline.model import Line, Source
from mib_pipeline.ocr import OcrWord, ScanOcrResult

CASE_ID = "MIB-000124"

_MODEL = ctcfill.available()
needs_model = pytest.mark.skipif(not _MODEL, reason="rec bundle not mounted")


def _word(text, conf=0.9, x=100.0, y=200.0, w=60.0, h=20.0,
          key=(1, 1, 1)):
    return OcrWord(text, conf, key, x, y, w, h)


def _result(words, shape=(600, 900), upright=True):
    gray = np.full(shape, 255, np.uint8)
    return ScanOcrResult(lines=[], gray=gray, upright=upright, words=words)


# ----------------------------------------------------------- field scope


class TestFieldScope:
    def test_only_decision_relevant_closed_menus(self):
        assert ctcfill.FIELDS == ("species_code", "home_world",
                                  "visa_class", "declared_purpose")

    def test_fee_status_excluded_by_construction(self):
        assert "fee_status" not in ctcfill.FIELDS
        assert "fee_status" not in ctcfill._MENUS
        assert "fee_status" not in ctcfill._LABELS

    def test_sponsor_excluded_by_construction(self):
        assert "sponsor_id" not in ctcfill.FIELDS
        assert "sponsor_id" not in ctcfill._MENUS

    def test_no_finding_or_flag_surface(self):
        for fld in ("risk_flags", "applicant_name", "arrival_date"):
            assert fld not in ctcfill.FIELDS

    def test_hard_embargo_worlds_never_accepted(self):
        assert ctcfill._EXCLUDED_VALUES == \
            frozenset(policy.HARD_EMBARGO_WORLDS)

    def test_conf_cap_below_affirmative_threshold(self):
        assert ctcfill.CONF_CAP < fields._KNOWN_MIN_OCR_CONF


# ------------------------------------------------------------- flag gate


class TestFlagGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_CTCFILL", raising=False)
        assert pipeline.CTCFILL_DEFAULT is False
        assert pipeline._ctcfill_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL", "1")
        assert pipeline._ctcfill_enabled() is True
        monkeypatch.setenv("MIB_CTCFILL", "0")
        assert pipeline._ctcfill_enabled() is False


# --------------------------------------------------------------- locator


class TestLocator:
    def test_locates_two_token_label(self):
        words = [_word("Home", x=100, w=55), _word("World:", x=160, w=60)]
        strips = ctcfill.locate_strips(_result(words), "home_world")
        assert len(strips) == 1
        strip, label_text, conf = strips[0]
        assert label_text == "Home World:"
        assert conf == pytest.approx(0.9)
        assert strip.shape[0] == 20 + 2 * ctcfill._STRIP_Y_PAD
        assert strip.shape[1] > 400          # label + value region

    def test_abstains_on_low_word_conf(self):
        words = [_word("Home", conf=0.4), _word("World:", x=160, conf=0.9)]
        assert ctcfill.locate_strips(_result(words), "home_world") == []

    def test_abstains_when_tokens_not_on_one_row(self):
        words = [_word("Home", y=200), _word("World:", x=160, y=260)]
        assert ctcfill.locate_strips(_result(words), "home_world") == []

    def test_abstains_on_wrong_reading_order(self):
        words = [_word("Home", x=400), _word("World:", x=100)]
        assert ctcfill.locate_strips(_result(words), "home_world") == []

    def test_reads_stash_even_when_orientation_was_estimated(self):
        # the stash was read on the UNROTATED render: a confident label
        # hit there is genuine even if the sparse-page orientation
        # estimator guessed a rotation
        words = [_word("Home", x=100, w=55), _word("World:", x=160, w=60)]
        strips = ctcfill.locate_strips(_result(words, upright=False),
                                       "home_world")
        assert len(strips) == 1

    def test_abstains_without_stash(self):
        assert ctcfill.locate_strips(_result(None), "home_world") == []

    def test_ignores_legacy_words_without_geometry(self):
        words = [OcrWord("Home", 0.9, (1, 1, 1), 100.0),
                 OcrWord("World:", 0.9, (1, 1, 1), 160.0)]
        assert ctcfill.locate_strips(_result(words), "home_world") == []

    def test_no_cross_field_label_match(self):
        words = [_word("Home", x=100, w=55), _word("World:", x=160, w=60)]
        assert ctcfill.locate_strips(_result(words), "visa_class") == []

    def test_dedupes_rereads_of_same_row(self):
        words = [_word("Home", x=100, w=55), _word("World:", x=160, w=60),
                 _word("Home", x=101, w=55, conf=0.7),
                 _word("World:", x=161, w=60, conf=0.7)]
        strips = ctcfill.locate_strips(_result(words), "home_world")
        assert len(strips) == 1
        assert strips[0][2] == pytest.approx(0.9)     # max-conf read kept

    def test_punctuation_and_case_tolerant_tokens(self):
        words = [_word("|Declared", x=100, w=90),
                 _word("PURPOSE:", x=200, w=90)]
        strips = ctcfill.locate_strips(_result(words), "declared_purpose")
        assert strips and strips[0][1] == "Declared Purpose:"


# -------------------------------------------------------- acceptance gate


class TestAcceptanceGate:
    def test_rejects_below_floor(self):
        scored = [(ctcfill.SCORE_FLOOR - 0.1, "Proxima-b"),
                  (-9.0, "Luyten-b"), (-9.5, "<none>")]
        assert ctcfill._gated_winner(scored) is None

    def test_rejects_thin_runner_up_margin(self):
        top = ctcfill.SCORE_FLOOR + 1.0
        scored = [(top, "Proxima-b"),
                  (top - ctcfill.MARGIN_RUNNER_UP + 0.01, "Luyten-b"),
                  (-9.5, "<none>")]
        assert ctcfill._gated_winner(scored) is None

    def test_rejects_none_hypothesis_winner(self):
        scored = [(-1.0, "<none>"), (-2.0, "Proxima-b")]
        assert ctcfill._gated_winner(scored) is None

    def test_rejects_when_none_is_close(self):
        top = ctcfill.SCORE_FLOOR + 1.0
        scored = [(top, "Proxima-b"), (top - 1.0, "Luyten-b"),
                  (top - ctcfill.MARGIN_NONE, "<none>")]
        # margin over none must be >= MARGIN_NONE; equality passes only
        # when MARGIN_NONE == 0, so probe just below it
        scored_close = [(top, "Proxima-b"), (top - 1.0, "Luyten-b"),
                        (top + 0.01 - ctcfill.MARGIN_NONE, "<none>")]
        assert ctcfill._gated_winner(scored_close) is None
        assert ctcfill._gated_winner(scored) == "Proxima-b"

    def test_accepts_clear_winner(self):
        scored = [(-1.0, "Proxima-b"), (-2.0, "Luyten-b"), (-6.0, "<none>")]
        assert ctcfill._gated_winner(scored) == "Proxima-b"


# ------------------------------------------------------------- CTC maths


class TestCtcForward:
    def test_matches_brute_force_path_sum(self):
        """Exact forward == exhaustive enumeration over all frame paths."""
        rng = np.random.default_rng(7)
        T, C = 4, 3
        p = rng.random((T, C)) + 0.05
        p /= p.sum(axis=1, keepdims=True)
        logp = np.log(p)
        labels = (1, 2)

        def collapse(path):
            out, prev = [], 0
            for ix in path:
                if ix != 0 and ix != prev:
                    out.append(ix)
                prev = ix
            return tuple(out)

        total = 0.0
        for path in np.ndindex(*(C,) * T):
            if collapse(path) == labels:
                total += float(np.prod([p[t, ix]
                                        for t, ix in enumerate(path)]))
        got = ctcfill._ctc_logp(logp, labels)
        assert got == pytest.approx(np.log(total), abs=1e-9)

    def test_empty_labels_is_all_blank_path(self):
        logp = np.log(np.full((3, 2), 0.5))
        assert ctcfill._ctc_logp(logp, ()) == pytest.approx(3 * np.log(0.5))

    def test_too_long_label_is_impossible(self):
        logp = np.log(np.full((2, 4), 0.25))
        assert ctcfill._ctc_logp(logp, (1, 2, 3)) == ctcfill._NEG


# ------------------------------------------------- model-dependent reads


def _render_row(text, width=900, height=44):
    import cv2

    img = np.full((height, width), 255, np.uint8)
    cv2.putText(img, text, (8, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    return img


@needs_model
class TestModelContract:
    def test_charset_matches_head(self):
        sess = ctcfill._session()
        assert sess.get_outputs()[0].shape[-1] == len(ctcfill._CHARS)
        assert ctcfill._CHARS[0] == "<blank>"
        assert ctcfill._CHARS[-1] == " "

    def test_menu_candidates_all_in_dictionary(self):
        ctcfill._session()
        for field, menu in ctcfill._MENUS.items():
            for label_pattern in ctcfill._LABELS[field]:
                label = " ".join(label_pattern) + ":"
                for value in menu:
                    assert ctcfill._label_ixs(f"{label} {value}") is not None

    def test_clean_synthetic_strip_ranks_gold_first(self):
        strip = _render_row("Home World: Proxima-b")
        scored = ctcfill.score_strip(strip, "Home World:",
                                     ctcfill._MENUS["home_world"])
        assert scored[0][1] == "Proxima-b"

    def test_blank_strip_prefers_none(self):
        strip = np.full((40, 700), 255, np.uint8)
        scored = ctcfill.score_strip(strip, "Home World:",
                                     ctcfill._MENUS["home_world"])
        assert scored[0][1] == "<none>"


@needs_model
class TestReadField:
    def _result_for(self, text, field="home_world",
                    tokens=("Home", "World:")):
        img = np.full((300, 900), 255, np.uint8)
        row = _render_row(text)
        img[100:144, 40:860] = row[:, :820]
        words, x = [], 40.0
        for tok in tokens:
            words.append(_word(tok, x=x, y=104, w=18.0 * len(tok), h=30))
            x += 18.0 * len(tok) + 12
        return ScanOcrResult(lines=[], gray=img, upright=True, words=words)

    def test_reads_clean_value(self):
        result = self._result_for("Home World: Proxima-b")
        got = ctcfill.read_field(result, "home_world")
        assert got is not None
        assert got[0] == "Proxima-b"
        assert got[1] <= ctcfill.CONF_CAP

    def test_hard_embargo_value_dropped(self):
        result = self._result_for("Home World: TRAPPIST-1e")
        assert ctcfill.read_field(result, "home_world") is None

    def test_blank_value_abstains(self):
        result = self._result_for("Home World:")
        assert ctcfill.read_field(result, "home_world") is None

    def test_mode_default_value_suppressed(self):
        # emitting the writer's fallback is a no-op by construction AND
        # was the only wrong-fire mode the calibration measured
        result = self._result_for("Home World: Luyten-b")
        assert ctcfill.read_field(result, "home_world") is None


# --------------------------------------------------------- fill contract


class TestFillContract:
    def test_unavailable_model_abstains(self, monkeypatch):
        monkeypatch.setattr(ctcfill, "available", lambda: False)
        result = _result([_word("Home"), _word("World:", x=160)])
        assert ctcfill.fill({0: result}, ["home_world"]) == {}

    def test_cross_page_disagreement_abstains(self, monkeypatch):
        # flag-off contract: fusion replaces this veto (test_ctcfill_fusion)
        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        monkeypatch.setattr(ctcfill, "available", lambda: True)
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [("strip", "Home World:",
                                                    0.9)])
        reads = {0: ("Proxima-b", 0.9), 1: ("Barnard-c", 0.9)}
        monkeypatch.setattr(ctcfill, "_accept_strips",
                            lambda strips, field, _r=iter([0, 1]), **kw:
                            reads[next(_r)])
        assert ctcfill.fill({0: 0, 1: 1}, ["home_world"]) == {}

    def test_agreeing_pages_fill_with_capped_conf(self, monkeypatch):
        # patches _accept_strips, which the fused fill() bypasses
        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        monkeypatch.setattr(ctcfill, "available", lambda: True)
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [("strip", "Home World:",
                                                    0.9)])
        monkeypatch.setattr(ctcfill, "_accept_strips",
                            lambda strips, field, **kw: ("Proxima-b", 0.9))
        got = ctcfill.fill({0: 0, 1: 1}, ["home_world"])
        assert got == {"home_world": ("Proxima-b", ctcfill.CONF_CAP)}

    def test_budget_exhaustion_stops(self, monkeypatch):
        monkeypatch.setattr(ctcfill, "available", lambda: True)
        calls = []
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: calls.append(1) or [])
        got = ctcfill.fill({0: 0}, ["home_world"], budget_left=lambda: False)
        assert got == {} and calls == []

    def test_grid_fallback_only_for_unaccepted_fields(self, monkeypatch):
        """A field the word-box path ACCEPTED never touches the grid; a
        field it did not accept falls through (one cached fit per
        page)."""
        # patches _accept_strips, which the fused fill() bypasses
        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        monkeypatch.setattr(ctcfill, "available", lambda: True)
        monkeypatch.setattr(
            ctcfill, "locate_strips",
            lambda result, field: [("strip", "Home World:", 0.9)]
            if field == "home_world" else [])

        def fake_accept(strips, field, **kw):
            if strips == ["GRID"]:
                return ("XW-1", 0.5) if field == "visa_class" else None
            return ("Proxima-b", 0.9) if strips else None

        monkeypatch.setattr(ctcfill, "_accept_strips", fake_accept)
        fitted = []
        monkeypatch.setattr(ctcfill, "page_grid_fit",
                            lambda result: fitted.append(1) or "FIT")
        grid_asked = []
        monkeypatch.setattr(
            ctcfill, "grid_strips",
            lambda result, field, fit=None:
            grid_asked.append(field) or ["GRID"])
        result = _result([])
        got = ctcfill.fill({0: result}, ["home_world", "visa_class"])
        assert got == {"home_world": ("Proxima-b", ctcfill.CONF_CAP),
                       "visa_class": ("XW-1", ctcfill.CONF_CAP)}
        assert "home_world" not in grid_asked   # accepted via word boxes
        assert fitted == [1]                    # one cached fit per page

    def test_grid_fallback_skips_foreign_typed_pages(self, monkeypatch):
        monkeypatch.setattr(ctcfill, "available", lambda: True)
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [])
        fitted = []
        monkeypatch.setattr(ctcfill, "page_grid_fit",
                            lambda result: fitted.append(1) or None)
        result = _result([])
        got = ctcfill.fill({0: result}, ["home_world"],
                           page_types={0: "attestation"})
        assert got == {} and fitted == []     # never fitted a foreign page


# --------------------------------------- pipeline wiring (real PDF, e2e)


def _make_scan_pdf(tmp_path, page_specs, case_id=CASE_ID):
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


# Intake with NO home-world row: the field stays unread through every
# engine, and the fee row keeps the rest of the case quiet.
INTAKE_NO_WORLD = [
    "FORM I-8090: Extraterrestrial Work Authorization Intake",
    f"Case ID: {CASE_ID}",
    "Applicant: Solul Zamora",
    "Species Code: ORION_GRAYS",
    "Visa Class: DIP-1",
    "Arrival Date: 2026-05-04",
    "Declared Purpose: research",
    "Fee Status: paid",
]
INTAKE_WITH_WORLD = INTAKE_NO_WORLD[:4] + ["Home World: Proxima-b"] \
    + INTAKE_NO_WORLD[4:]


def _quiet_escalation(monkeypatch):
    from mib_pipeline import crnn

    monkeypatch.setattr(ocr, "rapid_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "escalation_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "weld_sponsor_lines", lambda *a, **k: [])
    monkeypatch.setattr(crnn, "crnn_lines", lambda *a, **k: [])


class TestIntegrationPipeline:
    def test_flag_off_block_is_dead_code(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIB_CTCFILL", raising=False)
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_NO_WORLD])
        calls = []
        monkeypatch.setattr(ctcfill, "fill",
                            lambda *a, **k: calls.append(1) or {})
        row = pipeline.process_pdf(pdf)
        assert calls == []
        assert row["home_world"] == "Luyten-b"        # writer mode default

    def test_flag_on_fills_only_unread_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_NO_WORLD])
        asked = {}

        def fake_fill(scans, fields_needed, budget_left=lambda: True,
                      page_types=None):
            asked["fields"] = list(fields_needed)
            asked["page_types"] = page_types
            return {"home_world": ("Proxima-b", 0.5)}

        monkeypatch.setattr(ctcfill, "fill", fake_fill)
        row = pipeline.process_pdf(pdf)
        # only the genuinely unread menu field is asked about
        assert asked["fields"] == ["home_world"]
        assert row["home_world"] == "Proxima-b"       # the fill went through

    def test_fill_is_extraction_only_policy_stays_blind(self, tmp_path,
                                                        monkeypatch):
        """The filled value lands in the OUTPUT ROW but the decision path
        still treats the field as unread: known stays False, so R12 keeps
        the case in review exactly as without the lever."""
        monkeypatch.setenv("MIB_CTCFILL", "1")
        _quiet_escalation(monkeypatch)
        # visa read affirmatively as XW-1 -> home world matters (non-DIP)
        spec = [l for l in INTAKE_NO_WORLD]
        spec[4] = "Visa Class: XW-1"
        pdf = _make_scan_pdf(tmp_path, [spec])
        monkeypatch.setattr(
            ctcfill, "fill",
            lambda *a, **k: {"home_world": ("Wolf-1061c", 0.5)})
        row = pipeline.process_pdf(pdf)
        assert row["home_world"] == "Wolf-1061c"
        # a KNOWN Wolf-1061c on a non-DIP visa would be R5 DENIED; the
        # capped fill must never reach the policy layer
        assert row["adjudication"] == "NEEDS_REVIEW"
        assert row["_path"] in ("R12_world_unread", "R12_flags_unread",
                                "R12_sponsor_unread", "R9_arrival_not_visible")

    def test_fill_never_overrides_an_affirmative_read(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL", "1")
        _quiet_escalation(monkeypatch)
        pdf = _make_scan_pdf(tmp_path, [INTAKE_WITH_WORLD])
        monkeypatch.setattr(
            ctcfill, "fill",
            lambda *a, **k: {"home_world": ("Barnard-c", 0.5)})
        row = pipeline.process_pdf(pdf)
        # the affirmative read stands; the block never even asks for the
        # field (need list is empty -> fill's output is ignored for it)
        assert row["home_world"] == "Proxima-b"
