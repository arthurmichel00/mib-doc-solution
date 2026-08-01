"""Anchor-scored rotation probe for baked-in rotations (backlog item A2).

The probe is a rotation-SELECTION layer for the escalation engines: on
pages whose footer-stripped primary OCR is near-empty (<100 chars AND <2
form anchors) OR whose orientation the load ladder had to estimate, it
OCRs a downscaled render at np.rot90 turns (0, 1, 3, 2) and scores each
candidate by anchors + recognizable-word content + confidence, with the
always-upright vector footer stripped so a wrong rotation can't win on
its readable footer. Everything is gated behind MIB_ROT_PROBE=1; with the
flag unset the pipeline must behave byte-identically to the shipped code.

Idea: tylergibbs1 028ba78 (credited to naidx0) + kirtandesai memo (MIT).
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ocr, pipeline
from mib_pipeline.model import Line, Source
from mib_pipeline.ocr import OcrWord, ScanOcrResult


# --------------------------------------------------------------------------
# helpers

def _line(text: str, conf: float = 0.9) -> Line:
    return Line(text=text, page_index=0, source=Source.OCR, conf=conf)


def _words(texts: list[str], conf: float = 0.9) -> list[OcrWord]:
    return [
        OcrWord(text=t, conf=conf, line_key=(0, 0, i), x=float(i))
        for i, t in enumerate(texts)
    ]


_MARK = 250          # corner marker brightness
_MS = 20             # marker block size in PROBE (downscaled) pixels


def _marked_gray(rows: int = 200, cols: int = 400) -> np.ndarray:
    """Gray image with a bright top-left block so a fake engine can tell
    which np.rot90 turn it was handed after the probe's downscale."""
    gray = np.full((rows, cols), 128, dtype=np.uint8)
    gray[: _MS * 2, : _MS * 2] = _MARK
    return gray


def _infer_rotation(img: np.ndarray) -> int:
    """Recover the np.rot90 k applied to a _marked_gray image (any scale)."""
    m = _MS // 2  # generous inner corner window, survives resizing
    corners = {
        0: img[:m, :m],
        1: img[-m:, :m],
        2: img[-m:, -m:],
        3: img[:m, -m:],
    }
    bright = [k for k, region in corners.items() if region.mean() > 200]
    assert len(bright) == 1, f"ambiguous marker corners: {bright}"
    return bright[0]


class FakeProbeEngine:
    """OcrEngine double: scripted words per rotation, records probe order."""

    def __init__(self, by_rotation: dict[int, list[OcrWord]]):
        self.by_rotation = by_rotation
        self.calls: list[int] = []

    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        k = _infer_rotation(image)
        self.calls.append(k)
        return self.by_rotation.get(k, [])


class ExplodingEngine:
    def words(self, image: np.ndarray, sparse: bool = False) -> list[OcrWord]:
        raise AssertionError("engine must not be called")


FOOTER_WORDS = "Packet MIB-000123 / page 2 Synthetic hiring challenge document".split()

# ≥2 anchors and ≥80 footer-stripped chars: the early-exit profile.
RICH_WORDS = ("FORM I-771 ENTRY INTAKE Applicant Name: Zarel Vantox "
              "Species: ORION_GRAYS Home World: Titan Freeport "
              "Fee Status: paid").split()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MIB_ROT_PROBE", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("MIB_ROT_PROBE", raising=False)


# --------------------------------------------------------------------------
# env-flag gate

class TestFlagGate:
    def test_unset_is_disabled(self, flag_off):
        assert pipeline._rot_probe_enabled() is False

    def test_zero_is_disabled(self, monkeypatch):
        monkeypatch.setenv("MIB_ROT_PROBE", "0")
        assert pipeline._rot_probe_enabled() is False

    def test_one_enables(self, flag_on):
        assert pipeline._rot_probe_enabled() is True

    def test_flag_off_probes_nothing_and_never_touches_the_engine(
            self, flag_off):
        result = ScanOcrResult(lines=[_line("~~ junk")], gray=_marked_gray(),
                               upright=True)
        sel = pipeline._rot_probe_selections(
            {0: result}, ExplodingEngine(), lambda: True)
        assert sel == {}


# --------------------------------------------------------------------------
# trigger criterion: footer-stripped text <100 chars AND <2 form anchors

class TestTrigger:
    def test_clean_page_does_not_trigger(self):
        lines = [
            _line("FORM I-771 ENTRY INTAKE"),
            _line("Applicant Name: Zarel Vantox"),
            _line("Species: ORION_GRAYS"),
            _line("Home World: Titan Freeport"),
            _line("Arrival Date: 2026-05-04"),
        ]
        assert not pipeline._rot_probe_trigger(lines)

    def test_sparse_junk_page_triggers(self):
        assert pipeline._rot_probe_trigger([_line("|| ~~"), _line("e3 a")])

    def test_empty_page_triggers(self):
        assert pipeline._rot_probe_trigger([])

    def test_footer_only_page_triggers(self):
        lines = [
            _line("Packet MIB-000123 / page 2", conf=0.97),
            _line("Synthetic hiring challenge document", conf=0.97),
        ]
        assert pipeline._rot_probe_trigger(lines)

    def test_two_anchors_veto_the_trigger_even_when_short(self):
        lines = [_line("Fee Status: paid"), _line("Observed Flags: none")]
        assert not pipeline._rot_probe_trigger(lines)

    def test_long_anchorless_text_does_not_trigger(self):
        text = "the quick brown fox jumps over the lazy dog again and again"
        assert not pipeline._rot_probe_trigger([_line(text), _line(text)])

    def test_low_confidence_text_does_not_count(self):
        text = "the quick brown fox jumps over the lazy dog again and again"
        lines = [_line(text, conf=0.3), _line(text, conf=0.3)]
        assert pipeline._rot_probe_trigger(lines)

    def test_anchor_matches_are_word_bounded(self):
        # FORM inside PERFORMANCE / ARRIVAL inside ARRIVALS must not count.
        lines = [_line("PERFORMANCE INFORMATION ARRIVALS misc filler")]
        assert pipeline._rot_probe_trigger(lines)


# --------------------------------------------------------------------------
# footer stripping

class TestFooterStrip:
    def test_strips_both_footer_lines(self):
        text = ("Packet MIB-000123 / page 2 "
                "Synthetic hiring challenge document")
        assert pipeline._strip_footer(text) == ""

    def test_strips_case_id_tokens(self):
        assert pipeline._strip_footer("MIB-000802") == ""

    def test_keeps_content(self):
        out = pipeline._strip_footer("Fee Status: paid Packet MIB-000001 / "
                                     "page 3 Sponsor ID: SPN-2244")
        assert "Fee Status: paid" in out
        assert "SPN-2244" in out
        assert "Packet" not in out

    def test_upside_down_page_cannot_win_on_its_readable_footer(self):
        # k=0 shows the always-upright vector footer perfectly; the real
        # content only reads at k=2 (a 180-degree baked-in rotation) and is
        # modest: one anchor, a handful of words. Without footer stripping,
        # k=0 outscores it on crisp footer text.
        engine = FakeProbeEngine({
            0: _words(FOOTER_WORDS, conf=0.97),
            2: _words("Sponsor ID: SPN-2244 record attached".split(),
                      conf=0.7),
        })
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert choice is not None
        assert choice.k == 2


# --------------------------------------------------------------------------
# probe order, early exit, determinism

class TestProbeOrder:
    def test_rotations_probed_in_0_1_3_2_order(self):
        engine = FakeProbeEngine({})  # nothing readable anywhere
        pipeline._probe_rotation(_marked_gray(), engine)
        assert engine.calls == [0, 1, 3, 2]

    def test_early_exit_on_first_rotation(self):
        engine = FakeProbeEngine({0: _words(RICH_WORDS)})
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert (choice.k, choice.confident) == (0, True)
        assert engine.calls == [0]

    def test_early_exit_mid_sequence(self):
        engine = FakeProbeEngine({1: _words(RICH_WORDS)})
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert (choice.k, choice.confident) == (1, True)
        assert engine.calls == [0, 1]

    def test_one_anchor_is_not_confident_and_probes_everything(self):
        engine = FakeProbeEngine({
            3: _words("Sponsor ID: SPN-2244 record attached".split(),
                      conf=0.7),
        })
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert (choice.k, choice.confident) == (3, False)
        assert engine.calls == [0, 1, 3, 2]

    def test_tie_is_broken_by_probe_order(self):
        same = "Sponsor ID: SPN-2244 record attached".split()
        engine = FakeProbeEngine({k: _words(same, conf=0.7)
                                  for k in (0, 1, 2, 3)})
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert choice.k == 0

    def test_repeated_runs_are_deterministic(self):
        by_rot = {
            1: _words("Sponsor ID: SPN-2244 filler words here".split(),
                      conf=0.71),
            2: _words("Registry row content extra tokens".split(), conf=0.72),
        }
        picks = {
            pipeline._probe_rotation(_marked_gray(),
                                     FakeProbeEngine(by_rot)).k
            for _ in range(3)
        }
        assert len(picks) == 1

    def test_junk_everywhere_is_inconclusive(self):
        engine = FakeProbeEngine({
            k: _words(["e3", "||", "~~"], conf=0.8) for k in (0, 1, 2, 3)
        })
        assert pipeline._probe_rotation(_marked_gray(), engine) is None

    def test_anchor_beats_high_confidence_junk_volume(self):
        junk = [f"x{i}q" for i in range(80)]  # ~300 chars of confident junk
        engine = FakeProbeEngine({
            1: _words(junk, conf=0.9),
            2: _words("Sponsor ID: SPN-2244 record".split(), conf=0.6),
        })
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert choice.k == 2

    def test_recognizable_words_outrank_junk_characters(self):
        # The kirtandesai words-variant's discriminating case: with no
        # anchors anywhere, a flipped page's high-volume confident junk
        # must lose to a modest read made of real document words (a
        # character count would have preferred the junk).
        junk = [f"x{i}q" for i in range(80)]
        engine = FakeProbeEngine({
            1: _words(junk, conf=0.9),
            2: _words("paid receipt amount waiver intake attestation".split(),
                      conf=0.8),
        })
        choice = pipeline._probe_rotation(_marked_gray(), engine)
        assert choice.k == 2


# --------------------------------------------------------------------------
# selection loop

class TestSelections:
    def test_probes_only_triggered_pages_within_budget(self, flag_on):
        sparse = ScanOcrResult(lines=[_line("~~")], gray=_marked_gray(),
                               upright=True)
        clean = ScanOcrResult(
            lines=[_line("Fee Status: paid"), _line("Observed Flags: none")],
            gray=_marked_gray(), upright=True)
        engine = FakeProbeEngine({1: _words(RICH_WORDS)})
        sel = pipeline._rot_probe_selections(
            {0: sparse, 1: clean}, engine, lambda: True)
        assert set(sel) == {0}
        assert (sel[0].k, sel[0].confident) == (1, True)

    def test_estimator_run_page_is_probed_even_when_text_rich(self, flag_on):
        # A page the ladder could not read upright gets its orientation
        # ESTIMATED — the estimate is a guess (MIB-000321 p3 was scored
        # upright yet its content is at k=1), so the probe re-scores it
        # even though the pooled counter-rotation passes left plenty of
        # text on the page.
        rich_lines = [_line("Fee Status: paid"), _line("Observed Flags: none"),
                      _line("the quick brown fox jumps over the lazy dog")]
        estimated = ScanOcrResult(lines=rich_lines, gray=_marked_gray(),
                                  upright=True, orientation_estimated=True)
        engine = FakeProbeEngine({1: _words(RICH_WORDS)})
        sel = pipeline._rot_probe_selections({0: estimated}, engine,
                                             lambda: True)
        assert (sel[0].k, sel[0].confident) == (1, True)

    def test_readable_upright_page_is_never_probed(self, flag_on):
        rich_lines = [_line("Fee Status: paid"), _line("Observed Flags: none")]
        readable = ScanOcrResult(lines=rich_lines, gray=_marked_gray(),
                                 upright=True, orientation_estimated=False)
        sel = pipeline._rot_probe_selections({0: readable}, ExplodingEngine(),
                                             lambda: True)
        assert sel == {}

    def test_budget_exhausted_probes_nothing(self, flag_on):
        sparse = ScanOcrResult(lines=[], gray=_marked_gray(), upright=True)
        sel = pipeline._rot_probe_selections(
            {0: sparse}, ExplodingEngine(), lambda: False)
        assert sel == {}

    def test_inconclusive_probe_selects_nothing(self, flag_on):
        sparse = ScanOcrResult(lines=[], gray=_marked_gray(), upright=True)
        sel = pipeline._rot_probe_selections(
            {0: sparse}, FakeProbeEngine({}), lambda: True)
        assert sel == {}


# --------------------------------------------------------------------------
# escalation views: legacy path untouched, probe path selects

class TestEscalationViews:
    def test_upright_legacy_view_is_the_raw_gray(self):
        gray = _marked_gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=True)
        views = pipeline._escalation_views(result)
        assert len(views) == 1 and views[0] is gray

    def test_non_upright_legacy_views_unchanged(self, monkeypatch):
        # multi-view escalation is itself gated now (MIB_ESC_VIEWS; see
        # test_escalation_views_gate.py) — this test documents the
        # gate-ON behavior that predates the A2 probe.
        monkeypatch.setenv("MIB_ESC_VIEWS", "1")
        gray = _marked_gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=False, best_rot=1)
        views = pipeline._escalation_views(result)
        expected = [ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, k=k)))
                    for k in {1, 3}]
        assert len(views) == len(expected)
        for got, want in zip(views, expected):
            assert np.array_equal(got, want)

    def test_confident_probe_yields_single_selected_view(self):
        gray = _marked_gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=False, best_rot=1)
        choice = pipeline._RotChoice(k=2, confident=True)
        views = pipeline._escalation_views(result, choice)
        assert len(views) == 1
        assert np.array_equal(
            views[0], ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, 2))))

    def test_unconfident_probe_keeps_the_180_hedge(self):
        gray = _marked_gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=True)
        choice = pipeline._RotChoice(k=1, confident=False)
        views = pipeline._escalation_views(result, choice)
        assert len(views) == 2
        assert np.array_equal(
            views[0], ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, 1))))
        assert np.array_equal(
            views[1], ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, 3))))

    def test_unconfident_upright_probe_pads_and_hedges_180(self):
        gray = _marked_gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=True)
        choice = pipeline._RotChoice(k=0, confident=False)
        views = pipeline._escalation_views(result, choice)
        assert len(views) == 2
        assert np.array_equal(views[0], ocr.pad_for_ocr(gray))
        assert np.array_equal(
            views[1], ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, 2))))
