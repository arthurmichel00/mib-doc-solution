"""Likelihood fusion for the closed-menu CTC fill (MIB_CTCFILL_FUSION=1).

The flag-off scorer gates each view of a field independently — two
preprocessing variants per strip, every label instance, every page — and
vetoes the field whenever two views name different values. Fusion pools
those views instead: the per-candidate length-normalized CTC log-probs
are AVERAGED into one list and the calibrated gate fires once on it.

These tests drive the real scorer (score_strip + the exact CTC forward)
over SYNTHETIC posterior grids on a tiny synthetic charset, so no ONNX
session is ever created: posteriorgram is the only thing stubbed, which
is also what lets the inference-count parity test count model calls.

Grid shape: four frames spelling the label "L:" then one value glyph, so
each candidate "L: <v>" has exactly one CTC path and its score is
log(p_v) / 4 — the value-frame mass is the whole experiment.
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ctcfill
from mib_pipeline.ocr import OcrWord, ScanOcrResult

needs_model = pytest.mark.skipif(not ctcfill.available(),
                                 reason="rec bundle not mounted")

# tiny charset: blank + the glyphs the synthetic grids can emit
_CHARSET = ["<blank>", "L", ":", " ", "A", "B", "C"]
_LABEL = "L:"
_MENU = ("A", "B", "C")
_IX = {"A": 4, "B": 5, "C": 6}

# The value the pooled evidence should recover. Neither single view's
# winner: each view has its own favourite and they contradict.
GOLD = "C"


def _grid(**mass: float) -> np.ndarray:
    """(T, C) log-posteriors: frames 0-2 spell "L:" + a space, frame 3
    carries the value-glyph distribution."""
    p = np.full((4, len(_CHARSET)), 1e-6)
    p[0, 1] = 1.0                       # "L"
    p[1, 2] = 1.0                       # ":"
    p[2, 3] = 1.0                       # " "
    for value, m in mass.items():
        p[3, _IX[value]] = m
    p[3, 0] = max(1e-6, 1.0 - sum(mass.values()))
    return np.log(p)


# Two views that disagree on their argmax. Each one's winner is wrong,
# and each one's margin over C is too thin to clear MARGIN_RUNNER_UP —
# but C is the runner-up in BOTH, so it wins the average outright.
VIEW_A = _grid(A=0.60, C=0.35, B=0.005)
VIEW_B = _grid(B=0.60, C=0.35, A=0.005)
# A view that needs no help: C by a mile, accepted with or without the
# flag (fusing a view with itself must return that view unchanged).
VIEW_CLEAR = _grid(C=0.90, A=0.02, B=0.02)

# Strips are only ever dispatch keys here (posteriorgram is stubbed).
# min() distinguishes a raw crop from its MINMAX-stretched variant;
# argmax() survives the stretch, so it distinguishes pages instead.
STRIP_RAW_MIN = 100
PAGE_A = np.array([[100, 150]], np.uint8)      # argmax at index 1
PAGE_B = np.array([[150, 100]], np.uint8)      # argmax at index 0


class _FakePosteriorgram:
    """Stub with a call counter — the inference budget under test."""

    def __init__(self, rule):
        self._rule = rule
        self.calls = 0

    def __call__(self, img, scale=1.0):
        # `scale` is accepted but ignored: a sibling lever adds a second
        # rec resize as posteriorgram(strip, scale), and a stub that only
        # took one argument would break on merge rather than on merit.
        self.calls += 1
        return self._rule(np.asarray(img))


@pytest.fixture(autouse=True)
def _margin_layer_off(monkeypatch):
    """This file measures fusion on its own. A sibling lever adds a
    second-resolution confirmation under MIB_CTCFILL_MARGIN that composes
    with fusion — covered by test_ctcfill_fusion_margin.py after the
    merge — so pin it off here and an ambient flag cannot silently change
    what these tests measure. No-op while that lever is unmerged."""
    monkeypatch.delenv("MIB_CTCFILL_MARGIN", raising=False)


@pytest.fixture
def tiny(monkeypatch):
    """Synthetic charset + menu, no ONNX session, clean label cache."""
    monkeypatch.setattr(ctcfill, "_session", lambda: None)
    monkeypatch.setattr(ctcfill, "_CHARS", list(_CHARSET))
    monkeypatch.setattr(ctcfill, "_CHAR_TO_IX",
                        {c: i for i, c in enumerate(_CHARSET)})
    monkeypatch.setitem(ctcfill._MENUS, "home_world", _MENU)
    monkeypatch.setattr(ctcfill, "available", lambda: True)
    ctcfill._label_ixs.cache_clear()
    yield
    ctcfill._label_ixs.cache_clear()


def _by_variant(img):
    """raw crop -> VIEW_A, its stretched variant -> VIEW_B."""
    return VIEW_A if img.min() == STRIP_RAW_MIN else VIEW_B


def _by_page(img):
    """page A (both variants) -> VIEW_A, page B -> VIEW_B."""
    return VIEW_A if int(img.argmax()) == 1 else VIEW_B


def _install(monkeypatch, rule, strips_of=lambda result, field:
             [(result, _LABEL, 0.9)]):
    post = _FakePosteriorgram(rule)
    monkeypatch.setattr(ctcfill, "posteriorgram", post)
    monkeypatch.setattr(ctcfill, "locate_strips", strips_of)
    monkeypatch.setattr(ctcfill, "page_grid_fit", lambda result: None)
    return post


# --------------------------------------------------------------- flag gate


class TestFlagGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        assert ctcfill.fusion_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        assert ctcfill.fusion_enabled() is True
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "0")
        assert ctcfill.fusion_enabled() is False


# ------------------------------------------------------- fusion arithmetic


class TestFuseScores:
    def test_averages_rather_than_sums(self):
        fused = ctcfill.fuse_scores([[(-1.0, "A")], [(-3.0, "A")]])
        assert fused == [(-2.0, "A")]           # sum would be -4.0

    def test_scale_is_preserved_so_the_floors_stay_valid(self):
        """N copies of one view fuse back to that view exactly: the
        fused list lives on the single-view scale SCORE_FLOOR and the
        margins were calibrated against."""
        view = [(-0.5, "A"), (-2.25, "B"), (-9.0, "<none>")]
        assert ctcfill.fuse_scores([view] * 4) == view

    def test_best_first(self):
        fused = ctcfill.fuse_scores([[(-9.0, "A"), (-1.0, "B")]])
        assert [v for _, v in fused] == ["B", "A"]

    def test_absent_candidate_averaged_over_views_that_scored_it(self):
        """A candidate one view could not score is not penalised for the
        absence, and stays on the single-view scale."""
        fused = dict((v, s) for s, v in ctcfill.fuse_scores(
            [[(-1.0, "A"), (-2.0, "B")], [(-3.0, "A")]]))
        assert fused["A"] == pytest.approx(-2.0)
        assert fused["B"] == pytest.approx(-2.0)

    def test_empty_input(self):
        assert ctcfill.fuse_scores([]) == []


# ------------------------------------------------------- multi-view fusion


class TestMultiViewFusion:
    def _scored(self, grid):
        return ctcfill.score_strip(grid, _LABEL, _MENU)

    def test_setup_is_the_hard_case(self, tiny, monkeypatch):
        """Each view's winner is wrong AND unacceptable on its own: the
        two views disagree on argmax, and neither margin clears the
        gate. Flag off there is nothing to accept."""
        monkeypatch.setattr(ctcfill, "posteriorgram",
                            lambda img, scale=1.0: img)
        view_a = self._scored(VIEW_A)
        view_b = self._scored(VIEW_B)
        assert view_a[0][1] == "A" and view_b[0][1] == "B"
        assert view_a[0][1] != GOLD and view_b[0][1] != GOLD
        assert ctcfill._gated_winner(view_a) is None
        assert ctcfill._gated_winner(view_b) is None

    def test_fused_list_picks_the_best_average(self, tiny, monkeypatch):
        monkeypatch.setattr(ctcfill, "posteriorgram",
                            lambda img, scale=1.0: img)
        fused = ctcfill.fuse_scores([self._scored(VIEW_A),
                                     self._scored(VIEW_B)])
        assert fused[0][1] == GOLD
        assert ctcfill._gated_winner(fused) == GOLD

    def test_flag_off_abstains_flag_on_accepts_gold(self, tiny,
                                                    monkeypatch):
        _install(monkeypatch, _by_variant)
        strips = [(PAGE_A, _LABEL, 0.9)]

        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        assert ctcfill._accept_strips(strips, "home_world") is None

        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        assert ctcfill._accept_strips(strips, "home_world") == \
            (GOLD, ctcfill.CONF_CAP)


# ------------------------------------------------------- cross-page fusion


class TestCrossPageFusion:
    def test_recovers_a_value_neither_page_reaches_alone(self, tiny,
                                                         monkeypatch):
        """Both pages read consistently within themselves (their two
        variants agree), so the flag-off cross-page AGREEMENT rule sees
        two pages naming different values and vetoes; each page also
        fails its own gate. Pooled, C wins."""
        _install(monkeypatch, _by_page)
        scans = {0: PAGE_A, 1: PAGE_B}

        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        assert ctcfill.fill(scans, ["home_world"]) == {}

        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        assert ctcfill.fill(scans, ["home_world"]) == \
            {"home_world": (GOLD, ctcfill.CONF_CAP)}

    def test_page_that_located_nothing_contributes_no_penalty(
            self, tiny, monkeypatch):
        """A third page whose strip failed to locate must leave the
        fused verdict exactly where the two real pages put it."""
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        _install(monkeypatch, _by_page)
        two = ctcfill.fill({0: PAGE_A, 1: PAGE_B}, ["home_world"])

        _install(monkeypatch, _by_page,
                 strips_of=lambda result, field:
                 [] if result is None else [(result, _LABEL, 0.9)])
        three = ctcfill.fill({0: PAGE_A, 1: None, 2: PAGE_B},
                             ["home_world"])
        assert three == two == {"home_world": (GOLD, ctcfill.CONF_CAP)}

    def test_pages_weigh_equally_regardless_of_strip_count(self, tiny,
                                                           monkeypatch):
        """One page printing the label three times must not outvote a
        page printing it once: strips pool within a page first."""
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        _install(monkeypatch, _by_page,
                 strips_of=lambda result, field:
                 [(result, _LABEL, 0.9)] * (3 if result is PAGE_A else 1))
        assert ctcfill.fill({0: PAGE_A, 1: PAGE_B}, ["home_world"]) == \
            {"home_world": (GOLD, ctcfill.CONF_CAP)}


# ---------------------------------------------------------- grid fallback


class TestGridFallbackUnderFusion:
    """The grid pass keeps its own pooled decision under GRID_* floors:
    word-box and grid views are never averaged together (different
    strip scales, separately calibrated gates)."""

    def _setup(self, monkeypatch, word_box_view, grid_view):
        posts = []

        def rule(img):
            posts.append(int(img[0, 0]))
            return grid_view if int(img[0, 0]) == 7 else word_box_view

        monkeypatch.setattr(ctcfill, "posteriorgram",
                            _FakePosteriorgram(rule))
        monkeypatch.setattr(ctcfill, "page_grid_fit", lambda result: "FIT")
        monkeypatch.setattr(ctcfill, "grid_strips",
                            lambda result, field, fit=None:
                            [(np.array([[7, 7]], np.uint8), _LABEL, 0.4)])
        return posts

    def test_word_box_acceptance_never_touches_the_grid(self, tiny,
                                                        monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [(result, _LABEL, 0.9)])
        posts = self._setup(monkeypatch, VIEW_CLEAR, VIEW_A)
        assert ctcfill.fill({0: PAGE_A}, ["home_world"]) == \
            {"home_world": ("C", ctcfill.CONF_CAP)}
        assert 7 not in posts               # grid strip never scored

    def test_field_the_word_box_pass_rejected_falls_through(self, tiny,
                                                            monkeypatch):
        """Located but gate-rejected still falls through to the grid,
        exactly as the flag-off path does."""
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [(result, _LABEL, 0.9)])
        # word-box views are the ambiguous pair (rejected on one page),
        # the grid strip sees the value cleanly
        posts = self._setup(monkeypatch, VIEW_A, VIEW_CLEAR)
        assert ctcfill.fill({0: PAGE_A}, ["home_world"]) == \
            {"home_world": ("C", 0.4)}
        assert 7 in posts                   # grid strip was scored

    def test_foreign_typed_page_still_skipped(self, tiny, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [])
        fitted = []
        monkeypatch.setattr(ctcfill, "posteriorgram",
                            _FakePosteriorgram(lambda img: VIEW_CLEAR))
        monkeypatch.setattr(ctcfill, "page_grid_fit",
                            lambda result: fitted.append(1) or "FIT")
        assert ctcfill.fill({0: PAGE_A}, ["home_world"],
                            page_types={0: "attestation"}) == {}
        assert fitted == []


# ----------------------------------------------------------- safety gates


class TestSafetyGatesOnFusedWinner:
    def test_hard_embargo_winner_abstains(self):
        fused = [(-0.5, "TRAPPIST-1e"), (-4.0, "Proxima-b"),
                 (-9.0, "<none>")]
        assert ctcfill._accept_fused(fused, "home_world",
                                     ctcfill.SCORE_FLOOR,
                                     ctcfill.MARGIN_RUNNER_UP) is None

    def test_mode_default_winner_suppressed(self):
        fused = [(-0.5, ctcfill._mode_default("home_world")),
                 (-4.0, "Proxima-b"), (-9.0, "<none>")]
        assert ctcfill._accept_fused(fused, "home_world",
                                     ctcfill.SCORE_FLOOR,
                                     ctcfill.MARGIN_RUNNER_UP) is None

    def test_floor_and_margin_still_bite(self):
        below = [(ctcfill.SCORE_FLOOR - 0.1, "Proxima-b"), (-9.0, "<none>")]
        thin = [(-0.5, "Proxima-b"),
                (-0.5 - ctcfill.MARGIN_RUNNER_UP + 0.01, "Barnard-c"),
                (-9.0, "<none>")]
        none_wins = [(-0.5, "<none>"), (-1.0, "Proxima-b")]
        for fused in (below, thin, none_wins):
            assert ctcfill._accept_fused(fused, "home_world",
                                         ctcfill.SCORE_FLOOR,
                                         ctcfill.MARGIN_RUNNER_UP) is None

    def test_confidence_capped(self, tiny, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        _install(monkeypatch, lambda img: VIEW_CLEAR)
        got = ctcfill._accept_strips([(PAGE_A, _LABEL, 0.99)], "home_world")
        assert got == ("C", ctcfill.CONF_CAP)
        assert got[1] <= ctcfill.CONF_CAP


# ------------------------------------------------------- flag-off identity


class TestFlagOffIdentity:
    def _run(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        else:
            monkeypatch.setenv("MIB_CTCFILL_FUSION", value)
        _install(monkeypatch, _by_page)
        return ctcfill.fill({0: PAGE_A, 1: PAGE_B}, ["home_world"])

    def test_unset_and_zero_are_the_same_run(self, tiny, monkeypatch):
        assert self._run(monkeypatch, None) == self._run(monkeypatch, "0")

    def test_flag_off_output_unchanged_where_the_flag_on_output_differs(
            self, tiny, monkeypatch):
        """The identity check is only worth anything if the flag-on path
        actually diverges on this input."""
        assert self._run(monkeypatch, None) == {}
        assert self._run(monkeypatch, "1") == \
            {"home_world": (GOLD, ctcfill.CONF_CAP)}

    def test_agreeing_views_give_identical_output_either_way(
            self, tiny, monkeypatch):
        """Where the flag-off path already accepts, fusion returns the
        same value at the same confidence — the mean of agreeing views
        is those views."""
        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        _install(monkeypatch, lambda img: VIEW_CLEAR)
        off = ctcfill.fill({0: PAGE_A, 1: PAGE_B}, ["home_world"])

        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        _install(monkeypatch, lambda img: VIEW_CLEAR)
        on = ctcfill.fill({0: PAGE_A, 1: PAGE_B}, ["home_world"])

        assert off == on == {"home_world": ("C", ctcfill.CONF_CAP)}

    def test_flag_off_never_enters_the_fusion_code(self, tiny,
                                                   monkeypatch):
        """Byte-identity by construction: with the flag unset no fusion
        helper is reachable, so the flag-off scorer cannot have moved."""
        monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        _install(monkeypatch, _by_page)
        seen = []
        monkeypatch.setattr(ctcfill, "fuse_scores",
                            lambda lists: seen.append(1) or [])
        ctcfill.fill({0: PAGE_A, 1: PAGE_B}, ["home_world"])
        ctcfill._accept_strips([(PAGE_A, _LABEL, 0.9)], "home_world")
        ctcfill.read_field(PAGE_A, "home_world")
        assert seen == []


# --------------------------------------------------------- zero new cost


class TestInferenceParity:
    def _count(self, monkeypatch, value, scans):
        if value is None:
            monkeypatch.delenv("MIB_CTCFILL_FUSION", raising=False)
        else:
            monkeypatch.setenv("MIB_CTCFILL_FUSION", value)
        post = _install(monkeypatch, _by_page)
        ctcfill.fill(scans, ["home_world"])
        return post.calls

    def test_same_number_of_model_calls_with_and_without_fusion(
            self, tiny, monkeypatch):
        """Fusion is arithmetic on scores the flag-off path already
        computes: two pages x two preprocessing variants = four
        posteriorgram calls, flag on or off."""
        scans = {0: PAGE_A, 1: PAGE_B}
        off = self._count(monkeypatch, None, scans)
        on = self._count(monkeypatch, "1", scans)
        assert off == on == 4

    def test_parity_holds_on_the_single_page_path(self, tiny, monkeypatch):
        scans = {0: PAGE_A}
        assert self._count(monkeypatch, None, scans) == \
            self._count(monkeypatch, "1", scans) == 2


# ----------------------------------------- real recognizer, real strips


@needs_model
class TestRealModelUnderFusion:
    """A handful of strips through the actual PP-OCRv6 head: the flag-on
    path must keep every behaviour the flag-off path is trusted for."""

    def _result_for(self, text, tokens=("Home", "World:")):
        import cv2

        img = np.full((300, 900), 255, np.uint8)
        row = np.full((44, 900), 255, np.uint8)
        cv2.putText(row, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0,
                    2, cv2.LINE_AA)
        img[100:144, 40:860] = row[:, :820]
        words, x = [], 40.0
        for tok in tokens:
            words.append(OcrWord(tok, 0.9, (1, 1, 1), x, 104.0,
                                 18.0 * len(tok), 30.0))
            x += 18.0 * len(tok) + 12
        return ScanOcrResult(lines=[], gray=img, upright=True, words=words)

    def test_clean_value_still_read(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        got = ctcfill.read_field(self._result_for("Home World: Proxima-b"),
                                 "home_world")
        assert got is not None
        assert got[0] == "Proxima-b"
        assert got[1] <= ctcfill.CONF_CAP

    def test_hard_embargo_value_still_dropped(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        assert ctcfill.read_field(
            self._result_for("Home World: TRAPPIST-1e"),
            "home_world") is None

    def test_mode_default_still_suppressed(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        assert ctcfill.read_field(self._result_for("Home World: Luyten-b"),
                                  "home_world") is None

    def test_blank_value_still_abstains(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_FUSION", "1")
        assert ctcfill.read_field(self._result_for("Home World:"),
                                  "home_world") is None
