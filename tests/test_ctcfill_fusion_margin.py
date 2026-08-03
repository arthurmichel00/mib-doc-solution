"""MIB_CTCFILL_FUSION x MIB_CTCFILL_MARGIN — the two levers composed.

Fusion pools every view of a field into one score list and gates once;
the margin layer re-scores at a second rec resize and demands the same
winner under tighter per-field floors. Composed (integration patch
`integration-fusion-margin.patch`), the confirmation is taken on the
POOLED second resolution: every crop the fused verdict used is re-scored
at SECOND_RES_SCALE and those lists are fused the same way, so both
sides of the comparison answer the same question. On the single-strip
case — the common one, and the one the margin layer was calibrated on —
that is literally the margin layer's own single rec pass.

Skips entirely unless both levers AND the integration patch are present,
so the file is inert on any branch that holds only one of them.

Synthetic posterior grids keyed by (view, rec resize); no ONNX session.
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ctcfill

_READY = (hasattr(ctcfill, "fusion_enabled")
          and hasattr(ctcfill, "_margin_gates")
          and hasattr(ctcfill, "_second_resolution_confirms_fused"))
pytestmark = pytest.mark.skipif(
    not _READY, reason="needs lever-fusion + lever-ctcmargin + the "
                       "integration patch")

_CHARSET = ["<blank>", "L", ":", " ", "A", "B", "C"]
_LABEL = "L:"
_MENU = ("A", "B", "C")
_IX = {"A": 4, "B": 5, "C": 6}
FIELD = "home_world"
GOLD = "C"

PAGE_A = np.array([[100, 150]], np.uint8)
PAGE_B = np.array([[150, 100]], np.uint8)


def _grid(**mass: float) -> np.ndarray:
    p = np.full((4, len(_CHARSET)), 1e-6)
    p[0, 1], p[1, 2], p[2, 3] = 1.0, 1.0, 1.0
    for value, m in mass.items():
        p[3, _IX[value]] = m
    p[3, 0] = max(1e-6, 1.0 - sum(mass.values()))
    return np.log(p)


# Primary resolution: two views that disagree on argmax; C is the
# consistent runner-up, so only the fused list can reach it.
VIEW_A = _grid(A=0.60, C=0.35, B=0.005)
VIEW_B = _grid(B=0.60, C=0.35, A=0.005)

# Second resolution, four outcomes for the same fused winner C:
SECOND_CONFIRMS = _grid(C=0.90, A=0.02, B=0.02)      # margin 0.95
SECOND_NAMES_OTHER = _grid(A=0.90, C=0.02, B=0.02)   # winner A != C
SECOND_THIN = _grid(C=0.50, A=0.1233, B=0.005)       # margin ~0.35
SECOND_FAINT = _grid(C=1e-8, A=1e-11, B=1e-11)       # below slack floor


def by_variant(img):
    """One crop's two preprocessing variants disagree: the raw crop
    (min 100) favours A, its MINMAX-stretched twin (min 0) favours B."""
    return VIEW_A if img.min() == 100 else VIEW_B


def by_page(img):
    """Each page reads consistently within itself; the pages disagree.
    argmax survives the stretch, so both variants of a page agree."""
    return VIEW_A if int(img.argmax()) == 1 else VIEW_B


class _Post:
    """posteriorgram stub keyed by (crop, rec resize), recording the
    resize sequence — the layer's cost contract is a call pattern."""

    def __init__(self, second, dispatch=by_variant):
        self.second = second
        self.dispatch = dispatch
        self.scales: list[float] = []

    def __call__(self, img, scale=1.0):
        self.scales.append(scale)
        if scale != 1.0:
            return self.second
        return self.dispatch(np.asarray(img))


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(ctcfill, "_session", lambda: None)
    monkeypatch.setattr(ctcfill, "_CHARS", list(_CHARSET))
    monkeypatch.setattr(ctcfill, "_CHAR_TO_IX",
                        {c: i for i, c in enumerate(_CHARSET)})
    monkeypatch.setitem(ctcfill._MENUS, FIELD, _MENU)
    monkeypatch.setattr(ctcfill, "available", lambda: True)
    ctcfill._label_ixs.cache_clear()
    yield
    ctcfill._label_ixs.cache_clear()


def _flags(monkeypatch, fusion: bool, margin: bool):
    monkeypatch.setenv("MIB_CTCFILL", "1")
    for name, on in (("MIB_CTCFILL_FUSION", fusion),
                     ("MIB_CTCFILL_MARGIN", margin)):
        if on:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)


def _install(monkeypatch, second, pages=(PAGE_A,), dispatch=by_variant):
    post = _Post(second, dispatch)
    monkeypatch.setattr(ctcfill, "posteriorgram", post)
    monkeypatch.setattr(ctcfill, "locate_strips",
                        lambda result, field: [(result, _LABEL, 0.9)])
    monkeypatch.setattr(ctcfill, "page_grid_fit", lambda result: None)
    return post, {i: p for i, p in enumerate(pages)}


# ------------------------------------------- the composed verdict


class TestComposedAcceptance:
    def test_confirmed_fused_winner_fills(self, tiny, monkeypatch):
        _flags(monkeypatch, fusion=True, margin=True)
        post, scans = _install(monkeypatch, SECOND_CONFIRMS)
        assert ctcfill.fill(scans, [FIELD]) == \
            {FIELD: (GOLD, ctcfill.CONF_CAP)}
        # both primary variants, then ONE second-resolution pass on the
        # single pooled crop: the margin layer's own cost contract
        assert post.scales == [1.0, 1.0, ctcfill.SECOND_RES_SCALE]

    def test_second_resolution_naming_another_value_abstains(
            self, tiny, monkeypatch):
        _flags(monkeypatch, fusion=True, margin=True)
        post, scans = _install(monkeypatch, SECOND_NAMES_OTHER)
        assert ctcfill.fill(scans, [FIELD]) == {}
        assert ctcfill.SECOND_RES_SCALE in post.scales

    def test_second_resolution_inside_the_field_floor_abstains(
            self, tiny, monkeypatch):
        """A second-resolution margin that clears the shipped global
        MARGIN_RUNNER_UP but not the field's MARGIN2_FLOOR: the per-field
        floor is what removes it, exactly as on the unfused path."""
        _flags(monkeypatch, fusion=True, margin=True)
        post, scans = _install(monkeypatch, SECOND_THIN)
        scored = ctcfill.score_strip(PAGE_A, _LABEL, _MENU,
                                     ctcfill.SECOND_RES_SCALE)
        margin = ctcfill._score_margin(scored)[2]
        assert ctcfill.MARGIN_RUNNER_UP < margin < \
            ctcfill.MARGIN2_FLOOR[FIELD]
        assert ctcfill.fill(scans, [FIELD]) == {}

    def test_second_resolution_below_the_slack_abstains(self, tiny,
                                                        monkeypatch):
        _flags(monkeypatch, fusion=True, margin=True)
        post, scans = _install(monkeypatch, SECOND_FAINT)
        scored = ctcfill.score_strip(PAGE_A, _LABEL, _MENU,
                                     ctcfill.SECOND_RES_SCALE)
        _w, top, margin, _n = ctcfill._score_margin(scored)
        assert top < ctcfill.SCORE_FLOOR - ctcfill.SECOND_RES_FLOOR_SLACK
        assert margin > ctcfill.MARGIN2_FLOOR[FIELD]   # the floor, alone
        assert ctcfill.fill(scans, [FIELD]) == {}


# ------------------------------------- each lever alone is unchanged


class TestSingleLeverBehaviourPreserved:
    def test_fusion_alone_unchanged(self, tiny, monkeypatch):
        """Margin off: the fused fill stands and no second resolution is
        ever requested — byte-identical to the fusion-only lever."""
        _flags(monkeypatch, fusion=True, margin=False)
        post, scans = _install(monkeypatch, SECOND_NAMES_OTHER)
        assert ctcfill.fill(scans, [FIELD]) == \
            {FIELD: (GOLD, ctcfill.CONF_CAP)}
        assert post.scales == [1.0, 1.0]

    def test_margin_alone_unchanged(self, tiny, monkeypatch):
        """Fusion off: the two views disagree on argmax, so the shipped
        path abstains before any value is accepted — and an abstention
        never pays for a second-resolution pass."""
        _flags(monkeypatch, fusion=False, margin=True)
        post, scans = _install(monkeypatch, SECOND_CONFIRMS)
        assert ctcfill.fill(scans, [FIELD]) == {}
        assert post.scales == [1.0, 1.0]

    def test_both_off_unchanged(self, tiny, monkeypatch):
        _flags(monkeypatch, fusion=False, margin=False)
        post, scans = _install(monkeypatch, SECOND_CONFIRMS)
        assert ctcfill.fill(scans, [FIELD]) == {}
        assert post.scales == [1.0, 1.0]

    def test_the_confirmation_can_only_remove_a_fill(self, tiny,
                                                     monkeypatch):
        """Everything the composed pair fills is something fusion alone
        fills: the confirmation runs after the fused gate has passed."""
        for second in (SECOND_CONFIRMS, SECOND_NAMES_OTHER, SECOND_THIN,
                       SECOND_FAINT):
            _flags(monkeypatch, fusion=True, margin=False)
            _p, scans = _install(monkeypatch, second)
            alone = ctcfill.fill(scans, [FIELD])
            _flags(monkeypatch, fusion=True, margin=True)
            _p, scans = _install(monkeypatch, second)
            composed = ctcfill.fill(scans, [FIELD])
            assert not set(composed) - set(alone)
            for field, got in composed.items():
                assert alone[field] == got


# ------------------------------------------------- pooled confirmation


class TestPooledConfirmation:
    def test_confirms_once_per_pooled_crop(self, tiny, monkeypatch):
        """Two pages pooled: four primary passes, then one second-
        resolution pass per pooled crop — and only because the fused
        verdict passed, so the cost lands on fills, not on candidates."""
        _flags(monkeypatch, fusion=True, margin=True)
        post, scans = _install(monkeypatch, SECOND_CONFIRMS,
                               pages=(PAGE_A, PAGE_B), dispatch=by_page)
        assert ctcfill.fill(scans, [FIELD]) == \
            {FIELD: (GOLD, ctcfill.CONF_CAP)}
        assert post.scales.count(1.0) == 4
        assert post.scales.count(ctcfill.SECOND_RES_SCALE) == 2

    def test_no_second_resolution_when_the_fused_gate_already_failed(
            self, tiny, monkeypatch):
        """One page only: the fused list is that page's own ambiguous
        view, which fails the primary gate, so nothing is confirmed."""
        _flags(monkeypatch, fusion=True, margin=True)
        post, scans = _install(monkeypatch, SECOND_CONFIRMS,
                               dispatch=by_page)
        # PAGE_A alone -> VIEW_A on both variants -> margin 0.135 < 0.30
        assert ctcfill.fill(scans, [FIELD]) == {}
        assert ctcfill.SECOND_RES_SCALE not in post.scales


# --------------------------------------------------- grid path floors


class TestGridPathFloors:
    def test_grid_leg_uses_its_own_second_resolution_floors(
            self, tiny, monkeypatch):
        """A second-resolution margin between the grid floor and the
        word-box floor: accepted on the grid leg, rejected on the
        word-box leg. The two legs keep their separate calibration under
        fusion exactly as they do without it."""
        assert ctcfill.GRID_MARGIN2_FLOOR[FIELD] < 0.35 \
            < ctcfill.MARGIN2_FLOOR[FIELD]
        _flags(monkeypatch, fusion=True, margin=True)

        post = _Post(SECOND_THIN)   # variant dispatch: fused -> C
        monkeypatch.setattr(ctcfill, "posteriorgram", post)
        monkeypatch.setattr(ctcfill, "locate_strips",
                            lambda result, field: [])
        monkeypatch.setattr(ctcfill, "page_grid_fit", lambda result: "FIT")
        monkeypatch.setattr(
            ctcfill, "grid_strips",
            lambda result, field, fit=None: [(PAGE_A, _LABEL, 0.4)])
        # word-box leg located nothing -> falls through to the grid leg,
        # whose looser second-resolution floor admits this margin
        assert ctcfill.fill({0: PAGE_A}, [FIELD]) == {FIELD: (GOLD, 0.4)}
