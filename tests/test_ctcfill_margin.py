"""Second-resolution margin gates behind MIB_CTCFILL_MARGIN=1.

The layer re-scores every strip the shipped CTC gate accepts at a second
rec-model input resize and keeps the fill only when both resolutions name
the same winner and the second one clears its own score floor and the
field's runner-up margin floor. It is purely restrictive: with the flag
unset nothing below changes, which is the A/B contract.

Most tests here drive the CTC forward with SYNTHETIC POSTERIOR GRIDS over
a synthetic charset, so the margins asserted are real numbers on the
lever's own nats-per-character scale with no model bundle needed. The
model-dependent class at the end pins the same scale against the real
recognizer.
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ctcfill, vocab

_MODEL = ctcfill.available()
needs_model = pytest.mark.skipif(not _MODEL, reason="rec bundle not mounted")


# ------------------------------------------------- synthetic posteriors

def _charset(field: str) -> list[str]:
    """[blank] + every character the field's candidates can contain."""
    label = " ".join(ctcfill._LABELS[field][0]) + ":"
    seen = set(label)
    for value in ctcfill._MENUS[field]:
        seen |= set(value)
    return ["<blank>"] + sorted(seen)


def _posteriors(text: str, chars: list[str], peak: float = 0.999,
                per_char: int = 3,
                swap: tuple[str, float] | None = None) -> np.ndarray:
    """(T, C) log-posteriors that emit `text` as a CTC recognizer would.

    Each character gets `per_char` frames at probability `peak` with a
    blank frame between characters; the remaining mass is spread evenly.
    `swap` = (character, probability) puts a competing character on the
    LAST frame group instead of some of the peak mass, which is how the
    near-tie cases below dial an exact runner-up margin: with one
    distinguishing character the total log-prob gap is
    per_char * log(peak / swap_prob) and the reported margin is that gap
    divided by the candidate's length.
    """
    ix = {c: i for i, c in enumerate(chars)}
    C = len(chars)
    frames = []

    def frame(dist: dict[str, float]) -> np.ndarray:
        # Named classes get their weight, everything else an even share
        # of what is left; the row is normalized, which leaves the RATIO
        # between named classes — and therefore the margin — exact.
        row = np.full(C, (1.0 - peak) / C)
        for ch, p in dist.items():
            row[ix[ch]] = p
        return row / row.sum()

    for i, ch in enumerate(text):
        frames.append(frame({"<blank>": peak}))
        dist = {ch: peak}
        if swap is not None and i == len(text) - 1:
            dist = {ch: peak, swap[0]: swap[1]}
        for _ in range(per_char):
            frames.append(frame(dist))
    frames.append(frame({"<blank>": peak}))
    return np.log(np.asarray(frames))


def _install(monkeypatch, field: str, grids: dict[float, np.ndarray]):
    """Point the scorer at fixed posterior grids, keyed by rec resize."""
    chars = _charset(field)
    monkeypatch.setattr(ctcfill, "_session", lambda: None)
    monkeypatch.setattr(ctcfill, "_CHARS", chars)
    monkeypatch.setattr(ctcfill, "_CHAR_TO_IX",
                        {c: i for i, c in enumerate(chars)})
    ctcfill._label_ixs.cache_clear()
    calls = []

    def fake(gray, scale=1.0):
        calls.append(scale)
        return grids[scale]

    monkeypatch.setattr(ctcfill, "posteriorgram", fake)
    return calls


def _strip() -> np.ndarray:
    return np.full((24, 300), 255, np.uint8)


def _label(field: str) -> str:
    return " ".join(ctcfill._LABELS[field][0]) + ":"


@pytest.fixture(autouse=True)
def _clear_label_cache():
    yield
    ctcfill._label_ixs.cache_clear()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MIB_CTCFILL", "1")
    monkeypatch.setenv("MIB_CTCFILL_MARGIN", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv("MIB_CTCFILL", "1")
    monkeypatch.delenv("MIB_CTCFILL_MARGIN", raising=False)


# ------------------------------------------------------------ flag gate


class TestFlagGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL", "1")
        monkeypatch.delenv("MIB_CTCFILL_MARGIN", raising=False)
        assert ctcfill._margin_gates() is False

    def test_needs_both_flags(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL_MARGIN", "1")
        monkeypatch.delenv("MIB_CTCFILL", raising=False)
        assert ctcfill._margin_gates() is False
        monkeypatch.setenv("MIB_CTCFILL", "1")
        assert ctcfill._margin_gates() is True

    def test_zero_is_off(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL", "1")
        monkeypatch.setenv("MIB_CTCFILL_MARGIN", "0")
        assert ctcfill._margin_gates() is False


# ------------------------------------------------- floor derivation

class TestFloorDerivation:
    """The floors equalize TOTAL evidence (margin x candidate length)
    across menus that differ 1.7x in length, at the level the global
    floor already demands of the longest field."""

    def _mean_len(self, field: str) -> float:
        label = _label(field)
        lens = [len(f"{label} {v}") for v in ctcfill._MENUS[field]]
        return sum(lens) / len(lens)

    def test_floors_match_the_documented_formula(self):
        lengths = {f: self._mean_len(f) for f in ctcfill.FIELDS}
        longest = max(lengths.values())
        for field, floor in ctcfill.MARGIN2_FLOOR.items():
            want = ctcfill.MARGIN_RUNNER_UP * longest / lengths[field]
            assert floor == pytest.approx(want, abs=0.005), field

    def test_longest_field_keeps_the_shipped_margin(self):
        """Equalization is anchored on the longest menu, so that field's
        floor is the global one — the layer adds no margin there."""
        lengths = {f: self._mean_len(f) for f in ctcfill.FIELDS}
        longest = max(lengths, key=lengths.get)
        assert longest == "declared_purpose"
        assert ctcfill.MARGIN2_FLOOR[longest] == ctcfill.MARGIN_RUNNER_UP

    def test_grid_floors_hold_the_paths_own_ratio(self):
        ratio = ctcfill.GRID_MARGIN_RUNNER_UP / ctcfill.MARGIN_RUNNER_UP
        for field, floor in ctcfill.GRID_MARGIN2_FLOOR.items():
            assert floor == pytest.approx(
                ctcfill.MARGIN2_FLOOR[field] * ratio, abs=0.005), field

    def test_every_floor_can_only_restrict(self):
        """No field's floor may fall below the shipped global margin —
        that is what makes the flag a subset of today's behaviour."""
        for field in ctcfill.FIELDS:
            assert ctcfill.MARGIN2_FLOOR[field] >= ctcfill.MARGIN_RUNNER_UP
            assert (ctcfill.GRID_MARGIN2_FLOOR[field]
                    >= ctcfill.GRID_MARGIN_RUNNER_UP)

    def test_shortest_menu_is_gated_hardest(self):
        """visa_class: shortest candidates AND the only menu whose
        closest pair is one glyph apart (XW-1/XW-2)."""
        assert ctcfill.MARGIN2_FLOOR["visa_class"] == max(
            ctcfill.MARGIN2_FLOOR.values())
        assert "XW-1" in vocab.VISA_CLASSES and "XW-2" in vocab.VISA_CLASSES

    def test_all_four_fields_covered(self):
        assert set(ctcfill.MARGIN2_FLOOR) == set(ctcfill.FIELDS)
        assert set(ctcfill.GRID_MARGIN2_FLOOR) == set(ctcfill.FIELDS)


# --------------------------------------------- margin as the confidence

class TestScoreMargin:
    def test_returns_winner_with_its_margin(self):
        scored = [(-1.0, "Proxima-b"), (-1.7, "Luyten-b"), (-6.0, "<none>")]
        winner, top, margin, m_none = ctcfill._score_margin(scored)
        assert winner == "Proxima-b"
        assert top == pytest.approx(-1.0)
        assert margin == pytest.approx(0.7)
        assert m_none == pytest.approx(5.0)

    def test_none_is_not_the_runner_up(self):
        """<none> sits on its own gate; it must not soak up the margin."""
        scored = [(-1.0, "Proxima-b"), (-1.05, "<none>"), (-3.0, "Luyten-b")]
        assert ctcfill._score_margin(scored)[2] == pytest.approx(2.0)

    def test_empty_scored_has_no_winner(self):
        assert ctcfill._score_margin([])[0] is None

    def test_gated_winner_still_returns_the_bare_winner(self):
        scored = [(-1.0, "Proxima-b"), (-2.0, "Luyten-b"), (-6.0, "<none>")]
        assert ctcfill._gated_winner(scored) == "Proxima-b"


# ------------------------------------------------ synthetic-grid gating

class TestSecondResolutionGate:
    FIELD = "visa_class"
    GOLD = "XW-1"

    def _grids(self, swap_primary=None, swap_second=None,
               second_text=None):
        chars = _charset(self.FIELD)
        text = f"{_label(self.FIELD)} {self.GOLD}"
        return {
            1.0: _posteriors(text, chars, swap=swap_primary),
            ctcfill.SECOND_RES_SCALE: _posteriors(
                second_text or text, chars, swap=swap_second),
        }

    def test_clear_winner_passes_both_resolutions(self, flag_on,
                                                  monkeypatch):
        calls = _install(monkeypatch, self.FIELD, self._grids())
        got = ctcfill._accept_strips(
            [(_strip(), _label(self.FIELD), 0.9)], self.FIELD)
        assert got == (self.GOLD, 0.5)
        # both variants at the primary resolution, ONE rec pass at the
        # second — the cost contract for the layer
        assert calls == [1.0, 1.0, ctcfill.SECOND_RES_SCALE]

    def test_near_tie_inside_the_field_floor_is_rejected(self, flag_on,
                                                         monkeypatch):
        """A gap that clears the shipped global margin but not the
        field's floor: 6.4 nats of total evidence over a 16-char
        candidate = 0.40 nats/char, above MARGIN_RUNNER_UP (0.30) and
        below MARGIN2_FLOOR['visa_class'] (0.52)."""
        swap = ("2", 0.999 * np.exp(-6.4 / 3))
        grids = self._grids(swap_primary=swap, swap_second=swap)
        _install(monkeypatch, self.FIELD, grids)
        strips = [(_strip(), _label(self.FIELD), 0.9)]

        scored = ctcfill.score_strip(_strip(), _label(self.FIELD),
                                     ctcfill._MENUS[self.FIELD])
        margin = ctcfill._score_margin(scored)[2]
        assert ctcfill.MARGIN_RUNNER_UP < margin < \
            ctcfill.MARGIN2_FLOOR[self.FIELD]
        assert margin == pytest.approx(0.40, abs=0.02)

        assert ctcfill._accept_strips(strips, self.FIELD) is None

    def test_that_same_near_tie_is_accepted_with_the_flag_off(
            self, flag_off, monkeypatch):
        """The regression pair for the case above: today's lever takes
        it, so the flag is what removes it — nothing else changed."""
        swap = ("2", 0.999 * np.exp(-6.4 / 3))
        calls = _install(monkeypatch, self.FIELD,
                         self._grids(swap_primary=swap, swap_second=swap))
        got = ctcfill._accept_strips(
            [(_strip(), _label(self.FIELD), 0.9)], self.FIELD)
        assert got == (self.GOLD, 0.5)
        assert calls == [1.0, 1.0]          # no second-resolution pass

    def test_resolutions_disagreeing_on_the_winner_reject(self, flag_on,
                                                          monkeypatch):
        second = f"{_label(self.FIELD)} XW-2"
        _install(monkeypatch, self.FIELD, self._grids(second_text=second))
        assert ctcfill._accept_strips(
            [(_strip(), _label(self.FIELD), 0.9)], self.FIELD) is None

    def test_second_resolution_near_tie_rejects_a_clean_primary(
            self, flag_on, monkeypatch):
        """The measured wrong-fire mode: the primary resolution is
        confident, the smaller one collapses to a near-tie."""
        _install(monkeypatch, self.FIELD,
                 self._grids(swap_second=("2", 0.4)))
        assert ctcfill._accept_strips(
            [(_strip(), _label(self.FIELD), 0.9)], self.FIELD) is None

    def test_second_resolution_score_floor_gets_its_slack(self, flag_on,
                                                          monkeypatch):
        """A second view whose absolute score sits between the primary
        floor and the slackened one still confirms, as long as its margin
        holds: the measured resize shift must not masquerade as missing
        evidence. Uses home_world, whose runner-up is many glyphs away,
        so the absolute score can be pushed into the slack window while
        the margin stays clear of the floor."""
        field, gold = "home_world", "Proxima-b"
        chars = _charset(field)
        text = f"{_label(field)} {gold}"
        grids = {1.0: _posteriors(text, chars),
                 ctcfill.SECOND_RES_SCALE: _posteriors(text, chars,
                                                       peak=0.35)}
        _install(monkeypatch, field, grids)
        scored = ctcfill.score_strip(_strip(), _label(field),
                                     ctcfill._MENUS[field],
                                     ctcfill.SECOND_RES_SCALE)
        _w, top, margin, _n = ctcfill._score_margin(scored)
        assert (ctcfill.SCORE_FLOOR - ctcfill.SECOND_RES_FLOOR_SLACK
                <= top < ctcfill.SCORE_FLOOR)
        assert margin > ctcfill.MARGIN2_FLOOR[field]
        assert ctcfill._accept_strips(
            [(_strip(), _label(field), 0.9)], field) == (gold, 0.5)

    def test_second_resolution_below_the_slack_still_rejects(
            self, flag_on, monkeypatch):
        """The slackened floor is a floor, not an exemption."""
        field, gold = "home_world", "Proxima-b"
        chars = _charset(field)
        text = f"{_label(field)} {gold}"
        grids = {1.0: _posteriors(text, chars),
                 ctcfill.SECOND_RES_SCALE: _posteriors(text, chars,
                                                       peak=0.28)}
        _install(monkeypatch, field, grids)
        scored = ctcfill.score_strip(_strip(), _label(field),
                                     ctcfill._MENUS[field],
                                     ctcfill.SECOND_RES_SCALE)
        _w, top, margin, _n = ctcfill._score_margin(scored)
        assert top < ctcfill.SCORE_FLOOR - ctcfill.SECOND_RES_FLOOR_SLACK
        assert margin > ctcfill.MARGIN2_FLOOR[field]   # margin is fine
        assert ctcfill._accept_strips(
            [(_strip(), _label(field), 0.9)], field) is None

    def test_grid_path_uses_its_own_floors(self, flag_on, monkeypatch):
        """A margin that clears the grid path's floor but not the
        word-box path's is accepted only when the grid floors are passed
        down — the two paths keep their separate calibration."""
        gap = 0.44 * len(f"{_label(self.FIELD)} {self.GOLD}")
        swap = ("2", 0.999 * np.exp(-gap / 3))
        grids = self._grids(swap_primary=swap, swap_second=swap)
        _install(monkeypatch, self.FIELD, grids)
        strips = [(_strip(), _label(self.FIELD), 0.9)]
        assert ctcfill.GRID_MARGIN2_FLOOR[self.FIELD] < 0.44 \
            < ctcfill.MARGIN2_FLOOR[self.FIELD]
        assert ctcfill._accept_strips(strips, self.FIELD) is None
        assert ctcfill._accept_strips(
            strips, self.FIELD, floor=ctcfill.GRID_SCORE_FLOOR,
            m_ru=ctcfill.GRID_MARGIN_RUNNER_UP,
            m2=ctcfill.GRID_MARGIN2_FLOOR) == (self.GOLD, 0.5)


# ------------------------------------------------------ safety guards

class TestGuardsSurvive:
    """Every guard the shipped lever enforces still fires first, so the
    new pass never runs on a value that was going to be dropped."""

    def _grids(self, field, value):
        chars = _charset(field)
        text = f"{_label(field)} {value}"
        return {1.0: _posteriors(text, chars),
                ctcfill.SECOND_RES_SCALE: _posteriors(text, chars)}

    def test_hard_embargo_world_still_dropped_without_a_second_pass(
            self, flag_on, monkeypatch):
        field = "home_world"
        calls = _install(monkeypatch, field,
                         self._grids(field, "TRAPPIST-1e"))
        assert ctcfill._accept_strips(
            [(_strip(), _label(field), 0.9)], field) is None
        assert ctcfill.SECOND_RES_SCALE not in calls

    def test_mode_default_still_dropped_without_a_second_pass(
            self, flag_on, monkeypatch):
        field = "home_world"
        calls = _install(monkeypatch, field, self._grids(field, "Luyten-b"))
        assert ctcfill._accept_strips(
            [(_strip(), _label(field), 0.9)], field) is None
        assert ctcfill.SECOND_RES_SCALE not in calls

    def test_confidence_stays_capped(self, flag_on, monkeypatch):
        field = "home_world"
        _install(monkeypatch, field, self._grids(field, "Proxima-b"))
        got = ctcfill._accept_strips(
            [(_strip(), _label(field), 0.99)], field)
        assert got is not None and got[1] <= ctcfill.CONF_CAP


# ------------------------------------------------ flag-off regressions

class TestFlagOffIsUnchanged:
    def test_preprocess_default_scale_is_the_shipped_arithmetic(self):
        rng = np.random.default_rng(5)
        for shape in ((48, 684), (24, 300), (13, 40), (60, 4000)):
            gray = rng.integers(0, 255, shape, dtype=np.uint8)
            h, w = shape
            tw = min(ctcfill._REC_MAX_W,
                     max(16, int(round(ctcfill._REC_H * w / h))))
            got = ctcfill._preprocess(gray)
            assert got.shape == (1, 3, ctcfill._REC_H, tw)
            assert np.array_equal(got, ctcfill._preprocess(gray, 1.0))

    def test_no_second_pass_and_same_verdicts(self, flag_off, monkeypatch):
        """Sweep the near-tie margin across the field floor: with the
        flag off every case resolves exactly as the shipped gate does and
        the second resolution is never touched."""
        field, gold = "visa_class", "XW-1"
        chars = _charset(field)
        text = f"{_label(field)} {gold}"
        for gap in (2.0, 8.0, 14.0, 30.0):
            swap = ("2", 0.999 * np.exp(-gap / 3))
            grids = {1.0: _posteriors(text, chars, swap=swap),
                     ctcfill.SECOND_RES_SCALE: _posteriors(
                         f"{_label(field)} XW-2", chars)}
            calls = _install(monkeypatch, field, grids)
            got = ctcfill._accept_strips(
                [(_strip(), _label(field), 0.9)], field)
            scored = ctcfill.score_strip(_strip(), _label(field),
                                         ctcfill._MENUS[field])
            shipped = ctcfill._gated_winner(scored)
            assert (got[0] if got else None) == shipped, gap
            assert ctcfill.SECOND_RES_SCALE not in calls, gap

    def test_read_field_untouched_with_flag_off(self, flag_off,
                                                monkeypatch):
        field = "home_world"
        chars = _charset(field)
        text = f"{_label(field)} Proxima-b"
        calls = _install(monkeypatch, field,
                         {1.0: _posteriors(text, chars)})
        monkeypatch.setattr(
            ctcfill, "locate_strips",
            lambda result, fld: [(_strip(), _label(field), 0.9)])
        assert ctcfill.read_field(None, field) == ("Proxima-b", 0.5)
        assert calls == [1.0, 1.0]


# --------------------------------------------- real-recognizer contract

def _render(text: str, width: int = 900, height: int = 44) -> np.ndarray:
    import cv2

    img = np.full((height, width), 255, np.uint8)
    cv2.putText(img, text, (8, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)
    return img


@needs_model
class TestRealScoreScale:
    """Pin the constants against the recognizer's actual output scale, so
    a model or preprocessing change that moves the scale fails here
    rather than silently re-tuning the gate."""

    def test_second_resolution_is_a_genuinely_different_view(self):
        strip = _render("Home World: Proxima-b")
        a = ctcfill.posteriorgram(strip)
        b = ctcfill.posteriorgram(strip, ctcfill.SECOND_RES_SCALE)
        assert b.shape[0] < a.shape[0]          # fewer CTC frames

    def test_upscale_would_be_a_no_op_on_a_saturated_strip(self):
        """Why the second resolution is a DOWN-scale: a word-box strip
        already saturates the dynamic-width cap."""
        strip = _render("Home World: Proxima-b", width=900, height=44)
        assert ctcfill._preprocess(strip).shape[-1] == ctcfill._REC_MAX_W
        assert ctcfill._preprocess(strip, 1.25).shape[-1] == \
            ctcfill._REC_MAX_W

    def test_clean_read_clears_its_field_floor_at_both_resolutions(self):
        for field, value, text in (
                ("home_world", "Proxima-b", "Home World: Proxima-b"),
                ("declared_purpose", "research",
                 "Declared Purpose: research"),
                ("species_code", "ORION_GRAYS",
                 "Species Code: ORION_GRAYS")):
            strip = _render(text)
            for scale in (1.0, ctcfill.SECOND_RES_SCALE):
                scored = ctcfill.score_strip(strip, _label(field),
                                             ctcfill._MENUS[field], scale)
                winner, top, margin, _ = ctcfill._score_margin(scored)
                assert winner == value, (field, scale)
                assert margin > ctcfill.MARGIN2_FLOOR[field], (field, scale)
                assert top >= (ctcfill.SCORE_FLOOR
                               - ctcfill.SECOND_RES_FLOOR_SLACK)

    def test_one_glyph_menu_is_the_tight_one(self):
        """XW-1 vs XW-2 is the whole reason visa_class carries the
        highest floor: a clean read's margin there is ~4x smaller than a
        clean read on a menu whose members differ by many glyphs."""
        visa = ctcfill._score_margin(ctcfill.score_strip(
            _render("Visa Class: XW-1"), "Visa Class:",
            ctcfill._MENUS["visa_class"]))[2]
        world = ctcfill._score_margin(ctcfill.score_strip(
            _render("Home World: Proxima-b"), "Home World:",
            ctcfill._MENUS["home_world"]))[2]
        assert visa < world
        assert visa > ctcfill.MARGIN2_FLOOR["visa_class"]

    def test_clean_read_still_fills_with_the_flag_on(self, monkeypatch):
        """End-to-end through the real recognizer: the layer must not
        reject the reads the lever exists to make. Run with the flag both
        ways so a gate that rejected everything would show up here."""
        monkeypatch.setenv("MIB_CTCFILL", "1")
        strips = [(_render("Home World: Proxima-b"), "Home World:", 0.9)]
        monkeypatch.delenv("MIB_CTCFILL_MARGIN", raising=False)
        off = ctcfill._accept_strips(strips, "home_world")
        monkeypatch.setenv("MIB_CTCFILL_MARGIN", "1")
        on = ctcfill._accept_strips(strips, "home_world")
        assert off == ("Proxima-b", ctcfill.CONF_CAP)
        assert on == off

    def test_blank_strip_is_rejected_either_way(self, monkeypatch):
        monkeypatch.setenv("MIB_CTCFILL", "1")
        monkeypatch.setenv("MIB_CTCFILL_MARGIN", "1")
        blank = [(np.full((40, 700), 255, np.uint8), "Home World:", 0.9)]
        assert ctcfill._accept_strips(blank, "home_world") is None

    def test_scale_shift_fits_inside_the_slack(self):
        """The calibrated reason SECOND_RES_FLOOR_SLACK exists."""
        for text, field in (("Home World: Proxima-b", "home_world"),
                            ("Visa Class: XW-1", "visa_class")):
            strip = _render(text)
            a = ctcfill._score_margin(ctcfill.score_strip(
                strip, _label(field), ctcfill._MENUS[field]))[1]
            b = ctcfill._score_margin(ctcfill.score_strip(
                strip, _label(field), ctcfill._MENUS[field],
                ctcfill.SECOND_RES_SCALE))[1]
            assert a - b < ctcfill.SECOND_RES_FLOOR_SLACK, text
