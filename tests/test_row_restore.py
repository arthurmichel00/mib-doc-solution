"""Two-rail frame registration behind MIB_ROWRESTORE=1 (default OFF).

MIB_ROWRESTORE enables detection + repair + a closed-menu CTC re-read:
both frame rules are traced, a Theil-Sen baseline per rail separates
legitimate global skew from a defect's displacement, the two rails select
a repair MODEL (translation / shear / single-rail), only bands carrying a
row somebody needs are repaired, and the result must re-trace straight or
it is discarded. ctcfill.fill_restored then reads the registered page
under its own unchanged contract.

The geometry has one job and one anti-job. The job: a displaced strip is
put back so its cut glyphs line up. The anti-job: a page that is merely
SKEWED, or that draws no rule at all, must come through untouched. The
clean-page suite below is the property our own W1 detector failed (it
fired on 20 of 30 clean pages) and the reason this version is shippable.
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ctcfill, ocr, pipeline
from mib_pipeline import row_restore as R
from mib_pipeline.ocr import OcrWord, ScanOcrResult

# Synthetic page geometry. 600x400 at a nominal 100 DPI keeps the zone
# fractions (0.008..0.28 of width from each edge) around both drawn rules.
HEIGHT, WIDTH, DPI = 600, 400, 100.0
NEAR_X, FAR_X, RULE_W = 40, 350, 3
TOP_Y, BOTTOM_Y = 40, 550
BAND = (200, 260)
SHIFT = 25


def _page(skew: float = 0.0, near: bool = True, far: bool = True):
    """White page with a left and right frame rule and text-like bars.

    `skew` is the rules' px-per-line drift: legitimate global skew, which
    the Theil-Sen fit must absorb rather than report as damage.
    """
    gray = np.full((HEIGHT, WIDTH), 255, np.uint8)
    for y in range(HEIGHT):
        d = int(round(skew * y))
        if near:
            gray[y, NEAR_X + d:NEAR_X + d + RULE_W] = 0
        if far:
            gray[y, FAR_X + d:FAR_X + d + RULE_W] = 0
    # Top and bottom rules complete the rectangle, which is what the
    # transposed (vertical-band) pass reads. A 3-row horizontal line does
    # not survive the vertical opening, so it cannot disturb this pass.
    gray[TOP_Y:TOP_Y + RULE_W, NEAR_X:FAR_X + RULE_W] = 0
    gray[BOTTOM_Y:BOTTOM_Y + RULE_W, NEAR_X:FAR_X + RULE_W] = 0
    for top in range(80, HEIGHT - 80, 60):
        d = int(round(skew * top))
        for start in range(NEAR_X + 30 + d, FAR_X - 40, 24):
            gray[top:top + 10, start:start + 16] = 30
    return gray


def _shove(gray, band=BAND, near_dx=SHIFT, far_dx=None):
    """Displace a strip: translation (far_dx None) or a shear ramp."""
    import cv2

    out = gray.copy()
    y0, y1 = band
    width = gray.shape[1]
    columns = np.arange(width, dtype=np.float32)
    if far_dx is None:
        shift = np.full(width, float(near_dx), np.float32)
    else:
        t = (columns - NEAR_X) / float(FAR_X - NEAR_X)
        shift = (near_dx + t * (far_dx - near_dx)).astype(np.float32)
    map_x = np.ascontiguousarray((columns - shift)[None, :])
    map_y = np.zeros((1, width), np.float32)
    for y in range(y0, y1):
        out[y] = cv2.remap(gray[y][None, :], map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)[0]
    return out


def _ink_x(row):
    dark = np.flatnonzero(row < 128)
    return float(dark.mean()) if dark.size else float("nan")


def _models(bands):
    return [b.model for b in bands]


# --------------------------------------------------------------------------
# rail tracing


class TestRailTrace:
    def test_traces_both_rules(self):
        gray = _page()
        near = R._trace_rail(gray, DPI, far=False)
        far = R._trace_rail(gray, DPI, far=True)
        assert near is not None and far is not None
        assert abs(np.median(near.position[near.observed]) - NEAR_X) < 2.0
        assert abs(np.median(far.position[far.observed]) - FAR_X) < 2.0
        assert near.observed.mean() > 0.9 and far.observed.mean() > 0.9

    def test_follows_a_skewed_rule(self):
        skew = 0.02
        rail = R._trace_rail(_page(skew=skew), DPI, far=False)
        assert rail is not None
        expected = NEAR_X + skew * np.arange(HEIGHT)
        assert np.abs(rail.baseline - expected).max() < 3.0

    def test_follows_a_displaced_rule_into_the_band(self):
        rail = R._trace_rail(_shove(_page()), DPI, far=False)
        assert rail is not None
        inside = rail.position[BAND[0] + 5:BAND[1] - 5]
        assert abs(np.nanmedian(inside) - (NEAR_X + SHIFT)) < 2.0

    def test_no_rule_on_the_page(self):
        """Text alone must never be mistaken for a frame rule.

        This is the abstention that matters most: 209 of 255 train scan
        pages draw no vertical rule at all, and a tracker that follows a
        column of text instead is what fires garbage on them.
        """
        gray = np.full((HEIGHT, WIDTH), 255, np.uint8)
        for i, top in enumerate(range(20, HEIGHT - 20, 26)):
            start = 12 + (13 * i) % 60
            gray[top:top + 16, start:start + 18] = 30
        assert R._trace_rail(gray, DPI, far=False) is None
        assert R.detect(gray, DPI) == []

    def test_blank_page(self):
        blank = np.full((HEIGHT, WIDTH), 255, np.uint8)
        assert R._trace_rail(blank, DPI, far=False) is None

    def test_degenerate_size(self):
        assert R._trace_rail(np.full((HEIGHT, 8), 255, np.uint8),
                             DPI, far=False) is None
        assert R.detect(np.full((10, 10), 255, np.uint8), DPI) == []


class TestTheilSen:
    def test_recovers_a_line(self):
        lines = np.arange(200, dtype=np.float64)
        slope, intercept = R._theil_sen(lines, 3.0 + 0.05 * lines)
        assert abs(slope - 0.05) < 1e-6 and abs(intercept - 3.0) < 1e-6

    def test_survives_a_run_of_correlated_outliers(self):
        """A displaced band IS a run of correlated outliers."""
        lines = np.arange(400, dtype=np.float64)
        offsets = 10.0 + 0.02 * lines
        offsets[150:250] += 60.0
        slope, intercept = R._theil_sen(lines, offsets)
        assert abs(slope - 0.02) < 0.01 and abs(intercept - 10.0) < 3.0

    def test_flat_rule(self):
        lines = np.arange(100, dtype=np.float64)
        slope, intercept = R._theil_sen(lines, np.full(100, 42.0))
        assert abs(slope) < 1e-9 and abs(intercept - 42.0) < 1e-9


# --------------------------------------------------------------------------
# the frame-anchored gate on clean pages


class TestFrameAnchoredGateOnCleanPages:
    """The gate must not fire on an undamaged page, at any cost.

    This is the property that kept our own fragment realignment (lever
    W1) out of the build: its CONTENT-BASED detector fired on 20 of 30
    clean pages and cost 120-600 s/page, so Batch B rejected runtime
    realignment and LEVERS.md carried the frame-anchored variant as
    deferred, "blocked on a gate that is both harmless and useful".

    Anchoring on the frame is what changes the answer: a clean page's
    rules ARE the fitted baselines, so there is nothing displaced to
    find. Non-firing is structural, not a tuned threshold.
    """

    def _clean_pages(self):
        dense = _page()
        for top in range(30, HEIGHT - 30, 22):
            for start in range(NEAR_X + 20, FAR_X - 20, 19):
                dense[top:top + 11, start:start + 13] = 40
        faint = _page()
        faint[faint == 30] = 150
        return {
            "flat": _page(),
            "skew-0.005": _page(skew=0.005),
            "skew-0.02": _page(skew=0.02),
            "skew-0.05": _page(skew=0.05),
            "skew-negative": _page(skew=-0.03),
            "dense-text": dense,
            "faint-ink": faint,
            "near-rule-only": _page(far=False),
            "far-rule-only": _page(near=False),
        }

    def test_gate_does_not_fire_on_any_clean_page(self):
        fired = {name: _models(R.detect(gray, DPI))
                 for name, gray in self._clean_pages().items()}
        assert all(not v for v in fired.values()), fired

    def test_nothing_is_repaired_on_a_clean_page(self):
        fired = [name for name, gray in self._clean_pages().items()
                 if R.register(gray, DPI) is not None]
        assert fired == [], fired

    def test_clean_pages_stop_at_the_probe(self):
        """The cost half: an undamaged page pays one strided pass."""
        for name, gray in self._clean_pages().items():
            assert R.probe(gray, DPI) is False, name

    def test_a_damaged_page_still_fires(self):
        """Harmless is only half of it; the gate must also be useful."""
        assert R.probe(_shove(_page()), DPI) is True
        assert R.detect(_shove(_page()), DPI)
        assert R.register(_shove(_page()), DPI) is not None


# --------------------------------------------------------------------------
# model selection


class TestModelSelector:
    """Rail agreement selects a model; it does not veto.

    Arthur's objection to a strict agreement gate: "there may be times
    that only the left OR the right rail lined up gives the best result."
    Under shear dx_near != dx_far by construction, so rejecting on
    disagreement would throw away exactly the repairable bands — and on
    train renders EVERY surviving band is a shear (MIB-000953 p0: the
    right rule sits at x=2307 above and below the band and x=2151 inside
    it, with the left rule unmoved). A left-rail-only detector reads
    dx=0 there and is blind to the whole class.
    """

    def test_agreeing_rails_are_a_translation(self):
        bands = R.detect(_shove(_page(), near_dx=SHIFT), DPI)
        assert _models(bands) == ["translation"]
        assert abs(bands[0].near_dx - SHIFT) < 2.0
        assert abs(bands[0].far_dx - SHIFT) < 2.0
        assert abs(bands[0].spread) <= R._TRANSLATION_TOL_PX

    def test_disagreeing_rails_are_a_shear(self):
        bands = R.detect(_shove(_page(), near_dx=10, far_dx=25), DPI)
        assert _models(bands) == ["shear"]
        assert abs(bands[0].near_dx - 10) < 3.0
        assert abs(bands[0].far_dx - 25) < 3.0

    def test_shear_anchored_at_one_rail_is_still_a_shear(self):
        """The dominant real mode: one edge pulled in, the other fixed."""
        bands = R.detect(_shove(_page(), near_dx=0, far_dx=25), DPI)
        assert _models(bands) == ["shear"]
        assert abs(bands[0].near_dx) < 3.0
        assert abs(bands[0].far_dx - 25) < 4.0

    def test_single_rail_page_falls_back(self):
        bands = R.detect(_shove(_page(far=False), near_dx=40), DPI)
        assert _models(bands) == ["single"]

    def test_single_rail_demands_a_bigger_displacement(self):
        """Nothing corroborates a half-blind page, so ask for more."""
        modest = int(R.min_shift_px(DPI) * 1.2)
        assert R.detect(_shove(_page(far=False), near_dx=modest), DPI) == []
        assert R.detect(_shove(_page(), near_dx=modest), DPI)

    def test_ambiguous_spread_abstains(self):
        """Between "agree" and "shear" there is no confident model."""
        between = (R._TRANSLATION_TOL_PX + R._SHEAR_MIN_SPREAD_PX) // 2
        bands = R.detect(_shove(_page(), near_dx=20, far_dx=20 + between), DPI)
        assert bands == []

    def test_displacement_below_the_threshold_is_noise(self):
        small = int(R.min_shift_px(DPI)) - 2
        assert R.detect(_shove(_page(), near_dx=small), DPI) == []

    def test_threshold_is_env_tunable(self, monkeypatch):
        page = _shove(_page(), near_dx=SHIFT)
        assert R.detect(page, DPI)
        monkeypatch.setenv("MIB_ROWRESTORE_MIN_SHIFT", "400")
        assert R.detect(page, DPI) == []

    def test_threshold_scales_with_resolution(self):
        assert R.min_shift_px(288.0) == pytest.approx(12.0)
        assert R.min_shift_px(144.0) == pytest.approx(6.0)


# --------------------------------------------------------------------------
# repair


class TestRepair:
    def test_translation_is_put_back(self):
        clean = _page()
        reg = R.register(_shove(clean), DPI)
        assert reg is not None and _models(reg.repaired) == ["translation"]
        keep = slice(0, WIDTH - SHIFT - 5)
        for y in range(BAND[0] + 8, BAND[1] - 8):
            assert abs(_ink_x(reg.image[y, keep]) - _ink_x(clean[y, keep])) < 1.5

    def test_shear_is_put_back_by_interpolation(self):
        """The model a constant shift cannot express."""
        clean = _page()
        reg = R.register(_shove(clean, near_dx=10, far_dx=25), DPI)
        assert reg is not None and _models(reg.repaired) == ["shear"]
        repaired = reg.image
        for rail_x in (NEAR_X, FAR_X):
            for y in range(BAND[0] + 8, BAND[1] - 8):
                window = repaired[y, rail_x - 8:rail_x + 11]
                assert (window < 128).any(), (rail_x, y)

    def test_only_the_band_is_touched(self):
        damaged = _shove(_page())
        reg = R.register(damaged, DPI)
        assert reg is not None
        margin = max(8, int(0.25 * DPI))
        top = min(b.top for b in reg.repaired) - margin
        bottom = max(b.bottom for b in reg.repaired) + margin
        assert np.array_equal(reg.image[:max(0, top)], damaged[:max(0, top)])
        assert np.array_equal(reg.image[bottom:], damaged[bottom:])

    def test_skew_is_preserved_not_flattened(self):
        """Deskewing here would fight the ladder's own deskew pass."""
        skewed = _page(skew=0.02)
        assert R.register(skewed, DPI) is None
        rail = R._trace_rail(skewed, DPI, far=False)
        assert rail.baseline[-40] - rail.baseline[40] > 5.0

    def test_shift_at_reports_the_applied_correction(self):
        reg = R.register(_shove(_page()), DPI)
        band = reg.repaired[0]
        middle = (band.top + band.bottom) // 2
        assert abs(R.shift_at(reg.repaired, middle, 200.0) - SHIFT) < 2.0
        assert R.shift_at(reg.repaired, 10, 200.0) == 0.0

    def test_shift_at_ramps_across_a_shear(self):
        reg = R.register(_shove(_page(), near_dx=0, far_dx=25), DPI)
        band = reg.repaired[0]
        middle = (band.top + band.bottom) // 2
        at_near = R.shift_at(reg.repaired, middle, float(NEAR_X))
        at_far = R.shift_at(reg.repaired, middle, float(FAR_X))
        assert abs(at_near) < 3.0 and abs(at_far - 25) < 4.0


class TestSwapGuard:
    """A rule that is ABSENT is not a rule that MOVED.

    Where the border stops part-way across the page, the nearest-run pick
    latches onto the next rule along and reports the gap between them as
    a displacement. The tell is that the "destination" is occupied
    outside the band too: a real displacement vacates its origin and
    lands somewhere that was empty.

    Hand-verified on all six vertical corpus detections — the one real
    displacement puts 0 of 4 probes on ink, every mis-lock puts 1 or 2.
    """

    def _absent_rule_with_a_neighbour(self):
        """Near rule erased over the band; a second rule 50 px inboard."""
        gray = _page()
        gray[100:500, NEAR_X + 50:NEAR_X + 50 + RULE_W] = 0
        gray[BAND[0]:BAND[1], NEAR_X:NEAR_X + RULE_W] = 255
        return gray

    def test_a_swapped_rule_is_not_a_displacement(self):
        assert R.detect(self._absent_rule_with_a_neighbour(), DPI) == []
        assert R.register(self._absent_rule_with_a_neighbour(), DPI) is None

    def test_the_guard_itself_rejects_the_swap(self):
        rail = R._trace_rail(self._absent_rule_with_a_neighbour(), DPI,
                             far=False)
        assert rail is not None
        assert R._moved_not_swapped(rail, BAND[0], BAND[1], 50.0) is False

    def test_a_real_displacement_passes_the_guard(self):
        rail = R._trace_rail(_shove(_page()), DPI, far=False)
        assert R._moved_not_swapped(rail, BAND[0], BAND[1],
                                    float(SHIFT)) is True


class TestPostRepairValidation:
    """Acceptance is measured on the result, not predicted from the fit.

    It catches a repair that failed to straighten the frame — a botched
    remap, or a model that did not describe the damage. It cannot catch
    rails that were measured wrong in a mutually consistent way, since
    the repair is fitted to those same rails; that limit is real and is
    why the flatness and bracketing guards sit upstream of it.
    """

    def test_a_repair_that_does_not_straighten_is_discarded(self, monkeypatch):
        monkeypatch.setattr(R, "_remap", lambda strip, shift: strip.copy())
        reg = R.register(_shove(_page()), DPI)
        assert reg is None

    def test_the_validator_rejects_an_unstraightened_band(self):
        damaged = _shove(_page())
        band = R.detect(damaged, DPI)[0]
        assert R._straightened(damaged, DPI, band, 0) is False

    def test_the_validator_accepts_a_straightened_band(self):
        reg = R.register(_shove(_page()), DPI)
        assert reg is not None and reg.rejected == []


# --------------------------------------------------------------------------
# need-driven repair


class TestNeedDrivenRepair:
    """Arthur: "is it worth aligning every strip if there is no data in
    between?" No — the consumer reads four closed-menu fields, so a band
    holding none of their rows is detected, counted and left alone. It
    also kills the junk-token minting the free-form smoke test found,
    which lived exactly in content-free strips.
    """

    def test_band_outside_the_wanted_rows_is_never_repaired(self):
        damaged = _shove(_page())
        reg = R.register(damaged, DPI, wanted=[(500, 590)])
        assert reg is None

    def test_band_inside_the_wanted_rows_is_repaired(self):
        reg = R.register(_shove(_page()), DPI, wanted=[(210, 240)])
        assert reg is not None and len(reg.repaired) == 1

    def test_skipped_bands_are_counted_not_silently_dropped(self):
        damaged = _shove(_shove(_page(), band=(80, 130), near_dx=30))
        reg = R.register(damaged, DPI, wanted=[(210, 240)])
        assert reg is not None
        assert len(reg.repaired) == 1
        assert len(reg.skipped_no_need) >= 1
        assert all(b.bottom <= 210 or b.top >= 240
                   for b in reg.skipped_no_need)

    def test_none_means_repair_everything(self):
        damaged = _shove(_shove(_page(), band=(80, 130), near_dx=30))
        reg = R.register(damaged, DPI, wanted=None)
        assert reg is not None and len(reg.repaired) == 2


# --------------------------------------------------------------------------
# the transposed (vertical) axis


class TestVerticalAxis:
    """The border is a rectangle, so the same detector reads the other
    pair of rules on the transposed page and finds column bands shoved up
    or down. Arthur reports both defects in the corpus; a horizontal-only
    design sees half of them.
    """

    def test_column_band_shoved_vertically_is_found(self):
        damaged = _shove(np.ascontiguousarray(_page().T), band=(150, 210),
                         near_dx=25)
        damaged = np.ascontiguousarray(damaged.T)
        assert R.detect(damaged, DPI, axis=R.VERTICAL)
        assert R.detect(damaged, DPI, axis=R.HORIZONTAL) == []

    def test_clean_page_fires_on_neither_axis(self):
        for axis in (R.HORIZONTAL, R.VERTICAL):
            assert R.detect(_page(), DPI, axis=axis) == []
            assert R.probe(_page(), DPI, axis=axis) is False

    def test_repair_returns_the_page_the_right_way_round(self):
        damaged = _shove(np.ascontiguousarray(_page().T), band=(150, 210),
                         near_dx=25)
        damaged = np.ascontiguousarray(damaged.T)
        reg = R.register(damaged, DPI, axis=R.VERTICAL)
        assert reg is not None
        assert reg.image.shape == damaged.shape
        assert reg.axis == R.VERTICAL


# --------------------------------------------------------------------------
# rotation


class TestRotation:
    """The rails are only rails when the page is the right way up.

    The detector reads ScanOcrResult.gray, which the ladder has already
    orientation-corrected; where that correction was a guess
    (orientation_estimated) the channel abstains rather than measure the
    wrong axis off the wrong rules.

    FUTURE WORK (Arthur, not built): the rail fit's own strength at 0/90/
    180/270 is a ~4 ms orientation probe in its own right, and could
    rescue pages the anchor-scored probe gives up on.
    """

    def test_rotated_page_fires_nothing(self):
        turned = np.ascontiguousarray(np.rot90(_shove(_page())))
        assert R.detect(turned, DPI, axis=R.HORIZONTAL) == []
        assert R.register(turned, DPI) is None

    def test_channel_skips_a_non_upright_page(self):
        scan = ScanOcrResult(lines=[], gray=_shove(_page()), upright=False,
                             words=[])
        assert ctcfill.restored_view(scan, ["home_world"]) is None

    def test_channel_skips_a_guessed_orientation(self):
        scan = ScanOcrResult(lines=[], gray=_shove(_page()), upright=True,
                             orientation_estimated=True, words=[])
        assert ctcfill.restored_view(scan, ["home_world"]) is None


# --------------------------------------------------------------------------
# flag semantics


class ExplodingRegister:
    def __call__(self, *args, **kwargs):
        raise AssertionError("registration must not run with the flag off")


class RecordingEngine:
    def __init__(self):
        self.seen: list[tuple[int, int, bool]] = []

    def words(self, image, sparse=False):
        self.seen.append((image.shape[0], image.shape[1], sparse))
        return [OcrWord(text="RESTORED", conf=0.91, line_key=(1, 1, 1), x=1.0)]


class TestFlagSemantics:
    """MIB_ROWRESTORE must not buy the pass its measurement condemned."""

    def test_both_flags_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_ROWRESTORE", raising=False)
        monkeypatch.delenv("MIB_ROWRESTORE_FREEFORM", raising=False)
        assert R.enabled() is False and R.freeform_enabled() is False
        assert R.ROWRESTORE_DEFAULT is False and R.FREEFORM_DEFAULT is False

    def test_env_controls_each_flag_independently(self, monkeypatch):
        monkeypatch.setenv("MIB_ROWRESTORE", "1")
        monkeypatch.delenv("MIB_ROWRESTORE_FREEFORM", raising=False)
        assert R.enabled() is True and R.freeform_enabled() is False
        monkeypatch.setenv("MIB_ROWRESTORE", "0")
        monkeypatch.setenv("MIB_ROWRESTORE_FREEFORM", "1")
        assert R.enabled() is False and R.freeform_enabled() is True

    def test_rowrestore_alone_buys_no_free_form_pass(self, monkeypatch):
        """The arm must not pay ~204 ms/page for zero-upside code."""
        monkeypatch.setenv("MIB_ROWRESTORE", "1")
        monkeypatch.delenv("MIB_ROWRESTORE_FREEFORM", raising=False)
        monkeypatch.setattr(R, "register", ExplodingRegister())
        engine = RecordingEngine()
        result = ocr.ocr_scan_page(_shove(_page()), 0, engine)
        assert result.lines
        assert not any(w == WIDTH + 60 for _, w, _ in engine.seen)

    def test_freeform_flag_does_buy_the_pass(self, monkeypatch):
        monkeypatch.setenv("MIB_ROWRESTORE_FREEFORM", "1")
        engine = RecordingEngine()
        ocr.ocr_scan_page(_shove(_page()), 0, engine)
        assert any(w == WIDTH + 60 for _, w, _ in engine.seen)

    def test_clean_page_costs_no_ocr_pass(self, monkeypatch):
        monkeypatch.setenv("MIB_ROWRESTORE_FREEFORM", "1")
        engine = RecordingEngine()
        assert ocr.row_restore_lines(_page(), 0, engine, dpi=DPI) == []
        assert engine.seen == []

    def test_reconstructed_reads_are_confidence_capped(self, monkeypatch):
        monkeypatch.setenv("MIB_ROWRESTORE_FREEFORM", "1")
        lines = ocr.row_restore_lines(_shove(_page()), 0, RecordingEngine(),
                                      dpi=DPI)
        assert lines
        assert all(l.conf <= ocr._ROWRESTORE_CONF_CAP for l in lines)
        assert ocr._ROWRESTORE_CONF_CAP < 0.55   # fields._KNOWN_MIN_OCR_CONF


class TestPipelineWiring:
    def test_flag_gate(self, monkeypatch):
        monkeypatch.delenv("MIB_ROWRESTORE", raising=False)
        assert pipeline._rowrestore_enabled() is False
        monkeypatch.setenv("MIB_ROWRESTORE", "1")
        assert pipeline._rowrestore_enabled() is True

    def test_channel_is_independent_of_the_ctcfill_flag(self, monkeypatch):
        monkeypatch.setenv("MIB_ROWRESTORE", "1")
        monkeypatch.setenv("MIB_CTCFILL", "0")
        assert pipeline._rowrestore_enabled() is True
        assert pipeline._ctcfill_enabled() is False

    def test_allowance_is_bounded(self):
        assert pipeline.ROWRESTORE_ALLOWANCE <= pipeline.CTCFILL_ALLOWANCE


# --------------------------------------------------------------------------
# the shipped channel: registered page -> closed-menu CTC


_MODEL = ctcfill.available()
needs_model = pytest.mark.skipif(not _MODEL, reason="rec bundle not mounted")
_R_DPI = ocr.RENDER_DPI
_R_BAND = (300, 900)
_R_SHIFT = 30


def _rendered_page(text="Home World: Europa Station"):
    """A letter page with both frame rules and one real rendered form row."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 150), text, fontsize=11, fontname="hebo")
    pm = page.get_pixmap(dpi=int(_R_DPI), colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pm.samples, np.uint8).reshape(
        pm.height, pm.width).copy()
    doc.close()
    gray[:, 150:154] = 0
    gray[:, 2290:2294] = 0
    return gray


def _shove_render(gray, band, shift):
    out = gray.copy()
    y0, y1 = band
    strip = gray[y0:y1]
    out[y0:y1] = 255
    out[y0:y1, shift:] = strip[:, :strip.shape[1] - shift]
    return out


def _row_words(gray, damaged_shift=0):
    ink = gray < 128
    ink[:, 145:160] = False
    ink[:, 2285:2300] = False
    rows = np.flatnonzero(ink.any(axis=1))
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    cols = np.flatnonzero(ink[y0:y1].any(axis=0))
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 12) + 1)
    boxes = [(int(r[0]), int(r[-1]) + 1) for r in runs if len(r) > 4]
    out = []
    for (x0, x1), token in zip(boxes[:2], ("Home", "World:")):
        out.append(OcrWord(text=token, conf=0.9, line_key=(1, 1, 1),
                           x=float(x0 + damaged_shift), y=float(y0),
                           w=float(x1 - x0), h=float(y1 - y0)))
    return out


def _damaged_scan():
    clean = _rendered_page()
    damaged = _shove_render(clean, _R_BAND, _R_SHIFT)
    return ScanOcrResult(lines=[], gray=damaged, upright=True,
                         words=_row_words(clean, _R_SHIFT)), clean


class ExplodingRecognizer:
    def __call__(self, *args, **kwargs):
        raise AssertionError("clean pages must never reach the recognizer")


class TestGridPathScoping:
    """Registration is for grid-path pages only.

    Measured: where a label WAS read, the closed-menu scorer already
    returns the right value off a row split in two, at offsets from 30 to
    240 px, registered or not. So a page with located labels is not worth
    a gate, let alone a repair — the eligible population is exactly the
    pages where the lattice is the only locator left.
    """

    def test_located_labels_are_reported(self):
        scan, _ = _damaged_scan()
        assert ctcfill.located_labels(scan, ["home_world"]) == ["home_world"]
        assert ctcfill.located_labels(scan, ["visa_class"]) == []

    def test_low_confidence_label_does_not_count_as_located(self):
        scan, _ = _damaged_scan()
        faint = [OcrWord(text=w.text, conf=0.2, line_key=w.line_key, x=w.x,
                         y=w.y, w=w.w, h=w.h) for w in scan.words]
        scan = ScanOcrResult(lines=[], gray=scan.gray, upright=True,
                             words=faint)
        assert ctcfill.located_labels(scan, ["home_world"]) == []

    def test_word_box_page_is_not_registered(self):
        scan, _ = _damaged_scan()
        assert ctcfill.restored_view(scan, ["home_world"]) is None

    def test_grid_path_page_is_registered(self):
        scan, _ = _damaged_scan()
        blind = ScanOcrResult(lines=[], gray=scan.gray, upright=True,
                              words=[])
        assert ctcfill.restored_view(blind, ["home_world"]) is not None

    def test_wanted_rows_are_the_template_label_block(self):
        scan = ScanOcrResult(lines=[], gray=_rendered_page(), upright=True,
                             words=[])
        assert ctcfill.needed_spans(scan, ["home_world"]) == \
            [ctcfill._NEED_ENVELOPE]


class TestRestoredView:
    def test_none_when_nothing_is_displaced(self):
        scan = ScanOcrResult(lines=[], gray=_rendered_page(), upright=True,
                             words=[])
        assert ctcfill.restored_view(scan, ["home_world"]) is None

    def test_registered_page_carries_no_word_geometry(self):
        """It got here because no needed label was found on it."""
        scan, _ = _damaged_scan()
        blind = ScanOcrResult(lines=[], gray=scan.gray, upright=True,
                              words=[])
        out = ctcfill.restored_view(blind, ["home_world"])
        assert out is not None
        view, spans = out
        assert view.gray.shape == scan.gray.shape
        assert view.words == []
        assert spans and spans[0][0] >= _R_BAND[0] - 4


class TestFillRestored:
    def test_no_fields_needed_is_a_no_op(self):
        scan, _ = _damaged_scan()
        assert ctcfill.fill_restored({0: scan}, []) == {}

    def test_clean_page_never_reaches_the_recognizer(self, monkeypatch):
        monkeypatch.setattr(ctcfill, "posteriorgram", ExplodingRecognizer())
        scan = ScanOcrResult(lines=[], gray=_rendered_page(), upright=True,
                             words=[])
        assert ctcfill.fill_restored({0: scan}, list(ctcfill.FIELDS)) == {}

    def test_exhausted_budget_abstains(self):
        scan, _ = _damaged_scan()
        assert ctcfill.fill_restored({0: scan}, ["home_world"],
                                     budget_left=lambda: False) == {}

    def test_grid_row_span_is_render_scale(self):
        fit = (0.5, 100, 200, 31.2, 14)
        lo, hi = ctcfill._grid_row_span(fit, "home_world")
        top = 200 + int(round(3 * 31.2))
        assert lo == 2 * (top - 7) and hi == 2 * (top + 14 + 7)
        assert ctcfill._in_band(lo, hi, [(lo + 5, hi + 100)])
        assert not ctcfill._in_band(lo, hi, [(hi + 10, hi + 200)])
        assert ctcfill._in_band(lo, hi, None)      # vertical: no restriction

    def test_agreement_rule_matches_fill(self):
        assert ctcfill._agreed([]) is None
        assert ctcfill._agreed([("Europa Station", 0.9),
                                ("Kepler-186f", 0.9)]) is None
        value, conf = ctcfill._agreed([("Europa Station", 0.9),
                                       ("Europa Station", 0.7)])
        assert value == "Europa Station" and conf == ctcfill.CONF_CAP

    @needs_model
    def test_word_box_page_yields_nothing_here(self):
        """Its labels were read, so ordinary ctcfill owns it."""
        scan, _ = _damaged_scan()
        assert ctcfill.fill_restored({0: scan}, ["home_world"]) == {}

    @needs_model
    def test_contract_holds_conf_cap_and_field_scope(self):
        scan, _ = _damaged_scan()
        blind = ScanOcrResult(lines=[], gray=scan.gray, upright=True,
                              words=[])
        got = ctcfill.fill_restored({0: blind}, list(ctcfill.FIELDS))
        assert set(got) <= set(ctcfill.FIELDS)
        assert all(c <= ctcfill.CONF_CAP for _, c in got.values())
        for excluded in ("fee_status", "sponsor_id", "applicant_name"):
            assert excluded not in got


# --------------------------------------------------------------------------
# where the channel earns its cost: the intake-grid locator


_GRID_VALUES = {
    "Case ID:": "MIB-000124", "Applicant:": "Zarel Vantox",
    "Species Code:": "ORION_GRAYS", "Home World:": "Europa Station",
    "Visa Class:": "XW-2", "Sponsor ID:": "SPN-1345",
    "Arrival Date:": "2026-06-03", "Declared Purpose:": "trade delegation",
}
_GRID_PITCH_PT = 31.22 / 2.0
_GRID_FONT_PT = 6.0


def _intake_page():
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for k, label in enumerate(ctcfill._GRID_LABELS):
        y = 40.0 + k * _GRID_PITCH_PT
        page.insert_text((30.0, y), label, fontsize=_GRID_FONT_PT,
                         fontname="hebo")
        page.insert_text((92.0, y), _GRID_VALUES[label],
                         fontsize=_GRID_FONT_PT, fontname="hebo")
    pm = page.get_pixmap(dpi=int(_R_DPI), colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pm.samples, np.uint8).reshape(
        pm.height, pm.width).copy()
    doc.close()
    gray[:, 60:64] = 0
    gray[:, 2290:2294] = 0
    return gray


def _native(gray):
    import cv2

    return cv2.resize(gray, None, fx=0.5, fy=0.5,
                      interpolation=cv2.INTER_AREA)


class TestGridLatticeRecovery:
    """The measured reason this channel exists.

    The intake grid locator reads rows whose label no recognizer could
    see. A displaced band moves label ink off the lattice, and a joint
    eight-label fit that cannot explain eight rows with one
    (x0, y0, pitch) scores below _GRID_MIN_NCC and abstains: the locator
    is not degraded, it is dead. Registration puts the rows back.
    """

    def _damage(self):
        clean = _intake_page()
        ink = np.flatnonzero((clean < 128)[:, 200:2280].any(axis=1))
        return clean, (int(ink.min()) + 120, int(ink.max()) - 120)

    def test_clean_page_fits_the_lattice(self):
        clean, _band = self._damage()
        fit = ctcfill._grid_fit(_native(clean))
        assert fit is not None and fit[0] >= ctcfill._GRID_MIN_NCC

    @pytest.mark.parametrize("shift", (30, 80, 150))
    def test_displacement_kills_the_locator(self, shift):
        clean, band = self._damage()
        damaged = _shove_render(clean, band, shift)
        assert ctcfill._grid_fit(_native(damaged)) is None

    @pytest.mark.parametrize("shift", (30, 80, 150))
    def test_registration_restores_the_locator(self, shift):
        clean, band = self._damage()
        reference = ctcfill._grid_fit(_native(clean))
        damaged = _shove_render(clean, band, shift)
        scan = ScanOcrResult(lines=[], gray=damaged, upright=True, words=[])
        out = ctcfill.restored_view(scan, list(ctcfill.FIELDS))
        assert out is not None, "the gate must fire on this damage"
        fit = ctcfill.page_grid_fit(out[0])
        assert fit is not None
        assert abs(fit[0] - reference[0]) < 0.02
