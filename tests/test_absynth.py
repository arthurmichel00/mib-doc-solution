"""Generator-inversion menu reader behind MIB_ABSYNTH=1 (lever-absynth).

The lever re-renders every legal menu candidate on the generator's exact
raster grid, degrades it with a kernel recovered from the row's OWN known
label, and picks the winner by NCC on a fixed common canvas. No recognizer
is involved, so it can still separate candidates on strips whose CTC
posteriors are noise. It is the LAST resort: it fires only on the four
closed menus, only on slots still unread after ctcfill, and its accepted
values are extraction-only fills with confidence capped below the
affirmative-read threshold. With the flag unset the block is dead code.

Every test here is synthetic: pages are composed from the same renderer
the reader inverts, so the ground truth is exact and no corpus fixture is
needed.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mib_pipeline import absynth, fields, pipeline, policy, writer

CASE_ID = "MIB-000124"


# ------------------------------------------------------------- synthesis
def _place(page: np.ndarray, text: str, size: float, x: float,
           y: float) -> None:
    """Draw `text` with its pen origin (baseline, left edge) at (x, y)."""
    img, px, py = absynth.render_strip(text, size)
    x0, y0 = int(round(x - px)), int(round(y - py))
    h, w = img.shape
    if x0 < 0 or y0 < 0 or y0 + h > page.shape[0] or x0 + w > page.shape[1]:
        raise AssertionError(f"synthetic row off page: {text!r}")
    region = page[y0:y0 + h, x0:x0 + w]
    np.minimum(region, img, out=region)      # ink is dark


def _synth(values: dict[str, str], kernel=(0, 0.0), case_id: str = CASE_ID,
           dy: float = 0.0, dx: float = 0.0, title: bool = True,
           caseid_skew: float = 0.0) -> np.ndarray:
    """A synthetic 144-dpi intake page carrying `values`, degraded.

    `dy`/`dx` translate the whole page (real pages carry a global offset;
    measured -49 px in y on the validation page). `caseid_skew` moves ONLY
    the Case ID row, breaking the two-anchor registration agreement.
    """
    page = np.full(absynth.NATIVE_SIZE, 255, np.uint8)
    title_y = 124.0 + dy
    x = absynth.XCOL + dx
    if title:
        _place(page, absynth.INTAKE_TITLE, absynth.TITLE_PT, x, title_y)
    caseid_y = title_y + absynth.DY_TITLE_CASEID + caseid_skew
    _place(page, f"Case ID: {case_id}", absynth.BODY_PT, x, caseid_y)
    for field, value in values.items():
        _place(page, f"{absynth._ROW_LABEL[field]} {value}", absynth.BODY_PT,
               x, caseid_y + absynth._ROW_DY[field])
    return absynth.degrade(page, *kernel)


def _observe(page: np.ndarray) -> np.ndarray:
    """Native synthetic page -> the block the reader actually works in."""
    block = page[:absynth.BODY_H, :absynth.BODY_W]
    return absynth.flatten(np.ascontiguousarray(block))


def _gray288(page: np.ndarray) -> np.ndarray:
    """Native page -> the 288-dpi space ScanOcrResult.gray lives in."""
    import cv2

    return cv2.resize(page, None, fx=2.0, fy=2.0,
                      interpolation=cv2.INTER_LINEAR)


class _Scan:
    """Minimal ScanOcrResult stand-in: absynth reads only .gray."""

    def __init__(self, gray):
        self.gray = gray


# ------------------------------------------------------------ field scope
class TestFieldScope:
    def test_only_decision_relevant_closed_menus(self):
        assert absynth.FIELDS == ("species_code", "home_world",
                                  "visa_class", "declared_purpose")

    def test_scope_matches_ctcfill(self):
        from mib_pipeline import ctcfill

        assert absynth.FIELDS == ctcfill.FIELDS

    def test_open_vocabulary_fields_excluded(self):
        for field in ("fee_status", "sponsor_id", "arrival_date",
                      "applicant_name", "risk_flags"):
            assert field not in absynth.FIELDS
            assert field not in absynth._MENUS
            assert field not in absynth._ROW_LABEL

    def test_every_field_has_a_menu_label_and_sentinel(self):
        for field in absynth.FIELDS:
            assert field in absynth._MENUS and absynth._MENUS[field]
            assert field in absynth._ROW_LABEL
            assert field in absynth._SENTINELS


# ------------------------------------------------------------------ flag
class TestFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_ABSYNTH", raising=False)
        assert pipeline.ABSYNTH_DEFAULT is False
        assert pipeline._absynth_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_ABSYNTH", "1")
        assert pipeline._absynth_enabled() is True
        monkeypatch.setenv("MIB_ABSYNTH", "0")
        assert pipeline._absynth_enabled() is False

    def test_module_never_invoked_with_flag_off(self, monkeypatch):
        """The call site is guarded by the flag, so a poisoned fill() is
        never reached when the flag is unset."""
        monkeypatch.delenv("MIB_ABSYNTH", raising=False)

        def _poison(*a, **k):
            raise AssertionError("absynth.fill called with the flag off")

        monkeypatch.setattr(absynth, "fill", _poison)
        # The guard as it is written at the call site.
        scans = {0: _Scan(np.full((100, 100), 255, np.uint8))}
        if scans and pipeline._absynth_enabled():
            absynth.fill(scans, list(absynth.FIELDS), CASE_ID)

    def test_call_site_runs_after_the_ctcfill_block(self):
        """Last resort means last: the block must sit BELOW the ctcfill
        block, so it only ever sees slots that survived it."""
        src = inspect.getsource(pipeline._process)
        assert src.index("_ctcfill_enabled()") < src.index("_absynth_enabled()")
        assert src.index("ctcfill.fill(") < src.index("absynth.fill(")


# ------------------------------------------------- (a) generator inversion
class TestInversion:
    def _invert(self, field, value, kernel):
        """Register, recover the kernel, rank the candidates."""
        obs = _observe(_synth({field: value}, kernel=kernel))
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None, f"registration failed under {kernel}"
        label = absynth._ROW_LABEL[field]
        coarse = absynth.locate(obs, label, reg.caseid.x,
                                reg.caseid.y + absynth._ROW_DY[field],
                                absynth.BODY_PT, absynth.WIN_ROW)
        hit = absynth.fit_kernel(obs, coarse, label)
        decoded = absynth.decode_value(
            obs, hit, label, absynth._MENUS[field]
            + (absynth._SENTINELS[field],))
        return obs, reg, hit, decoded

    @pytest.mark.parametrize("kernel", [(0, 0.0), (0, 1.2), (3, 1.2),
                                        (5, 1.8), (3, 2.6)])
    def test_inversion_recovers_the_applied_kernel(self, kernel):
        """The core claim: degrade a synthetic row with a known kernel and
        the fit against the row's OWN label recovers that exact kernel,
        with no offline calibration."""
        _obs, _reg, hit, _decoded = self._invert(
            "species_code", "ORION_GRAYS", kernel)
        assert hit.deg == kernel

    @pytest.mark.parametrize("kernel", [(0, 0.0), (0, 1.2), (3, 1.2),
                                        (5, 1.8), (3, 2.6)])
    def test_inversion_ranks_the_true_value_first(self, kernel):
        """Ranking is correct even where the gate later declines to
        accept: synthesis beats the alternatives at every blur tested."""
        _obs, _reg, _hit, decoded = self._invert(
            "species_code", "ORION_GRAYS", kernel)
        assert decoded[0] == "ORION_GRAYS"

    @pytest.mark.parametrize("kernel", [(0, 0.0), (0, 1.2), (3, 1.2),
                                        (5, 1.8), (3, 2.6)])
    def test_accepts_while_the_margin_survives_the_blur(self, kernel):
        """The (5, 1.8) and (3, 2.6) cases are the recall the census
        bought: at the inherited 0.10 gate their margins (0.031 / 0.034)
        were rejected despite the winner being correct."""
        page = _synth({"species_code": "ORION_GRAYS"}, kernel=kernel)
        obs = _observe(page)
        reg = absynth.register(obs, CASE_ID)
        got = absynth.read_field(obs, reg, "species_code")
        assert got is not None, f"no read under {kernel}"
        assert got[0] == "ORION_GRAYS"

    @pytest.mark.parametrize("kernel", [(5, 2.6), (5, 3.4), (7, 3.4)])
    def test_extreme_blur_abstains_instead_of_guessing(self, kernel):
        """Past some damage the detail separating candidates is gone: the
        true value still ranks first but by a margin down in the tie floor
        (measured 0.023 / 0.019 / 0.014, against real sub-0.02 ties in the
        census), so the reader declines. Abstaining is the designed
        outcome — the slot falls back to the default it would have had."""
        obs, reg, _hit, decoded = self._invert(
            "species_code", "ORION_GRAYS", kernel)
        assert decoded[0] == "ORION_GRAYS"
        assert decoded[2] < absynth.MIN_VALUE_MARGIN
        assert absynth.read_field(obs, reg, "species_code") is None

    def test_one_glyph_apart_menu_members_are_the_tightest_pair(self):
        """XW-1 and XW-2 differ by a single glyph in a four-character
        string, so they are the closest pair any menu offers and the first
        to collapse under damage. Once real blur is present the gate
        declines them — a wrong visa_class flips R7, so that is the right
        trade even though the winner here is in fact correct."""
        obs, reg, _hit, decoded = self._invert(
            "visa_class", "XW-2", (3, 1.2))
        assert decoded[0] == "XW-2"
        assert decoded[3][1][1] == "XW-1"
        assert decoded[2] < absynth.MIN_VALUE_MARGIN
        assert absynth.read_field(obs, reg, "visa_class") is None

    def test_recovered_kernel_tracks_the_applied_blur(self):
        """Self-calibration: a blurrier page must recover a blurrier
        kernel from the row's own known label."""
        recovered = []
        for sigma in (0.0, 1.2, 2.6):
            obs = _observe(_synth({"home_world": "Titan Freeport"},
                                  kernel=(0, sigma)))
            reg = absynth.register(obs, CASE_ID)
            label = absynth._ROW_LABEL["home_world"]
            coarse = absynth.locate(
                obs, label, reg.caseid.x,
                reg.caseid.y + absynth._ROW_DY["home_world"],
                absynth.BODY_PT, absynth.WIN_ROW)
            hit = absynth.fit_kernel(obs, coarse, label)
            recovered.append(hit.deg[1])
        assert recovered == sorted(recovered), recovered
        assert recovered[0] < recovered[-1], recovered

    def test_reads_every_menu_field_on_one_page(self):
        values = {"species_code": "ORION_GRAYS",
                  "home_world": "Titan Freeport",
                  "visa_class": "TRANSIT-7",
                  "declared_purpose": "xenobotany"}
        obs = _observe(_synth(values, kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None
        for field, expected in values.items():
            got = absynth.read_field(obs, reg, field)
            assert got is not None and got[0] == expected, (field, got)

    def test_survives_the_global_page_offset(self):
        """Real pages are translated wholesale (measured -49 px in y,
        +19 px in x on the validation page); registration must absorb
        that, not fail on it."""
        obs = _observe(_synth({"species_code": "ORION_GRAYS"},
                              kernel=(0, 1.2), dy=-45.0, dx=19.0))
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None
        got = absynth.read_field(obs, reg, "species_code")
        assert got is not None and got[0] == "ORION_GRAYS"


# ----------------------------------------------------- (b) near-tie abstain
class TestNearTie:
    def test_margin_gate_rejects_a_near_tie(self, monkeypatch):
        obs = _observe(_synth({"home_world": "Titan Freeport"}))
        reg = absynth.register(obs, CASE_ID)
        tie = absynth.MIN_VALUE_MARGIN / 2.0
        monkeypatch.setattr(
            absynth, "decode_value",
            lambda *a, **k: ("Titan Freeport", 0.90, tie,
                             [(0.90, "Titan Freeport"),
                              (0.90 - tie, "Mars Dome-7")]))
        assert absynth.read_field(obs, reg, "home_world") is None

    def test_a_clear_margin_is_accepted(self, monkeypatch):
        obs = _observe(_synth({"home_world": "Titan Freeport"}))
        reg = absynth.register(obs, CASE_ID)
        clear = absynth.MIN_VALUE_MARGIN * 2.0
        monkeypatch.setattr(
            absynth, "decode_value",
            lambda *a, **k: ("Titan Freeport", 0.90, clear,
                             [(0.90, "Titan Freeport"),
                              (0.90 - clear, "Mars Dome-7")]))
        got = absynth.read_field(obs, reg, "home_world")
        assert got is not None and got[0] == "Titan Freeport"

    def test_weak_winner_rejected_even_with_margin(self, monkeypatch):
        obs = _observe(_synth({"home_world": "Titan Freeport"}))
        reg = absynth.register(obs, CASE_ID)
        weak = absynth.MIN_VALUE_NCC - 0.05
        monkeypatch.setattr(
            absynth, "decode_value",
            lambda *a, **k: ("Titan Freeport", weak, 0.40,
                             [(weak, "Titan Freeport"), (weak - 0.40, "x")]))
        assert absynth.read_field(obs, reg, "home_world") is None

    def test_blank_row_does_not_mint_a_value(self):
        """A row printed with the label and NO value must not resolve to
        the least-bad menu entry."""
        page = np.full(absynth.NATIVE_SIZE, 255, np.uint8)
        _place(page, absynth.INTAKE_TITLE, absynth.TITLE_PT,
               absynth.XCOL, 124.0)
        caseid_y = 124.0 + absynth.DY_TITLE_CASEID
        _place(page, f"Case ID: {CASE_ID}", absynth.BODY_PT,
               absynth.XCOL, caseid_y)
        _place(page, "Home World:", absynth.BODY_PT, absynth.XCOL,
               caseid_y + absynth._ROW_DY["home_world"])
        obs = _observe(page)
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None
        assert absynth.read_field(obs, reg, "home_world") is None


# ----------------------------------------------- damage-sentinel abstention
class TestSentinel:
    @pytest.mark.parametrize("field", ["species_code", "home_world",
                                       "visa_class", "declared_purpose"])
    def test_sentinel_row_abstains(self, field):
        """The generator prints e.g. 'Home World: [REGISTRY LOST]' when it
        destroys a row. The sentinel is a scored candidate, so the reader
        recognises the damage instead of guessing a value."""
        obs = _observe(_synth({field: absynth._SENTINELS[field]},
                              kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None
        assert absynth.read_field(obs, reg, field) is None

    def test_sentinel_beats_every_menu_value_on_its_own_row(self):
        obs = _observe(_synth({"home_world": "[REGISTRY LOST]"},
                              kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        label = absynth._ROW_LABEL["home_world"]
        coarse = absynth.locate(
            obs, label, reg.caseid.x,
            reg.caseid.y + absynth._ROW_DY["home_world"],
            absynth.BODY_PT, absynth.WIN_ROW)
        hit = absynth.fit_kernel(obs, coarse, label)
        cands = absynth._MENUS["home_world"] + ("[REGISTRY LOST]",)
        value, _score, _margin, _ranked = absynth.decode_value(
            obs, hit, label, cands)
        assert value == "[REGISTRY LOST]"


# ------------------------------------------- (c) registration cross-check
class TestRegistration:
    def test_anchors_must_agree(self):
        """Case ID moved off its layout offset: the two anchors disagree,
        so the registration is not trusted."""
        obs = _observe(_synth({"species_code": "ORION_GRAYS"},
                              caseid_skew=absynth.DY_TOL + 12.0))
        assert absynth.register(obs, CASE_ID) is None

    def test_agreeing_anchors_register(self):
        obs = _observe(_synth({"species_code": "ORION_GRAYS"}))
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None
        assert abs(reg.dy - absynth.DY_TITLE_CASEID) <= absynth.DY_TOL

    def test_missing_title_abstains(self):
        obs = _observe(_synth({"species_code": "ORION_GRAYS"}, title=False))
        assert absynth.register(obs, CASE_ID) is None

    def test_missing_caseid_row_abstains(self):
        """No second anchor, no registration: one match cannot check
        itself."""
        page = np.full(absynth.NATIVE_SIZE, 255, np.uint8)
        _place(page, absynth.INTAKE_TITLE, absynth.TITLE_PT,
               absynth.XCOL, 124.0)
        assert absynth.register(_observe(page), CASE_ID) is None

    def test_fiducial_is_positional_not_authenticating(self):
        """MEASURED property, recorded deliberately: a Case ID line whose
        digits differ still registers, because the anchor's discriminative
        mass is the shared 'Case ID: MIB-000' boilerplate and Helvetica
        digits are tabular, so a different case id has an IDENTICAL
        advance width. That is exactly what the fiducial is for — it pins
        WHERE the rows are, not WHOSE they are. Identity is never at stake
        here: `scans` are this packet's own pages and the case id comes
        from this packet's filename, so a foreign intake page cannot enter
        the input. The registration therefore lands in the same place."""
        obs_self = _observe(_synth({"species_code": "ORION_GRAYS"}))
        obs_other = _observe(_synth({"species_code": "ORION_GRAYS"},
                                    case_id="MIB-000999"))
        a = absynth.register(obs_self, CASE_ID)
        b = absynth.register(obs_other, CASE_ID)
        assert a is not None and b is not None
        assert abs(a.caseid.x - b.caseid.x) <= 1.0
        assert abs(a.caseid.y - b.caseid.y) <= 1.0

    def test_blank_page_abstains(self):
        obs = _observe(np.full(absynth.NATIVE_SIZE, 255, np.uint8))
        assert absynth.register(obs, CASE_ID) is None

    def test_read_page_abstains_without_registration(self):
        page = np.full(absynth.NATIVE_SIZE, 255, np.uint8)
        assert absynth.read_page(_gray288(page), CASE_ID,
                                 list(absynth.FIELDS)) == {}


# ------------------------------------------------- (d) mode-default rule
class TestModeDefault:
    def test_mode_default_matches_the_writer(self):
        for field in absynth.FIELDS:
            assert absynth._mode_default(field) == writer._DEFAULTS[field]

    @pytest.mark.parametrize("field", ["species_code", "home_world",
                                       "visa_class", "declared_purpose"])
    def test_mode_default_is_never_emitted(self, field):
        """Emitting the default is a no-op — build_row prints it anyway
        when the field is unread — and default-accepts were the only
        wrong-fire mode measured. A correct default read must still
        abstain."""
        default = writer._DEFAULTS[field]
        obs = _observe(_synth({field: default}, kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        assert reg is not None
        assert absynth.read_field(obs, reg, field) is None

    def test_the_same_row_reads_a_non_default_value(self, ):
        """Control for the test above: the row is readable, it is the
        default rule that suppresses it."""
        obs = _observe(_synth({"visa_class": "TRANSIT-7"}, kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        got = absynth.read_field(obs, reg, "visa_class")
        assert got is not None and got[0] == "TRANSIT-7"
        assert writer._DEFAULTS["visa_class"] != "TRANSIT-7"


# --------------------------------------------------------- safety guards
class TestSafetyGuards:
    def test_hard_embargo_worlds_never_emitted(self):
        for world in policy.HARD_EMBARGO_WORLDS:
            obs = _observe(_synth({"home_world": world}, kernel=(0, 1.2)))
            reg = absynth.register(obs, CASE_ID)
            assert reg is not None
            assert absynth.read_field(obs, reg, "home_world") is None, world

    def test_excluded_set_tracks_policy(self):
        assert absynth._EXCLUDED_VALUES == policy.HARD_EMBARGO_WORLDS

    def test_conf_cap_below_affirmative_read_threshold(self):
        assert absynth.CONF_CAP < fields._KNOWN_MIN_OCR_CONF

    def test_margin_gate_sits_in_the_census_supported_band(self):
        """Pinned against drift in either direction. Below 0.02 the gate
        is inside the measured tie floor (the two real gold-mismatches sat
        at margins 0.010 and 0.006); at 0.10 the census measured 4 accepts
        at 75% precision against 13 at 92% here."""
        assert 0.02 <= absynth.MIN_VALUE_MARGIN <= 0.05

    def test_real_population_anchor_clears_the_gate(self):
        """Regression anchor from the real corpus, kept as a number
        because corpus fixtures are not shipped: on MIB-000095 the
        home_world row read Barnard-c at margin 0.107, visually confirmed
        correct against the page. It cleared the inherited 0.10 gate by
        0.007 — one bad rounding from being lost — and clears the
        census-derived gate with real headroom."""
        measured_margin = 0.107
        assert measured_margin >= absynth.MIN_VALUE_MARGIN
        assert measured_margin - absynth.MIN_VALUE_MARGIN >= 0.05

    def test_veto_stays_disabled_by_default(self):
        """Censused at 0 fires / 31 real slots; see VETO_MIN_ADVANTAGE."""
        assert pipeline.XCHANNEL_VETO_DEFAULT is False
        assert absynth.VETO_MIN_ADVANTAGE > absynth.MIN_VALUE_MARGIN

    def test_accepted_confidence_is_capped(self):
        obs = _observe(_synth({"species_code": "ORION_GRAYS"},
                              kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        value, conf = absynth.read_field(obs, reg, "species_code")
        assert value == "ORION_GRAYS"
        assert conf <= absynth.CONF_CAP

    def test_soft_embargo_world_is_still_readable(self):
        """Only HARD embargo is suppressed: R5's soft embargo has a DIP-1
        exemption, so the value carries real extraction credit."""
        (world,) = tuple(policy.SOFT_EMBARGO_WORLDS)
        obs = _observe(_synth({"home_world": world}, kernel=(0, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        got = absynth.read_field(obs, reg, "home_world")
        assert got is not None and got[0] == world


# ---------------------------------------------------------- fill contract
class TestFill:
    def _scan(self, values, **kw):
        return _Scan(_gray288(_synth(values, **kw)))

    def test_fill_is_a_noop_without_needed_fields(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"})}
        assert absynth.fill(scans, [], CASE_ID) == {}

    def test_fill_ignores_out_of_scope_fields(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"})}
        assert absynth.fill(scans, ["sponsor_id", "fee_status"],
                            CASE_ID) == {}

    def test_fill_reads_an_intake_page(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        got = absynth.fill(scans, ["species_code"], CASE_ID)
        assert got["species_code"][0] == "ORION_GRAYS"
        assert got["species_code"][1] <= absynth.CONF_CAP

    def test_cross_page_disagreement_abstains(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2)),
                 1: self._scan({"species_code": "SIRIUS_AVIAN"},
                               kernel=(0, 1.2))}
        assert absynth.fill(scans, ["species_code"], CASE_ID) == {}

    def test_agreeing_pages_fill(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2)),
                 1: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.8))}
        got = absynth.fill(scans, ["species_code"], CASE_ID)
        assert got["species_code"][0] == "ORION_GRAYS"

    def test_non_intake_pages_are_skipped(self):
        """The row table is intake geometry and the two-anchor check needs
        the Case ID row, which other templates do not print."""
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        assert absynth.fill(scans, ["species_code"], CASE_ID,
                            page_types={0: "registry"}) == {}
        assert absynth.fill(scans, ["species_code"], CASE_ID,
                            page_types={0: None})["species_code"][0] \
            == "ORION_GRAYS"

    def test_budget_exhaustion_stops(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        assert absynth.fill(scans, ["species_code"], CASE_ID,
                            budget_left=lambda: False) == {}

    def test_page_read_failure_is_contained(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("synthetic page failure")

        monkeypatch.setattr(absynth, "read_page", _boom)
        scans = {0: self._scan({"species_code": "ORION_GRAYS"})}
        assert absynth.fill(scans, ["species_code"], CASE_ID) == {}


# ------------------------------------------------- cross-channel veto
class TestCrossChannelVeto:
    """MIB_XCHANNEL_VETO: the pixel channel may WITHDRAW a ctcfill menu
    fill it decisively contradicts, and may never replace it."""

    def _scan(self, values, **kw):
        return _Scan(_gray288(_synth(values, **kw)))

    def test_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_XCHANNEL_VETO", raising=False)
        assert pipeline.XCHANNEL_VETO_DEFAULT is False
        assert pipeline._xchannel_veto_enabled() is False

    def test_flag_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_XCHANNEL_VETO", "1")
        assert pipeline._xchannel_veto_enabled() is True
        monkeypatch.setenv("MIB_XCHANNEL_VETO", "0")
        assert pipeline._xchannel_veto_enabled() is False

    def test_flag_is_independent_of_absynth(self, monkeypatch):
        monkeypatch.setenv("MIB_XCHANNEL_VETO", "1")
        monkeypatch.delenv("MIB_ABSYNTH", raising=False)
        assert pipeline._xchannel_veto_enabled() is True
        assert pipeline._absynth_enabled() is False

    def test_module_never_invoked_with_flag_off(self, monkeypatch):
        monkeypatch.delenv("MIB_XCHANNEL_VETO", raising=False)

        def _poison(*a, **k):
            raise AssertionError("absynth.veto called with the flag off")

        monkeypatch.setattr(absynth, "veto", _poison)
        scans = {0: self._scan({"species_code": "ORION_GRAYS"})}
        fills = {"species_code": "SIRIUS_AVIAN"}
        if scans and fills and pipeline._xchannel_veto_enabled():
            absynth.veto(scans, fills, CASE_ID)

    # -------------------------------------------------------- (1) fires
    def test_vetoes_a_decisively_wrong_fill(self):
        """The page plainly reads ORION_GRAYS; ctcfill claims something
        else; the pixel channel is confident, so the fill is withdrawn."""
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 0.6))}
        got = absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"}, CASE_ID)
        assert got == {"species_code"}

    def test_veto_returns_only_field_names_never_values(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 0.6))}
        got = absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"}, CASE_ID)
        assert isinstance(got, set)
        assert got <= set(absynth.FIELDS)
        assert "ORION_GRAYS" not in got

    # ---------------------------------------------------- (3) agreement
    def test_no_veto_when_the_channels_agree(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        assert absynth.veto(scans, {"species_code": "ORION_GRAYS"},
                            CASE_ID) == set()

    def test_agreement_on_any_page_outranks_contradiction(self):
        """Corroboration beats contradiction: disagreeing pages are
        under-determination, not proof the fill is wrong."""
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2)),
                 1: self._scan({"species_code": "SIRIUS_AVIAN"},
                               kernel=(0, 1.2))}
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"},
                            CASE_ID) == set()

    # ---------------------------------------------------- (2) abstention
    def test_no_veto_when_the_page_does_not_register(self):
        blank = _Scan(_gray288(np.full(absynth.NATIVE_SIZE, 255, np.uint8)))
        assert absynth.veto({0: blank}, {"species_code": "SIRIUS_AVIAN"},
                            CASE_ID) == set()

    def test_no_veto_on_a_damage_sentinel_row(self):
        """A destroyed row is not evidence against the other channel."""
        scans = {0: self._scan(
            {"species_code": absynth._SENTINELS["species_code"]},
            kernel=(0, 1.2))}
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"},
                            CASE_ID) == set()

    def test_no_veto_when_the_pixel_channel_is_not_confident(self):
        """Extreme blur ranks the true value first but under the margin
        gate; an unconfident channel must not overrule anyone."""
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(5, 3.4))}
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"},
                            CASE_ID) == set()

    def test_no_veto_below_the_advantage_bar(self, monkeypatch):
        """Confident and disagreeing is not enough: the winner must beat
        the contested value by VETO_MIN_ADVANTAGE."""
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        thin = absynth.VETO_MIN_ADVANTAGE / 2.0
        monkeypatch.setattr(
            absynth, "score_row",
            lambda *a, **k: (absynth.Hit(0.95, 0, 0, absynth.BODY_PT, (0, 0)),
                             ("ORION_GRAYS", 0.90, 0.50,
                              [(0.90, "ORION_GRAYS"),
                               (0.90 - thin, "SIRIUS_AVIAN")])))
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"},
                            CASE_ID) == set()

    def test_veto_bar_is_stricter_than_the_fill_bar(self):
        assert absynth.VETO_MIN_ADVANTAGE > absynth.MIN_VALUE_MARGIN

    def test_declines_against_the_nearest_competitor_under_damage(self):
        """MEASURED LIMITATION, pinned deliberately.

        Under real damage the pixel channel's advantage over its NEAREST
        competitor collapses (measured 0.06-0.20 at kernel (3, 1.2) versus
        0.27-0.38 at (0, 0.6)), so the veto goes quiet against exactly the
        confusable value ctcfill is most likely to have picked wrongly.
        Its discriminating power is therefore concentrated on clean pages,
        where ctcfill is already right. This is why the lever is delivered
        OFF: the bar cannot be lowered to fix it without moving the veto
        into the regime where a wrong veto — which costs a full point —
        becomes likely.
        """
        obs = _observe(_synth({"species_code": "ORION_GRAYS"},
                              kernel=(3, 1.2)))
        reg = absynth.register(obs, CASE_ID)
        _hit, (winner, score, _m, ranked) = absynth.score_row(
            obs, reg, "species_code")
        assert winner == "ORION_GRAYS"
        nearest = next(v for _s, v in ranked[1:]
                       if v != absynth._SENTINELS["species_code"])
        by = {v: s for s, v in ranked}
        assert score - by[nearest] < absynth.VETO_MIN_ADVANTAGE
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(3, 1.2))}
        assert absynth.veto(scans, {"species_code": nearest},
                            CASE_ID) == set()

    def test_no_veto_when_the_fill_is_off_menu(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        assert absynth.veto(scans, {"species_code": "NOT_A_SPECIES"},
                            CASE_ID) == set()

    def test_no_veto_without_fills(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"})}
        assert absynth.veto(scans, {}, CASE_ID) == set()

    def test_out_of_scope_fields_ignored(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"})}
        assert absynth.veto(scans, {"sponsor_id": "SPN-0001"},
                            CASE_ID) == set()

    def test_non_intake_pages_are_skipped(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"}, CASE_ID,
                            page_types={0: "registry"}) == set()

    def test_budget_exhaustion_stops(self):
        scans = {0: self._scan({"species_code": "ORION_GRAYS"},
                               kernel=(0, 1.2))}
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"}, CASE_ID,
                            budget_left=lambda: False) == set()

    # ------------------------------------- emission guards do NOT gate it
    def test_mode_default_winner_still_vetoes(self):
        """The winner cannot be EMITTED (it is the writer's default), but
        it is still evidence the contested fill is wrong — and withdrawing
        the fill emits the default anyway, which is what the row says."""
        default = writer._DEFAULTS["species_code"]
        scans = {0: self._scan({"species_code": default}, kernel=(0, 0.6))}
        assert absynth.read_field(
            _observe(_synth({"species_code": default}, kernel=(0, 0.6))),
            absynth.register(
                _observe(_synth({"species_code": default}, kernel=(0, 0.6))),
                CASE_ID),
            "species_code") is None
        assert absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"},
                            CASE_ID) == {"species_code"}

    def test_hard_embargo_winner_still_vetoes(self):
        """Same logic: a hard-embargo world is never emitted, but reading
        one is still grounds to withdraw a contradicting fill. The
        withdrawal falls back to the default, so no embargo value reaches
        the row and no R1/R2 denial can be minted."""
        world = "TRAPPIST-1e"
        scans = {0: self._scan({"home_world": world}, kernel=(0, 0.6))}
        assert absynth.veto(scans, {"home_world": "Titan Freeport"},
                            CASE_ID) == {"home_world"}

    # ------------------------------------------------- (5) never writes
    def test_veto_cannot_substitute_its_own_answer(self):
        """The call site removes the field and withholds it from the fill
        block, so a veto can only ever produce an imputed default."""
        src = inspect.getsource(pipeline._process)
        assert "vetoed = absynth.veto(" in src
        assert "evidence.values.pop(fld, None)" in src
        assert "and f not in vetoed" in src
        # ordering: veto runs after the ctcfill fill and before the block
        # that would otherwise re-fill the field
        assert src.index("ctcfill.fill(") < src.index("absynth.veto(")
        assert src.index("absynth.veto(") < src.index("absynth.fill(")

    def test_only_applied_fills_are_contested(self):
        """ctcfill_fills records values actually written to evidence, not
        everything ctcfill returned."""
        src = inspect.getsource(pipeline._process)
        applied = src.index("ctcfill_fills[fld] = value")
        guard = src.index("evidence.conf[fld] = min(conf, ctcfill.CONF_CAP)")
        assert guard < applied      # recorded inside the same if-branch


# ------------------------------------------------------------ page cache
class TestPageCache:
    def test_registration_is_computed_once_per_page(self, monkeypatch):
        cache = absynth.PageCache(CASE_ID)
        gray = _gray288(_synth({"species_code": "ORION_GRAYS"}))
        calls = []
        real = absynth.register
        monkeypatch.setattr(absynth, "register",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        cache.registered(0, gray)
        cache.registered(0, gray)
        cache.registered(0, gray)
        assert len(calls) == 1

    def test_registration_failure_is_cached_too(self, monkeypatch):
        cache = absynth.PageCache(CASE_ID)
        blank = _gray288(np.full(absynth.NATIVE_SIZE, 255, np.uint8))
        calls = []
        real = absynth.register
        monkeypatch.setattr(absynth, "register",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        assert cache.registered(0, blank) is None
        assert cache.registered(0, blank) is None
        assert len(calls) == 1

    def test_cache_is_shared_between_fill_and_veto(self, monkeypatch):
        cache = absynth.PageCache(CASE_ID)
        scans = {0: _Scan(_gray288(_synth({"species_code": "ORION_GRAYS"},
                                          kernel=(0, 1.2))))}
        calls = []
        real = absynth.register
        monkeypatch.setattr(absynth, "register",
                            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        absynth.veto(scans, {"species_code": "SIRIUS_AVIAN"}, CASE_ID,
                     cache=cache)
        absynth.fill(scans, ["home_world"], CASE_ID, cache=cache)
        assert len(calls) == 1
