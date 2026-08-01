"""MIB_ESC_VIEWS gate for the multi-view rotation-aware escalation.

The multi-view escalation (best_rot + 180-degree hedge + border pad) costs
extra OCR passes on every escalated case and is the prime suspect for the
6.0 s/PDF budget breach (docker_check 6.06-6.32 vs ship 5.39). This gate
makes it switchable: "1" = multi-view (the current tree's behavior),
"0"/unset = the legacy pre-escalation single view (the shipped image's
behavior). The Docker image runs with NO custom environment, so the
module-level default constants ARE the ship behavior; the final-rebuild
decision sets them.
"""
from __future__ import annotations

import numpy as np
import pytest

from mib_pipeline import ocr, pipeline
from mib_pipeline.ocr import ScanOcrResult


def _gray() -> np.ndarray:
    return np.full((100, 200), 128, dtype=np.uint8)


@pytest.fixture
def esc_on(monkeypatch):
    monkeypatch.setenv("MIB_ESC_VIEWS", "1")


@pytest.fixture
def esc_unset(monkeypatch):
    monkeypatch.delenv("MIB_ESC_VIEWS", raising=False)


class TestFlagResolution:
    def test_unset_falls_back_to_module_constant(self, esc_unset, monkeypatch):
        monkeypatch.setattr(pipeline, "ESC_VIEWS_DEFAULT", False)
        assert pipeline._esc_views_enabled() is False
        monkeypatch.setattr(pipeline, "ESC_VIEWS_DEFAULT", True)
        assert pipeline._esc_views_enabled() is True

    def test_env_1_forces_on_over_constant(self, esc_on, monkeypatch):
        monkeypatch.setattr(pipeline, "ESC_VIEWS_DEFAULT", False)
        assert pipeline._esc_views_enabled() is True

    def test_env_0_forces_off_over_constant(self, monkeypatch):
        monkeypatch.setenv("MIB_ESC_VIEWS", "0")
        monkeypatch.setattr(pipeline, "ESC_VIEWS_DEFAULT", True)
        assert pipeline._esc_views_enabled() is False

    def test_shipped_default_is_legacy_single_view(self):
        # The rebuild decision may flip this; the test documents (and
        # protects) what the tree currently ships without env overrides.
        assert pipeline.ESC_VIEWS_DEFAULT is False

    def test_rot_probe_honors_its_default_constant(self, monkeypatch):
        monkeypatch.delenv("MIB_ROT_PROBE", raising=False)
        monkeypatch.setattr(pipeline, "ROT_PROBE_DEFAULT", True)
        assert pipeline._rot_probe_enabled() is True
        monkeypatch.setenv("MIB_ROT_PROBE", "0")
        assert pipeline._rot_probe_enabled() is False


class TestViewsUnderGate:
    def test_flag_off_non_upright_gets_the_raw_render(self, esc_unset):
        # Legacy (shipped-image) behavior: escalation engines read the
        # render as-is, sideways or not — no counter-rotation, no pad.
        gray = _gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=False, best_rot=1)
        views = pipeline._escalation_views(result)
        assert len(views) == 1 and views[0] is gray

    def test_flag_off_upright_unchanged(self, esc_unset):
        gray = _gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=True)
        views = pipeline._escalation_views(result)
        assert len(views) == 1 and views[0] is gray

    def test_flag_on_multi_view_matches_current_behavior(self, esc_on):
        gray = _gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=False, best_rot=1)
        views = pipeline._escalation_views(result)
        expected = [ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, k=k)))
                    for k in {1, 3}]
        assert len(views) == len(expected)
        for got, want in zip(views, expected):
            assert np.array_equal(got, want)

    def test_flag_on_upright_still_single_raw_view(self, esc_on):
        gray = _gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=True)
        views = pipeline._escalation_views(result)
        assert len(views) == 1 and views[0] is gray

    def test_probe_selection_takes_precedence_even_when_gate_off(
            self, esc_unset):
        # The A2 probe carries its own flag; when it produced a selection
        # the selection defines the views regardless of MIB_ESC_VIEWS.
        gray = _gray()
        result = ScanOcrResult(lines=[], gray=gray, upright=False, best_rot=1)
        choice = pipeline._RotChoice(k=2, confident=True)
        views = pipeline._escalation_views(result, choice)
        assert len(views) == 1
        assert np.array_equal(
            views[0], ocr.pad_for_ocr(np.ascontiguousarray(np.rot90(gray, 2))))
