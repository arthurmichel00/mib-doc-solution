"""Closed-vocab fee-value shape classifier (spec §9, D-FEE reading extension).

The fee value regions print one word from a closed vocabulary in 9 pt regular
Helvetica at template-fixed positions. Templates are rendered by pymupdf (the
same Helvetica metrics that rendered the corpus). Acceptance is abstention-
first: margin-gated correlation with width + residual-ink vetoes; anything
ambiguous returns None. Thresholds are tuned on synthetic degradations ONLY
(spec §9.2) — these tests pin the behavior the tuning must preserve.
"""
from __future__ import annotations

import cv2
import numpy as np
import pymupdf
import pytest

from mib_pipeline import fee_shape

RNG = np.random.default_rng(8090)


# ------------------------------------------------------------- crop factory


def _render_receipt_page(status="paid", amount="$809.00", waiver="N/A",
                         dpi=576):
    """Rasterize a receipt page at the corpus template's exact geometry."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 48), "MIB Fee Receipt", fontsize=14,
                     fontname="hebo")
    rows = [("Case ID", "MIB-000123", 133), ("Fee Status", status, 157),
            ("Amount", amount, 181), ("Waiver Code", waiver, 205)]
    for label, value, y in rows:
        page.insert_text((88, y - 2), label, fontsize=8, fontname="hebo")
        page.insert_text((238, y), value, fontsize=9, fontname="helv")
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width).copy()
    doc.close()
    return gray


def _degrade(gray, jpeg_q=None, blur=0.0, skew=0.0, noise=0.0):
    img = gray.copy()
    if skew:
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        img = np.clip(img.astype(np.float32)
                      + RNG.normal(0, noise, img.shape), 0, 255
                      ).astype(np.uint8)
    if jpeg_q:
        ok, buf = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def _crops(status="paid", amount="$809.00", waiver="N/A", **degrade_kw):
    page = _degrade(_render_receipt_page(status, amount, waiver),
                    **degrade_kw)
    return fee_shape.crop_regions(page)


# ---------------------------------------------------------------- templates


class TestTemplates:
    def test_all_vocab_words_have_ink(self):
        for vocab in (fee_shape.STATUS_VOCAB, fee_shape.AMOUNT_VOCAB,
                      fee_shape.WAIVER_VOCAB):
            for word in vocab:
                tmpl = fee_shape._template(word)
                assert tmpl.size > 0 and float(tmpl.max()) > 0, word

    def test_crop_regions_cover_the_value_rows(self):
        crops = _crops()
        assert set(crops) == {"fee_status", "fee_amount", "waiver_code"}
        for crop in crops.values():
            assert crop.size > 0


# ------------------------------------------------------- clean classification


class TestCleanClassification:
    @pytest.mark.parametrize("word", fee_shape.STATUS_VOCAB)
    def test_status_words_classify_clean(self, word):
        read = fee_shape.classify_status(_crops(status=word)["fee_status"])
        assert read is not None and read.value == word

    @pytest.mark.parametrize("word", fee_shape.AMOUNT_VOCAB)
    def test_amount_words_classify_clean(self, word):
        read = fee_shape.classify_amount(_crops(amount=word)["fee_amount"])
        assert read is not None and read.value == word

    @pytest.mark.parametrize("word", fee_shape.WAIVER_VOCAB)
    def test_waiver_words_classify_clean(self, word):
        read = fee_shape.classify_waiver(_crops(waiver=word)["waiver_code"])
        assert read is not None and read.value == word


# --------------------------------------------------- corpus-profile damage


# The measured corpus damage profile: JPEG q~58, blur, small skew (spec §9.2).
_MILD = dict(jpeg_q=58, blur=1.5, skew=0.5)
_HEAVY = dict(jpeg_q=40, blur=4.0, skew=0.8, noise=12.0)


class TestDegradedClassification:
    @pytest.mark.parametrize("word", fee_shape.STATUS_VOCAB)
    def test_mild_corpus_damage_still_classifies(self, word):
        read = fee_shape.classify_status(
            _crops(status=word, **_MILD)["fee_status"])
        assert read is not None and read.value == word

    @pytest.mark.parametrize("word", fee_shape.AMOUNT_VOCAB)
    def test_mild_corpus_damage_amounts_classify(self, word):
        read = fee_shape.classify_amount(
            _crops(amount=word, **_MILD)["fee_amount"])
        assert read is not None and read.value == word

    @pytest.mark.parametrize("word", fee_shape.STATUS_VOCAB)
    def test_heavy_damage_is_correct_or_abstain_never_wrong(self, word):
        read = fee_shape.classify_status(
            _crops(status=word, **_HEAVY)["fee_status"])
        assert read is None or read.value == word


# ------------------------------------------------------------- hazard vetoes


class TestHazards:
    def test_unpaid_never_classifies_paid(self):
        # "unpaid" contains "paid" as a sub-image; the residual-ink veto
        # must reject the substring alignment (spec §9.3/§9.5).
        for kw in ({}, _MILD, _HEAVY):
            read = fee_shape.classify_status(
                _crops(status="unpaid", **kw)["fee_status"])
            assert read is None or read.value == "unpaid", kw

    def test_blank_region_abstains(self):
        read = fee_shape.classify_status(_crops(status=" ")["fee_status"])
        assert read is None

    def test_noise_only_region_abstains(self):
        crop = RNG.integers(0, 255, (104, 880), dtype=np.uint8)
        assert fee_shape.classify_status(crop) is None

    def test_forged_amount_500_abstains(self):
        read = fee_shape.classify_amount(
            _crops(amount="$500.00")["fee_amount"])
        assert read is None

    def test_oversized_ink_abstains(self):
        # A value printed at triple size is not this template's value row.
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((238, 165), "paid", fontsize=27, fontname="helv")
        pix = page.get_pixmap(dpi=576, colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width).copy()
        doc.close()
        read = fee_shape.classify_status(
            fee_shape.crop_regions(gray)["fee_status"])
        assert read is None

    def test_wrong_word_for_the_region_abstains(self):
        # An amount string in the status region matches no status template.
        read = fee_shape.classify_status(
            _crops(status="$809.00")["fee_status"])
        assert read is None


# ------------------------------------------------------------ read contract


class TestReadContract:
    def test_reads_carry_score_and_margin(self):
        read = fee_shape.classify_status(_crops(status="waived")["fee_status"])
        assert read is not None
        assert 0.0 < read.score <= 1.0
        assert read.margin > 0.0
