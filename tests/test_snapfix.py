"""MIB_SNAPFIX: three text-level decode-repair mechanisms (default OFF).

1. Fusion (2-gram <-> 1-gram) edit costs with stricter re-accept
   (vocab.fusion_distance / vocab._fusion_rematch under match_vocab).
2. Context-licensed flag truncation-prefix (vocab._truncated_flag via
   parse_flags(flag_context=True)).
3. Cross-page per-digit sponsor vote (fields.sponsor_digit_vote, hooked
   in pipeline._process fill-only).

Every class carries flag-off inertness coverage: with MIB_SNAPFIX unset
the three mechanisms must be dead code.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mib_pipeline import fields, pipeline, vocab  # noqa: E402
from mib_pipeline.model import Line, Page, PageKind, Source  # noqa: E402

CASE_ID = "MIB-000123"


@pytest.fixture
def snapfix_on(monkeypatch):
    monkeypatch.setenv("MIB_SNAPFIX", "1")


@pytest.fixture
def snapfix_off(monkeypatch):
    monkeypatch.delenv("MIB_SNAPFIX", raising=False)


# ------------------------------------------------------------ flag gate


class TestFlagGate:
    def test_default_off(self, snapfix_off):
        assert pipeline.SNAPFIX_DEFAULT is False
        assert pipeline._snapfix_enabled() is False
        assert vocab._snapfix_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_SNAPFIX", "1")
        assert pipeline._snapfix_enabled() is True
        assert vocab._snapfix_enabled() is True
        monkeypatch.setenv("MIB_SNAPFIX", "0")
        assert pipeline._snapfix_enabled() is False
        assert vocab._snapfix_enabled() is False


# ------------------------------------- feature 1: fusion edit costs


class TestFusionDistance:
    def test_fusion_pairs_cost_one_reduced_charge(self):
        # rn->m benefits from the cheap-sub table (1.2 raw), cl->d does
        # not (2.0 raw); both cost exactly one _FUSION_COST fused.
        assert vocab.weighted_distance("rnedical", "medical") == \
            pytest.approx(1.2)
        assert vocab.fusion_distance("rnedical", "medical") == \
            pytest.approx(0.4)
        assert vocab.weighted_distance("cleclared", "declared") == \
            pytest.approx(2.0)
        assert vocab.fusion_distance("cleclared", "declared") == \
            pytest.approx(0.4)

    def test_both_directions(self):
        assert vocab.fusion_distance("medical", "rnedical") == \
            pytest.approx(0.4)

    def test_bridged_singles_only_on_fusion_path(self):
        # c<->o and e<->a are 1.0 raw, 0.4 fused
        assert vocab.weighted_distance("Qcrvera", "Qorvara") == \
            pytest.approx(2.0)
        assert vocab.fusion_distance("Qcrvera", "Qorvara") == \
            pytest.approx(0.8)

    def test_identity_and_case_fold(self):
        assert vocab.fusion_distance("same", "same") == 0.0
        assert vocab.fusion_distance("SAME", "same") == 0.0

    def test_never_exceeds_weighted_distance(self):
        import random
        import string

        rng = random.Random(8090)
        alphabet = string.ascii_letters + " .-|09"
        for _ in range(500):
            a = "".join(rng.choices(alphabet, k=rng.randint(0, 9)))
            b = "".join(rng.choices(alphabet, k=rng.randint(0, 9)))
            assert vocab.fusion_distance(a, b) <= \
                vocab.weighted_distance(a, b) + 1e-9, (a, b)


class TestMatchVocabFusionBridge:
    # "Sohx" is the name token Solix with its li strokes fused to h and
    # nothing else: raw distance 2.0/5 = 0.40 > 0.34, fused 0.4/5 = 0.08.
    # (li<->h has no cheap single-sub shadow, unlike the rn/ri/nn family.)
    def test_flag_off_keeps_the_rejection(self, snapfix_off):
        assert vocab.match_vocab("Sohx", vocab.NAME_TOKENS, 0.34) is None

    def test_flag_on_bridges(self, snapfix_on):
        assert vocab.match_vocab("Sohx", vocab.NAME_TOKENS, 0.34) == "Solix"

    def test_raw_accepts_stay_byte_identical(self, snapfix_on):
        # anything the raw pass accepts must not be re-routed
        assert vocab.match_vocab(
            "ORION GRAYS", vocab.SPECIES_CODES, 0.34) == "ORION_GRAYS"
        assert vocab.match_vocab(
            "rnedical consult", vocab.PURPOSES, 0.34) == "medical consult"
        assert vocab.match_vocab(
            "waived", vocab.FEE_STATUSES, 0.34) == "waived"

    def test_ambiguity_rejections_are_not_bridged(self, snapfix_on):
        # raw distance passes but the margin fails: the fusion path must
        # not be consulted (an ambiguous read stays unknown)
        assert vocab.match_vocab("abx", ["abc", "abd"], 0.5) is None

    def test_fusion_match_needs_tighter_threshold(self, snapfix_on):
        # one fusion + one plain error in a 6-char read of a 5-char entry:
        # raw 2.2/6 = 0.367 (rejected at both cuts), fused 1.4/6 = 0.233.
        # Accepted at the 0.34 cut (tight 0.255) but rejected at the 0.30
        # cut (tight 0.225) even though 0.233 <= 0.30: the fusion bridge
        # only accepts at the TIGHTER threshold.
        assert vocab.match_vocab("rnedyc", ["medic"], 0.34) == "medic"
        assert vocab.match_vocab("rnedyc", ["medic"], 0.30) is None

    def test_fusion_near_tie_rejected(self, snapfix_on):
        # both entries reachable by one fusion move (rn->m / nn->m), so
        # the fused distances tie exactly: margin must abstain
        assert vocab.match_vocab("abmc", ["abrnc", "abnnc"], 0.34) is None


# ------------------------- feature 2: flag truncation-prefix decode


class TestTruncatedFlag:
    def test_exact_unique_prefix(self, snapfix_on):
        got = vocab.parse_flags("illegi", flag_context=True)
        assert got == ({"illegible_biometrics"}, True)

    def test_fuzzy_short_prefix(self, snapfix_on):
        # 'resc' (clipped rescinded_denial) and 'ifle' (clipped AND
        # misread illegible_biometrics) -- the reference's two motivating
        # reads
        assert vocab.parse_flags("resc", flag_context=True) == \
            ({"rescinded_denial"}, True)
        assert vocab.parse_flags("ifle", flag_context=True) == \
            ({"illegible_biometrics"}, True)

    def test_flag_off_is_inert(self, snapfix_off):
        assert vocab.parse_flags("resc", flag_context=True) == (set(), False)

    def test_free_text_keeps_strict_rule(self, snapfix_on):
        # unlicensed callers never get the truncation decode: a bare
        # 'sponsor' from a Sponsor ID label must not invent a flag
        assert vocab.parse_flags("sponsor") == (set(), False)
        assert vocab.parse_flags("resc") == (set(), False)

    def test_too_short_rejected(self, snapfix_on):
        assert vocab.parse_flags("re", flag_context=True) == (set(), False)

    def test_near_tie_rejected(self, snapfix_on, monkeypatch):
        monkeypatch.setattr(vocab, "_FLAG_CANON", {
            "flag_one_x": "flagonex", "flag_one_y": "flagoney"})
        assert vocab._truncated_flag("flagone") is None

    def test_full_reads_and_none_unchanged(self, snapfix_on):
        assert vocab.parse_flags("rescinded_denial", flag_context=True) == \
            ({"rescinded_denial"}, True)
        assert vocab.parse_flags("none", flag_context=True) == (set(), True)

    def test_anchored_flags_row_is_the_licensed_site(self, snapfix_on):
        page = Page(index=0, kind=PageKind.SCAN)
        page.lines = [Line(text="Observed flags: resc", page_index=0,
                           source=Source.OCR, conf=0.9)]
        _, flag_cands, _ = fields.collect_candidates([page], CASE_ID)
        assert len(flag_cands) == 1
        assert set(flag_cands[0].flags) == {"rescinded_denial"}
        assert flag_cands[0].parsed_ok is True

    def test_anchored_flags_row_inert_flag_off(self, snapfix_off):
        page = Page(index=0, kind=PageKind.SCAN)
        page.lines = [Line(text="Observed flags: resc", page_index=0,
                           source=Source.OCR, conf=0.9)]
        _, flag_cands, _ = fields.collect_candidates([page], CASE_ID)
        assert len(flag_cands) == 1
        assert flag_cands[0].parsed_ok is False
        assert not flag_cands[0].flags


# --------------------- feature 3: cross-page per-digit sponsor vote


def _scan_page(idx, texts):
    page = Page(index=idx, kind=PageKind.SCAN)
    page.lines = [Line(text=t, page_index=idx, source=Source.OCR, conf=0.8)
                  for t in texts]
    return page


class TestSponsorDigitVote:
    def test_majority_decodes_fused_digits(self):
        pages = [
            _scan_page(0, ["Sponsor ID: SPH 4732"]),   # mangled N
            _scan_page(1, ["| SPN-4Z32 |"]),           # Z->2, one garble
            _scan_page(2, ["SPN. 47E2"]),              # E unrepaired
        ]
        assert fields.sponsor_digit_vote(pages, CASE_ID) == "SPN-4732"

    def test_needs_two_distinct_pages(self):
        pages = [_scan_page(0, ["Sponsor ID: SPN 4Z32", "SPN-4732"])]
        assert fields.sponsor_digit_vote(pages, CASE_ID) is None

    def test_tied_position_abstains(self):
        pages = [_scan_page(0, ["SPN-4732"]), _scan_page(1, ["SPN-4832"])]
        assert fields.sponsor_digit_vote(pages, CASE_ID) is None

    def test_non_digit_majority_abstains(self):
        pages = [_scan_page(0, ["SPN-47E2"]), _scan_page(1, ["SPN-47E2"])]
        assert fields.sponsor_digit_vote(pages, CASE_ID) is None

    def test_outside_cluster_read_abstains(self):
        # a second mention (decoy bait / other id): no winner to anchor on
        pages = [
            _scan_page(0, ["SPN-4732"]), _scan_page(1, ["SPN-4Z32"]),
            _scan_page(2, ["SPN-9911"]),
        ]
        assert fields.sponsor_digit_vote(pages, CASE_ID) is None

    def test_revoked_result_abstains(self):
        pages = [_scan_page(0, ["SPN 4O4O"]), _scan_page(1, ["SPN-4040"])]
        guarded = frozenset({"SPN-4040"})
        assert fields.sponsor_digit_vote(pages, CASE_ID, guarded) is None

    def test_revoked_cluster_member_abstains(self):
        pages = [_scan_page(0, ["SPN-4040"]), _scan_page(1, ["SPN-4041"]),
                 _scan_page(2, ["SPN-4041"])]
        guarded = frozenset({"SPN-4040"})
        assert fields.sponsor_digit_vote(pages, CASE_ID, guarded) is None

    def test_foreign_pages_never_vote(self):
        from mib_pipeline.model import TextSpan

        foreign = _scan_page(1, ["SPN-4Z32"])
        foreign.visible_spans = [TextSpan(
            text="Packet MIB-000999 / page 2", bbox=(0, 0, 1, 1), size=6.0,
            color=(0, 0, 0), opacity=1.0, render_type=0, font="helv")]
        pages = [_scan_page(0, ["SPN-4732"]), foreign]
        assert fields.sponsor_digit_vote(pages, CASE_ID) is None

    def test_prose_needs_digit_mass(self):
        # an accidental prefix hit on prose carries no repaired digits
        pages = [_scan_page(0, ["SPN abcd"]), _scan_page(1, ["SPN abcd"])]
        assert fields.sponsor_digit_vote(pages, CASE_ID) is None


# --------------------------------------- pipeline wiring (real PDF, e2e)


def _make_scan_pdf(tmp_path, page_specs, case_id=CASE_ID):
    """Synthetic packet whose pages are full-page raster images (SCAN
    kind) with the genuine vector footer (pattern from test_userwords)."""
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


# No readable sponsor row: with a non-DIP visa the sponsor stays a missing
# decision input, so the vote hook's fill-only precondition holds.
INTAKE_SPEC = [
    "FORM I-8090: Extraterrestrial Work Authorization Intake",
    f"Case ID: {CASE_ID}",
    "Applicant: Solul Zamora",
    "Species Code: ORION_GRAYS",
    "Home World: Proxima-b",
    "Visa Class: MED-3",
    "Arrival Date: 2026-05-04",
    "Declared Purpose: research",
    "Fee Status: paid",
]


def _quiet_escalation(monkeypatch):
    """Silence the heavy under-determined engines irrelevant to the hook
    under test (their own suites cover them)."""
    from mib_pipeline import crnn, ocr

    monkeypatch.setattr(ocr, "rapid_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "escalation_lines", lambda *a, **k: [])
    monkeypatch.setattr(ocr, "weld_sponsor_lines", lambda *a, **k: [])
    monkeypatch.setattr(crnn, "crnn_lines", lambda *a, **k: [])


class TestIntegrationPipeline:
    def test_flag_off_never_calls_the_vote(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIB_SNAPFIX", raising=False)
        _quiet_escalation(monkeypatch)
        calls = []
        monkeypatch.setattr(fields, "sponsor_digit_vote",
                            lambda *a, **k: calls.append(1) or None)
        row = pipeline.process_pdf(_make_scan_pdf(tmp_path, [INTAKE_SPEC]))
        assert calls == []                        # inert: hook never taken
        assert row["sponsor_id"] == "SPN-0000"    # writer default

    def test_flag_on_fills_unread_sponsor(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIB_SNAPFIX", "1")
        _quiet_escalation(monkeypatch)
        seen = {}

        def fake_vote(pages, case_id, guarded=frozenset()):
            seen["case_id"] = case_id
            seen["guarded"] = set(guarded)
            return "SPN-6712"

        monkeypatch.setattr(fields, "sponsor_digit_vote", fake_vote)
        row = pipeline.process_pdf(_make_scan_pdf(tmp_path, [INTAKE_SPEC]))
        assert row["sponsor_id"] == "SPN-6712"
        assert seen["case_id"] == CASE_ID
        from mib_pipeline import policy

        assert seen["guarded"] == set(policy.REVOKED_SPONSORS)

    def test_flag_on_vote_abstains_keeps_default(self, tmp_path,
                                                 monkeypatch):
        monkeypatch.setenv("MIB_SNAPFIX", "1")
        _quiet_escalation(monkeypatch)
        monkeypatch.setattr(fields, "sponsor_digit_vote",
                            lambda *a, **k: None)
        row = pipeline.process_pdf(_make_scan_pdf(tmp_path, [INTAKE_SPEC]))
        assert row["sponsor_id"] == "SPN-0000"
