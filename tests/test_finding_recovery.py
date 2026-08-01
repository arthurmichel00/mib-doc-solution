"""Fuzzy whole-phrase Finding/Reason recovery (backlog item A1).

Covers the two recovery paths added for garbled Manual Adjudicator Notes:
whole-phrase "FINDING <LABEL>" matching (partial ratio >= 78, >= 3 over the
runner-up label) and reason-template matching (>= 85, >= 10 margin) feeding
the existing stamp+Reason two-signal path. The reason template NEVER
supplies a Finding label on its own: 2 of the 18 corpus templates are
organizer-seeded multi-label traps (research/12-package-b-spec.md).
"""
from __future__ import annotations

from mib_pipeline import fields
from mib_pipeline.model import Line, Page, PageKind, Source

NOTE_HEADER = "Manual Adjudicator Note"


def _page(texts: list[str], index: int = 0, conf: float = 0.8,
          source: Source = Source.OCR, dark: bool = True,
          tier1_ok: bool = True) -> Page:
    lines = [
        Line(text=t, page_index=index, source=source, conf=conf,
             dark=dark, tier1_ok=tier1_ok)
        for t in texts
    ]
    return Page(index=index, kind=PageKind.SCAN, lines=lines)


def _findings(texts: list[str], **page_kw) -> list[fields.Finding]:
    page = _page(texts, **page_kw)
    _, _, findings = fields.collect_candidates([page], "MIB-000123")
    return findings


class TestExactPathUnchanged:
    def test_exact_finding_line_still_parses(self):
        found = _findings([NOTE_HEADER,
                           "Finding: DENIED. Reason: Mandatory fee unpaid."])
        assert [f.label for f in found] == ["DENIED"]
        assert "Mandatory fee unpaid" in found[0].reason

    def test_stamp_plus_clean_reason_still_fires(self):
        found = _findings([NOTE_HEADER, "APPROVED",
                           "Reason: Clean or exception-qualified packet."])
        assert [f.label for f in found] == ["APPROVED"]


class TestFuzzyWholePhrase:
    def test_two_error_finding_read_recovers(self):
        # "Fnding: DEN1ED." defeats the regex (finding misspelled) but is a
        # 2-error read of the whole phrase: 86.7 vs runner-up 73.3.
        found = _findings([NOTE_HEADER, "Fnding: DEN1ED.",
                           "Reason: Mandatory fee unpaid."])
        assert [f.label for f in found] == ["DENIED"]
        assert "Mandatory fee unpaid" in found[0].reason

    def test_two_error_approved_read_recovers(self):
        found = _findings([NOTE_HEADER, "PINDING: APPR0VED"])
        assert [f.label for f in found] == ["APPROVED"]

    def test_digit_confused_read_survives_fast_gate(self):
        # 1<->I twice destroys every raw 4-gram of "FINDING:"; the
        # confusion-normalized gate must keep the line for scoring.
        found = _findings([NOTE_HEADER, "F1ND1NG: DEN1ED."])
        assert [f.label for f in found] == ["DENIED"]

    def test_below_threshold_abstains(self):
        # The heavily damaged read from the kirtandesai memo scores 69.2
        # for DENIED with a 0.7 margin over NEEDS_REVIEW — an honest
        # partial ratio cannot separate the labels, so it must abstain.
        assert _findings([NOTE_HEADER, "Pining: GED 7"]) == []

    def test_margin_tie_abstains(self):
        # 80.0 DENIED vs 77.8 NEEDS_REVIEW: above threshold, under margin.
        assert _findings([NOTE_HEADER, "FINDING: NEEDED"]) == []

    def test_truncated_longer_label_abstains(self):
        # Real corpus read (MIB-000660 p1): a truncated NEEDS_REVIEW line
        # scores 80.0 for DENIED under whole-string comparison against its
        # own longer phrase. The symmetric runner-up margin must catch
        # that the line is a clean prefix of the competitor and abstain.
        assert _findings([NOTE_HEADER, "Finding: NEEDS_F N"]) == []

    def test_conflicting_fuzzy_lines_abstain(self):
        # Two qualifying lines naming different labels: a genuine note
        # carries one finding, so the page is under-determined.
        assert _findings([NOTE_HEADER, "FNDING: DENVED",
                          "PINDING: APPR0VED"]) == []

    def test_sub_trusted_line_never_mints(self):
        assert _findings([NOTE_HEADER, "Fnding: DEN1ED."],
                         tier1_ok=False) == []

    def test_light_ink_line_never_mints(self):
        assert _findings([NOTE_HEADER, "Fnding: DEN1ED."],
                         source=Source.DIGITAL, dark=False) == []

    def test_foreign_template_page_ineligible(self):
        # A page showing another template's distinctive label must never
        # have a finding honored, fuzzy or otherwise.
        assert _findings(["Declared Purpose: research",
                          "Fnding: DEN1ED."]) == []


class TestReasonTemplateTwoSignal:
    def test_stamp_plus_two_error_reason_fires(self):
        # The documented blocker (LOOP_STATE miss audit): a 2-error read of
        # the "Reason" token ("Reosan", weighted distance 2.0 > 1.2) defeats
        # the first-token test, but the line is a high-scoring read of a
        # closed reason template. With the 20pt stamp surviving, the
        # two-signal path must now fire.
        found = _findings([NOTE_HEADER, "APPROVED",
                           "Reosan: Clean or exception-quailfied packet."])
        assert [f.label for f in found] == ["APPROVED"]

    def test_stamp_plus_junk_line_abstains(self):
        assert _findings([NOTE_HEADER, "APPROVED",
                          "quarterly revenue for the office"]) == []

    def test_stamp_plus_template_tail_fragment_abstains(self):
        # Real corpus read (MIB-000023 p0): a garbled field label covers
        # only the tail of "Embargo home world:". The template must fit
        # inside the line, so a suffix fragment never counts as a Reason.
        assert _findings([NOTE_HEADER, "DENIED", "iF Home World:"]) == []

    def test_reason_flags_still_mined_via_template_path(self):
        page = _page([NOTE_HEADER, "DENIED",
                      "Reosan: Disqualifying risk flag: biohazard_red."])
        cands, flag_cands, found = fields.collect_candidates(
            [page], "MIB-000123")
        ev = fields.reconcile(cands, flag_cands, found)
        assert ev.finding is not None and ev.finding.label == "DENIED"
        assert "biohazard_red" in ev.flags


class TestTrapGuards:
    def test_trap_reason_template_alone_never_mints(self):
        # 'Revoked sponsor' maps DENIED 6 / NEEDS_REVIEW 2 in the corpus —
        # an organizer-seeded trap. Without a stamp or Finding line the
        # page must stay finding-less no matter how clean the read.
        assert _findings([NOTE_HEADER,
                          "Reason: Revoked sponsor: SPN-4040."]) == []

    def test_multilabel_trap_template_alone_never_mints(self):
        assert _findings([
            NOTE_HEADER,
            "Reason: Review-only risk flag present: illegible_biometrics.",
        ]) == []

    def test_clean_template_alone_never_mints(self, monkeypatch):
        # Even the single-label 'Clean or exception-qualified packet.'
        # template implies nothing without a stamp or Finding line — in the
        # DEFAULT configuration. Under MIB_REASON_ADJ=1 this exact line is
        # the designed rescue (test_reason_adjudication.py; spec
        # 2026-07-29-pure-template-reason-adjudication.md), so this test
        # pins the flag off.
        monkeypatch.delenv("MIB_REASON_ADJ", raising=False)
        assert _findings([NOTE_HEADER,
                          "Reason: Clean or exception-qualified packet."]) == []

    def test_decoy_stamp_word_alone_no_finding(self):
        # Red decoy stamps print on non-note templates with no Reason
        # line; a standalone stamp word must not become a finding, and
        # the whole-phrase matcher's coverage guard must reject it too.
        assert _findings(["DENIED"]) == []
