"""Truncation-prefix guard in fields._fuzzy_phrase_finding (497 latent bug).

Mechanism (root-caused on MIB-000497, note-route-diag 2026-07-31): damage
truncates the note's "Finding: NEEDS_REVIEW" to the prefix "NEEDS" plus
trailing junk. On the minting line "| Finding: NEEDS 7. | |" the junk chars
penalize the longer true label more than a wrong shorter one: FINDING:
DENIED scores 80.0 while FINDING: NEEDS_REVIEW collapses to 66.7, the 13.3
margin clears the 3.0 bar, and a spurious N0 DENIED mints at 0.97.

The guard re-scores every label on a junk-stripped label zone, where a
truncation-prefix regains its full symmetric-partial-ratio credit
("FINDING: NEEDS" scores NEEDS_REVIEW at 100). A winner that cannot also
win that scoring by the existing margin is truncation-ambiguous and the
line abstains, so policy falls through to the R-paths. Unambiguous reads
keep the exact pre-guard thresholds.
"""
from __future__ import annotations

import pytest

from mib_pipeline import fields, policy
from mib_pipeline.model import Line, Page, PageKind, Source

# The exact runtime line that minted the spurious DENIED on MIB-000497
# (container tesseract 5.3; 16/240 render cells reproduce it).
MINTING_LINE_497 = "| Finding: NEEDS 7. | |"


def _page(texts: list[str], index: int = 0, conf: float = 0.97,
          source: Source = Source.OCR, dark: bool = True,
          tier1_ok: bool = True) -> Page:
    lines = [
        Line(text=t, page_index=index, source=source, conf=conf,
             dark=dark, tier1_ok=tier1_ok)
        for t in texts
    ]
    return Page(index=index, kind=PageKind.SCAN, lines=lines)


class Test497Regression:
    def test_exact_minting_line_abstains(self):
        assert fields._fuzzy_phrase_finding(_page([MINTING_LINE_497])) is None

    def test_collect_candidates_mints_no_finding(self):
        _, _, findings = fields.collect_candidates(
            [_page([MINTING_LINE_497])], "MIB-000497")
        assert findings == []

    def test_case_falls_through_to_r8(self):
        """No finding -> policy under-determined R-path (gold NEEDS_REVIEW)."""
        cands, flags, findings = fields.collect_candidates(
            [_page([MINTING_LINE_497])], "MIB-000497")
        ev = fields.reconcile(cands, flags, findings)
        assert ev.finding is None and not ev.finding_conflict
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R8_fee_unread")

    @pytest.mark.parametrize("text", [
        # sibling reads observed across the 240-cell render sweep
        # (hunt_497_p1.txt); all are truncation reads of NEEDS_REVIEW and
        # none may mint any label.
        "Finding: NEEDS",
        "Finding NEEDS",
        ": Finding: NEEDS",
        "Finding: NEEDS coy",
        "Finding: NEEDS ~",
        'Finding: NEEDS "7"',
        "Finding: NEEDS | »",
        "Finding: WEEDS",
    ])
    def test_truncation_family_never_mints(self, text):
        assert fields._fuzzy_phrase_finding(_page([text])) is None


class TestZoneHelper:
    def test_497_zone_is_the_stripped_label_region(self):
        upper = " ".join(MINTING_LINE_497.split()).upper()
        # DENIED's aligned window on this line is upper[0:15]; the helper
        # extends through the cut "NEED"->"NEEDS" run and strips the edges.
        assert fields._truncation_guard_zone(upper, 15, 15) == "FINDING: NEEDS"

    def test_clean_line_zone_is_the_phrase_itself(self):
        upper = "FINDING: DENIED."
        zone = fields._truncation_guard_zone(upper, 15, 15)
        assert zone == "FINDING: DENIED"

    def test_empty_zone_never_vetoes(self):
        assert not fields._truncation_ambiguous("| |", 3, "DENIED",
                                                "FINDING: DENIED")


class TestUnambiguousReadsUnchanged:
    """The guard must not move any bar for reads that are not
    truncation-ambiguous — same inputs, same mints, same confidences."""

    @pytest.mark.parametrize("text,label", [
        ("Finding: DENIED.", "DENIED"),
        ("FNDING: DEN1ED.", "DENIED"),
        ("F1NDING: DEN1ED.", "DENIED"),
        ("FNDING: APPROVED,", "APPROVED"),
        ("Finding: NEEDS_REVIEW.", "NEEDS_REVIEW"),
        ("| Finding: NEEDS_REVIEW |", "NEEDS_REVIEW"),
    ])
    def test_still_mints_same_label(self, text, label):
        found = fields._fuzzy_phrase_finding(_page([text]))
        assert found is not None
        assert found.label == label
        assert found.conf == 0.97

    def test_guard_reports_unambiguous_for_clean_reads(self):
        import re

        from mib_pipeline import vocab

        for text, label in [("FINDING: DENIED.", "DENIED"),
                            ("FINDING: NEEDS_REVIEW.", "NEEDS_REVIEW"),
                            ("FINDING: APPROVED.", "APPROVED")]:
            phrase = fields._FINDING_PHRASES[label]
            _, end = vocab.partial_ratio_alignment(phrase, text)
            assert not fields._truncation_ambiguous(text, end, label, phrase)

    def test_ambiguous_line_does_not_poison_a_clean_sibling(self):
        """The veto is per-line: a junk truncation read abstains while a
        clean read of the same note still mints (both reads come from the
        same pooled OCR ladder)."""
        found = fields._fuzzy_phrase_finding(
            _page([MINTING_LINE_497, "Finding: NEEDS_REVIEW."]))
        assert found is not None
        assert found.label == "NEEDS_REVIEW"
