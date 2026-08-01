"""Pure-template Reason adjudication behind MIB_REASON_ADJ=1 (Decision 2).

Spec: docs/superpowers/specs/2026-07-29-pure-template-reason-adjudication.md.
A Reason line matching one of SEVEN single-label templates (fresh-mined
2026-07-29: 162 digital Finding+Reason lines, note==gold 162/162) may
adjudicate a case whose field policy is under-determined. The two
organizer-seeded traps and every tail-bearing or low-support template are
excluded by content. With the flag unset, behavior is identical to the
shipped code.
"""
from __future__ import annotations

import pytest

from mib_pipeline import calibration, decision, fields, policy
from mib_pipeline.fields import CaseEvidence, Finding
from mib_pipeline.model import Line, Page, PageKind, Source

NOTE_HEADER = "Manual Adjudicator Note"


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MIB_REASON_ADJ", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("MIB_REASON_ADJ", raising=False)


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
    _, _, found = fields.collect_candidates([page], "MIB-000123")
    return found


def _template_findings(texts: list[str], **page_kw) -> list[fields.Finding]:
    return [f for f in _findings(texts, **page_kw) if f.template_only]


def _clean_evidence(**overrides) -> CaseEvidence:
    """Evidence that would reach R11 unless a field is knocked out."""
    ev = CaseEvidence()
    values = {
        "applicant_name": "Solul Zamora", "species_code": "ORION_GRAYS",
        "home_world": "Proxima-b", "visa_class": "XW-1",
        "sponsor_id": "SPN-1234", "arrival_date": "2026-05-04",
        "declared_purpose": "research", "fee_status": "paid",
    }
    values.update({k: v for k, v in overrides.items() if k in values})
    for fld, value in values.items():
        ev.values[fld] = value
        ev.known[fld] = value is not None
        ev.conf[fld] = 0.9 if value is not None else 0.0
    ev.flags_known = True
    ev.arrival_on_intake = overrides.get("arrival_on_intake", True)
    return ev


def _template_finding(label: str, template: str, conf: float = 0.8) -> Finding:
    return Finding(label=label, reason=f"Reason: {template}", source=Source.OCR,
                   conf=conf, template_only=True, template=template)


# ---------------------------------------------------------------- fields


class TestTemplateMinting:
    def test_clean_template_alone_mints_template_finding(self, flag_on):
        found = _template_findings(
            [NOTE_HEADER, "Reason: Clean or exception-qualified packet."])
        assert [f.label for f in found] == ["APPROVED"]
        assert found[0].template == "Clean or exception-qualified packet."

    def test_802_real_divblur_read_mints(self, flag_on):
        # The recorded blocker read (LOOP_STATE miss audit / threshold lab:
        # score 97.2, margin 38.4).
        found = _template_findings(
            [NOTE_HEADER, "Reassr: Clean or exception-quailfied packet."],
            conf=0.72)
        assert [f.label for f in found] == ["APPROVED"]

    def test_every_single_label_template_mints_its_mined_label(self, flag_on):
        for template, label in fields._REASON_ADJ_LABELS.items():
            found = _template_findings([NOTE_HEADER, f"Reason: {template}"])
            assert [f.label for f in found] == [label], template

    def test_full_finding_line_preempts_template_path(self, flag_on):
        found = _findings([NOTE_HEADER,
                           "Finding: DENIED. Reason: Mandatory fee unpaid."])
        assert [f.template_only for f in found] == [False]

    def test_stamp_two_signal_path_preempts_template_path(self, flag_on):
        found = _findings([NOTE_HEADER, "APPROVED",
                           "Reason: Clean or exception-qualified packet."])
        assert [f.template_only for f in found] == [False]

    def test_flag_off_never_mints(self, flag_off):
        assert _findings(
            [NOTE_HEADER, "Reason: Clean or exception-qualified packet."]) == []


class TestTrapAndExclusionGuards:
    # The two organizer-seeded traps, each by name, plus every excluded
    # template: none may ever adjudicate alone, however clean the read.
    @pytest.mark.parametrize("line", [
        "Reason: Review-only risk flag present: illegible_biometrics.",
        "Reason: Review-only risk flag present: illegible_biometrics. REVIEW",
        "Reason: Review-only risk flag present: identity_conflict. REVIEW",
        "Reason: Review-only risk flag present: rescinded_denial. REVIEW",
    ])
    def test_trap_review_only_never_adjudicates(self, flag_on, line):
        assert _findings([NOTE_HEADER, line]) == []

    @pytest.mark.parametrize("line", [
        "Reason: Revoked sponsor: SPN-0007.",
        "Reason: Revoked sponsor: SPN-0139. REVIEW",
        "Reason: Revoked sponsor: SPN-2718.",
        "Reason: Revoked sponsor: SPN-4040.",
    ])
    def test_trap_revoked_sponsor_never_adjudicates(self, flag_on, line):
        assert _findings([NOTE_HEADER, line]) == []

    @pytest.mark.parametrize("line", [
        # tail-bearing (deny-ness depends on the tail / DIP-1 exemptions)
        "Reason: Disqualifying risk flag: biohazard_red.",
        "Reason: Disqualifying risk flag: planetary_embargo.",
        "Reason: Embargo home world: Wolf-1061c.",
        # support < 5 in the mined inventory
        "Reason: Ambiguous packet.",
        "Reason: Fee status unknown.",
    ])
    def test_excluded_templates_never_adjudicate(self, flag_on, line):
        assert _findings([NOTE_HEADER, line]) == []

    def test_two_error_trap_read_never_drifts_into_adj_set(self, flag_on):
        # t_trapfuzz: the trap stem must keep winning the scoring; the lab
        # corruption ceiling for adjudicating templates is 44.4 (bar: 90).
        assert _findings([
            NOTE_HEADER,
            "Reason: Reviel-only risk fleg present: rescinded_denial.",
        ]) == []


class TestMatchStrictness:
    def test_truncated_template_below_cover_abstains(self, flag_on):
        assert _findings([NOTE_HEADER, "Reason: Clean or exception-qual"]) == []

    def test_heavily_damaged_read_abstains(self, flag_on):
        assert _findings([NOTE_HEADER, "Rsn: Cln r xcptn-qlfd pckt"]) == []

    def test_junk_prose_abstains(self, flag_on):
        assert _findings([NOTE_HEADER,
                          "quarterly revenue for the office"]) == []

    def test_two_templates_different_labels_abstain(self, flag_on):
        assert _findings([
            NOTE_HEADER,
            "Reason: Clean or exception-qualified packet.",
            "Reason: Mandatory fee unpaid.",
        ]) == []

    def test_same_label_twice_takes_best_conf(self, flag_on):
        page = _page([NOTE_HEADER,
                      "Reason: Clean or exception-qualified packet.",
                      "Reason: Approval supported by surviving visible "
                      "evidence and exception notes."])
        _, _, found = fields.collect_candidates([page], "MIB-000123")
        assert len(found) == 1 and found[0].label == "APPROVED"

    def test_light_ink_line_never_mints(self, flag_on):
        assert _findings(
            [NOTE_HEADER, "Reason: Clean or exception-qualified packet."],
            source=Source.DIGITAL, dark=False) == []

    def test_sub_trusted_line_never_mints(self, flag_on):
        assert _findings(
            [NOTE_HEADER, "Reason: Clean or exception-qualified packet."],
            tier1_ok=False) == []

    def test_typed_foreign_page_never_mints(self, flag_on):
        # A page carrying another template's distinctive label is not
        # finding-eligible; the template path honors the same gate.
        assert _findings(["Declared Purpose: research",
                          "Reason: Clean or exception-qualified packet."]) == []


class TestReconcile:
    def test_template_finding_lands_on_evidence(self, flag_on):
        page = _page([NOTE_HEADER,
                      "Reason: Clean or exception-qualified packet."])
        cands, flag_cands, found = fields.collect_candidates([page], "MIB-000123")
        ev = fields.reconcile(cands, flag_cands, found)
        assert ev.finding is None and not ev.finding_conflict
        assert ev.template_finding is not None
        assert ev.template_finding.label == "APPROVED"

    def test_template_finding_never_mines_reason_fields(self, flag_on):
        # _mine_reason_fields would set visa_class=TRANSIT-7 from this
        # sentence under a full N0 finding; a template-only finding mints
        # the label and nothing else. (The fee-unpaid template is not used
        # here: the ordinary _fee_word_rescue harvests fee words from any
        # OCR line, flag-independent.)
        page = _page([NOTE_HEADER,
                      "Reason: Transit class cannot authorize declared work."])
        cands, flag_cands, found = fields.collect_candidates([page], "MIB-000123")
        ev = fields.reconcile(cands, flag_cands, found)
        assert ev.template_finding is not None
        assert ev.values.get("visa_class") is None  # label only, no fields

    def test_full_finding_keeps_n0_semantics(self, flag_on):
        full = Finding(label="DENIED", reason="Mandatory fee unpaid.",
                       source=Source.DIGITAL, conf=0.99)
        templ = _template_finding("APPROVED",
                                  "Clean or exception-qualified packet.")
        ev = fields.reconcile([], [], [full, templ])
        assert ev.finding is not None and ev.finding.label == "DENIED"
        assert not ev.finding_conflict  # template findings are not N0 votes


# ---------------------------------------------------------------- policy


class TestPolicyPrecedence:
    def test_rescues_underdetermined_case_to_approved(self, flag_on):
        ev = _clean_evidence(visa_class=None)  # -> R12_visa_unread
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("APPROVED", "N1_reason_approved")

    def test_rescues_underdetermined_case_to_denied(self, flag_on):
        ev = _clean_evidence(fee_status=None)  # -> R8_fee_unread
        ev.template_finding = _template_finding(
            "DENIED", "Mandatory fee unpaid.")
        assert policy.adjudicate(ev) == ("DENIED", "N1_reason_denied")

    def test_rescues_underdetermined_case_to_needs_review(self, flag_on):
        ev = _clean_evidence(fee_status=None)
        ev.template_finding = _template_finding(
            "NEEDS_REVIEW", "Packet contains damaged or contradictory "
                            "visible evidence.")
        assert policy.adjudicate(ev) == \
            ("NEEDS_REVIEW", "N1_reason_needs_review")

    def test_n0_finding_always_wins(self, flag_on):
        ev = _clean_evidence(fee_status=None)
        ev.finding = Finding(label="DENIED", reason="Mandatory fee unpaid.",
                             source=Source.DIGITAL, conf=0.99)
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("DENIED", "N0_note_denied")

    def test_n0_conflict_always_wins(self, flag_on):
        ev = _clean_evidence(fee_status=None)
        ev.finding_conflict = True
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "N0_note_conflict")

    @pytest.mark.parametrize("overrides,path", [
        ({"fee_status": "unpaid"}, "R3_fee_unpaid"),
        ({"visa_class": "TRANSIT-7"}, "R7_transit_visa"),
        ({"home_world": "TRAPPIST-1e"}, "R2_hard_embargo_world"),
        ({"fee_status": "unknown"}, "R8_fee_unknown"),
    ])
    def test_positive_decisions_never_overridden(self, flag_on, overrides, path):
        ev = _clean_evidence(**overrides)
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == (policy._adjudicate_core(ev)[0], path)

    def test_review_flags_never_overridden(self, flag_on):
        ev = _clean_evidence()
        ev.flags = {"identity_conflict"}
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R10_review_flags")

    def test_r11_approval_never_relabeled(self, flag_on):
        ev = _clean_evidence()
        ev.template_finding = _template_finding(
            "DENIED", "Denial supported by damaged registry evidence and "
                      "visible policy notes.")
        assert policy.adjudicate(ev) == ("APPROVED", "R11_default_approve")

    def test_low_conf_template_never_rescues(self, flag_on):
        ev = _clean_evidence(fee_status=None)
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.", conf=0.4)
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R8_fee_unread")


class TestFieldConflictVetoes:
    def test_approved_vetoed_by_revoked_sponsor_when_visa_unread(self, flag_on):
        ev = _clean_evidence(visa_class=None, sponsor_id="SPN-0007")
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R12_visa_unread")

    def test_approved_allowed_when_dip1_exemption_affirmed(self, flag_on):
        ev = _clean_evidence(visa_class="DIP-1", sponsor_id="SPN-0007",
                             fee_status=None)  # -> R8_fee_unread
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("APPROVED", "N1_reason_approved")

    def test_approved_vetoed_by_soft_embargo_when_visa_unread(self, flag_on):
        ev = _clean_evidence(visa_class=None, home_world="Wolf-1061c")
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R12_visa_unread")

    def test_approved_vetoed_by_visible_stale_arrival(self, flag_on):
        ev = _clean_evidence(visa_class=None, arrival_date="2025-11-01")
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R12_visa_unread")

    def test_denied_fee_template_vetoed_by_affirmative_paid(self, flag_on):
        ev = _clean_evidence(arrival_on_intake=False, arrival_date=None)
        ev.template_finding = _template_finding(
            "DENIED", "Mandatory fee unpaid.")
        assert policy.adjudicate(ev) == \
            ("NEEDS_REVIEW", "R9_arrival_not_visible")

    def test_denied_transit_template_vetoed_by_affirmative_visa(self, flag_on):
        ev = _clean_evidence(fee_status=None)  # visa reads XW-1
        ev.template_finding = _template_finding(
            "DENIED", "Transit class cannot authorize declared work.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R8_fee_unread")

    def test_denied_damaged_registry_template_fires_without_field_veto(
            self, flag_on):
        ev = _clean_evidence(fee_status=None)
        ev.template_finding = _template_finding(
            "DENIED", "Denial supported by damaged registry evidence and "
                      "visible policy notes.")
        assert policy.adjudicate(ev) == ("DENIED", "N1_reason_denied")


class TestFlagOffIdentity:
    def test_flag_off_ignores_template_finding(self, flag_off):
        ev = _clean_evidence(visa_class=None)
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R12_visa_unread")

    def test_flag_zero_is_disabled(self, monkeypatch):
        monkeypatch.setenv("MIB_REASON_ADJ", "0")
        ev = _clean_evidence(visa_class=None)
        ev.template_finding = _template_finding(
            "APPROVED", "Clean or exception-qualified packet.")
        assert policy.adjudicate(ev) == ("NEEDS_REVIEW", "R12_visa_unread")

    def test_adjudicate_equals_core_on_every_path_flag_off(self, flag_off):
        fixtures = [
            _clean_evidence(),
            _clean_evidence(fee_status=None),
            _clean_evidence(visa_class=None),
            _clean_evidence(fee_status="unpaid"),
            _clean_evidence(home_world="TRAPPIST-1e"),
            _clean_evidence(arrival_on_intake=False, arrival_date=None),
        ]
        for ev in fixtures:
            ev.template_finding = _template_finding(
                "APPROVED", "Clean or exception-qualified packet.")
            assert policy.adjudicate(ev) == policy._adjudicate_core(ev)


# ------------------------------------------------------------ calibration


class TestCalibration:
    def test_n1_paths_have_stats(self):
        for path in ("N1_reason_approved", "N1_reason_denied",
                     "N1_reason_needs_review"):
            assert path in calibration.PATH_STATS

    def test_n1_decisions_follow_the_minted_label(self):
        assert decision.decide("APPROVED", "N1_reason_approved")[0] == "APPROVED"
        assert decision.decide("DENIED", "N1_reason_denied")[0] == "DENIED"
        label, conf = decision.decide("NEEDS_REVIEW", "N1_reason_needs_review")
        # Confidence = the clamped calibrated path constant (0.97 prior;
        # the A9 in-container refit shrinks tiny-n N1 paths toward global).
        expected = calibration.clamp_confidence(
            calibration.PATH_STATS["N1_reason_needs_review"].accuracy)
        assert label == "NEEDS_REVIEW" and conf == pytest.approx(expected)
