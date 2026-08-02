"""Joint-grammar name decode behind MIB_JOINTNAME=1 (ctc-branch).

vocab.correct_name_joint scores whole candidate names over the attested
144-token grammar so a clean token carries its garbled partner, instead
of both tokens snapping (or failing) independently. fields._correct only
invokes it AFTER vocab.correct_name failed to produce a grammar-legal
two-token name, so every read the shipped matcher already resolves is
byte-identical with the flag on. Flag unset = dead code.
"""
from __future__ import annotations

import pytest

from mib_pipeline import fields, vocab

# ------------------------------------------------------ attested grammar


class TestAttestedGrammar:
    def test_144_tokens_all_inside_wide_lexicon(self):
        assert len(vocab.NAME_TOKENS_ATTESTED) == 144
        assert len(set(vocab.NAME_TOKENS_ATTESTED)) == 144
        assert set(vocab.NAME_TOKENS_ATTESTED) <= set(vocab.NAME_TOKENS)

    def test_mira_takes_the_a_link_everyone_else_the_bare_form(self):
        attested = set(vocab.NAME_TOKENS_ATTESTED)
        assert "Miradane" in attested and "Mirdane" not in attested
        assert "Nexdane" in attested and "Nexadane" not in attested
        assert "Zazarn" in attested and "Zaazarn" not in attested

    def test_is_grammar_name(self):
        assert vocab.is_grammar_name("Solul Zamora")
        assert not vocab.is_grammar_name("Solul")
        assert not vocab.is_grammar_name("Qomax Qorquell")   # phantom-ish
        assert not vocab.is_grammar_name(None)
        assert not vocab.is_grammar_name("")


# --------------------------------------------------------- joint decoder


class TestJointDecode:
    def test_clean_token_carries_garbled_partner(self):
        # 'Ccrul' alone would never reach 'Qorul'; the joint read does.
        assert vocab.correct_name_joint("Wirequail Ccrul") \
            == "Miraquell Qorul"

    def test_recovers_dropped_glyphs(self):
        assert vocab.correct_name_joint("Iovara Miravara") \
            == "Ixovara Miravara"

    def test_ambiguous_read_abstains_on_margin(self):
        # 'lozan' ties between 'Luzarn' and 'Ixozarn' under the weighted
        # metric: a coin-flip mint must abstain.
        assert vocab.correct_name_joint("Aririx lozan") is None

    def test_debris_read_abstains_on_floor(self):
        assert vocab.correct_name_joint("[NAME CUT") is None

    def test_non_two_token_reads_abstain(self):
        assert vocab.correct_name_joint("Solul") is None
        assert vocab.correct_name_joint("") is None
        assert vocab.correct_name_joint("PASSPORT IMAGE Solul") is None

    def test_output_is_always_grammar_legal(self):
        got = vocab.correct_name_joint("Wirequail Ccrul")
        assert vocab.is_grammar_name(got)

    def test_exact_attested_name_decodes_to_itself(self):
        assert vocab.correct_name_joint("Miraquell Qorul") \
            == "Miraquell Qorul"


# ------------------------------------------------------------- flag gate


class TestFlagGate:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MIB_JOINTNAME", raising=False)
        assert fields._jointname_enabled() is False

    def test_env_controls(self, monkeypatch):
        monkeypatch.setenv("MIB_JOINTNAME", "1")
        assert fields._jointname_enabled() is True
        monkeypatch.setenv("MIB_JOINTNAME", "0")
        assert fields._jointname_enabled() is False


# ------------------------------------------------------- fields wiring


class TestFieldsWiring:
    def test_flag_off_is_byte_identical(self, monkeypatch):
        monkeypatch.delenv("MIB_JOINTNAME", raising=False)
        raw = "Wirequail Ccrul"
        assert fields._correct("applicant_name", raw) \
            == vocab.correct_name(raw)

    def test_flag_on_decodes_garbled_read(self, monkeypatch):
        monkeypatch.setenv("MIB_JOINTNAME", "1")
        base = vocab.correct_name("Wirequail Ccrul")
        assert not vocab.is_grammar_name(base)       # base matcher fails
        assert fields._correct("applicant_name", "Wirequail Ccrul") \
            == "Miraquell Qorul"

    def test_flag_on_never_touches_grammar_valid_base_reads(self,
                                                            monkeypatch):
        monkeypatch.setenv("MIB_JOINTNAME", "1")

        def boom(raw):
            raise AssertionError("joint decoder must not run on clean reads")

        monkeypatch.setattr(vocab, "correct_name_joint", boom)
        # base matcher resolves this to a legal name (2 lexicon tokens)
        assert fields._correct("applicant_name", "Solul Zamora") \
            == "Solul Zamora"

    def test_flag_on_keeps_base_result_when_joint_abstains(self,
                                                           monkeypatch):
        monkeypatch.setenv("MIB_JOINTNAME", "1")
        raw = "Aririx lozan"
        assert vocab.correct_name_joint(raw) is None  # margin abstain
        assert fields._correct("applicant_name", raw) \
            == vocab.correct_name(raw)

    def test_other_fields_untouched(self, monkeypatch):
        monkeypatch.setenv("MIB_JOINTNAME", "1")
        assert fields._correct("home_world", "Proxima-b") == "Proxima-b"
        assert fields._correct("species_code", "ORION_GRAYS") \
            == "ORION_GRAYS"


# ------------------------------------------------- pruning is joint-only


class TestBaseMatcherUntouched:
    def test_wide_lexicon_still_serves_the_base_matcher(self):
        # the phantom-token prune lives in the joint decoder ONLY: the
        # shipped per-token matcher keeps the full cartesian lexicon, so
        # its behavior is unchanged whether or not the flag is set
        assert len(vocab.NAME_TOKENS) == 288
        assert vocab.correct_name("Solul Zamora") == "Solul Zamora"
