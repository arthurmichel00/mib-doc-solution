"""Field harvesting and cross-page reconciliation.

Lines from every page (digital text layer or OCR) are parsed against known
label anchors into per-field candidates carrying provenance: the evidence
tier of the page's document type, the source (digital vs OCR) and a
confidence. Digital sponsor-attestation letters use prose instead of label
rows and get dedicated parsing. Candidates are reconciled per field by
majority-weighted voting (agreement across pages beats any single page,
matching observed generator behavior), with italic "Manual correction:"
lines overriding and Manual Adjudicator Note findings kept as tier-1
signals. Pages that belong to an archived adjacent applicant (foreign
Case ID such as MIB-000000) are dropped wholesale: they carry planted bait
values, including embargoed home worlds.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dc_field

from . import vocab
from .model import FieldCandidate, Line, Page, Source

# Document types by header line, with FIELD_MANUAL evidence tier
# (1 note > 2 intake > 3 biometric > 4 attestation > 5 registry).
# The fee receipt is the authoritative fee document; treated like intake.
_DOC_HEADERS: dict[str, tuple[str, float]] = {
    "Manual Adjudicator Note": ("note", 1.0),
    "FORM I-8090: Extraterrestrial Work Authorization Intake": ("intake", 2.0),
    "MIB Fee Receipt": ("fee_receipt", 2.5),
    "FORM B-13: Biometric Scan Slip": ("biometric", 3.0),
    "Sponsor Attestation Letter": ("attestation", 4.0),
    "Planetary Registry Extract": ("registry", 5.0),
}
_UNKNOWN_TIER = 5.0
_CORRECTION_TIER = 1.5

# Damaged scans drop whole words from headers ("FORM J-2080: Work
# Authorization inteka."), so each type also matches on a distinctive
# fragment, and both the head and the tail of the line are tried.
_DOC_ALIASES: list[tuple[str, str, float]] = [
    (header, doc, tier) for header, (doc, tier) in _DOC_HEADERS.items()
] + [
    ("Work Authorization Intake", "intake", 2.0),
    ("Biometric Scan Slip", "biometric", 3.0),
]

# Label anchors, canonical field -> printed labels (multi-word first).
_ANCHORS: list[tuple[str, str]] = [
    ("declared_purpose", "Declared Purpose"),
    ("applicant_name", "Registry Name"),
    ("species_code", "Species Code"),
    ("species_code", "Species Match"),
    ("home_world", "Home World"),
    ("visa_class", "Visa Class"),
    ("sponsor_id", "Sponsor ID"),
    ("arrival_date", "Arrival Date"),
    ("risk_flags", "Observed flags"),
    ("fee_status", "Fee Status"),
    ("registry_status", "Registry Status"),
    ("waiver_code", "Waiver Code"),
    ("case_id", "Case ID"),
    ("applicant_name", "Applicant"),
    ("declared_purpose", "Purpose"),
    ("fee_amount", "Amount"),
]
_ANCHOR_MAX_DIST = 0.36
_DATE_WORD_MAX_DIST = 0.34

_CORRECTION_RE = re.compile(r"manual\s+correction\b[:.]?\s*(.+)", re.IGNORECASE)
_CORRECTION_FIELDS: list[tuple[str, str]] = [
    ("applicant", "applicant_name"),
    ("sponsor", "sponsor_id"),
    ("species", "species_code"),
    ("home world", "home_world"),
    ("world", "home_world"),
    ("visa", "visa_class"),
    ("arrival", "arrival_date"),
    ("purpose", "declared_purpose"),
    ("fee", "fee_status"),
]
_FINDING_RE = re.compile(r"finding\b[:.]?\s*([A-Z_ ]+?)[.,]", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\$?\s*([0-9oOlI]{1,6})(?:[.,][0-9oO]{2})?\b")

# Digital attestation letters are prose, not label rows.
_ATTEST_SPONSOR_RE = re.compile(r"\bSPN[-–—._ ]?(\d{4})\b")
_ATTEST_NAME_RE = re.compile(r"attests\s+that\s+(.+?)\s+is\s+expected", re.IGNORECASE)
_ATTEST_PURPOSE_RE = re.compile(
    r"is\s+expected\s+on\s+Earth\s+for\s+(.+?)[.\n]", re.IGNORECASE)
_ATTEST_VISA_RE = re.compile(r"\bclass\s+(\S{3,12})\s+compliance", re.IGNORECASE)

_DECOY_MARKER = "adjacent applicant"
FOOTER_ID_RE = re.compile(r"Packet\s+(MIB-[0-9]{6})\s*/\s*page")

_ADJUDICATIONS = ["APPROVED", "DENIED", "NEEDS_REVIEW"]

# Whole-phrase recovery of garbled note lines (idea from the MIT-licensed
# tylergibbs1 and kirtandesai public solutions; see ATTRIBUTION.md).
# Scores are partial-ratio points (0..100); margins over the runner-up.
_FINDING_PHRASES = {label: f"FINDING: {label}" for label in _ADJUDICATIONS}
_PHRASE_MIN_SCORE, _PHRASE_MIN_MARGIN = 78.0, 3.0
_REASON_MIN_SCORE, _REASON_MIN_MARGIN = 85.0, 10.0
# The line must be able to cover most of the phrase it claims to be: a
# standalone stamp word ("DENIED") is a perfect substring of its finding
# phrase and must never qualify on its own.
_PHRASE_MIN_COVER = 0.7
_REASON_TEMPLATES_UC = [t.upper() for t in vocab.REASON_TEMPLATES]
# Templates allowed to adjudicate ALONE under MIB_REASON_ADJ=1 (default
# OFF), per docs/superpowers/specs/2026-07-29-pure-template-reason-
# adjudication.md: single-label in the fresh-mined digital-note inventory
# (162 Finding+Reason lines, note==gold 162/162), tail-less, support >= 5.
# The organizer-seeded traps ("Review-only risk flag present:" NR 27/D 2,
# "Revoked sponsor:" D 6/NR 2) and every tail-bearing or low-support
# template are excluded by content: they compete in scoring but can never
# mint a label.
_REASON_ADJ_LABELS = {
    "Approval supported by surviving visible evidence and exception notes.":
        "APPROVED",
    "Clean or exception-qualified packet.": "APPROVED",
    "Arrival date missing from trusted visible evidence.": "NEEDS_REVIEW",
    "Packet contains damaged or contradictory visible evidence.":
        "NEEDS_REVIEW",
    "Denial supported by damaged registry evidence and visible policy notes.":
        "DENIED",
    "Mandatory fee unpaid.": "DENIED",
    "Transit class cannot authorize declared work.": "DENIED",
}
_REASON_ADJ_UC = {t.upper(): t for t in _REASON_ADJ_LABELS}
# Label-minting is stricter than identification (85/10/0.7 above): measured
# on 118k real scan-OCR lines (0 label-inconsistent accepts at every setting
# swept) and on ~40k corruptions of trap lines (ceiling 44.4 vs this bar).
_REASON_ADJ_MIN_SCORE, _REASON_ADJ_MIN_MARGIN = 90.0, 15.0
_REASON_ADJ_MIN_COVER = 0.75


def _reason_adj_enabled() -> bool:
    return os.environ.get("MIB_REASON_ADJ") == "1"
# Fast gate before the quadratic matcher: a >= 78 read of a 15-21 char
# phrase carries at most ~3 errors, which cannot destroy every 4-gram of
# both the "FINDING:" prefix and the label word at once. Checked on a
# confusion-normalized copy (1->I, 0->O, ...) so common digit misreads
# ("F1ND1NG:") keep their grams.
_PHRASE_GRAMS = (
    "FIND", "INDI", "NDIN", "DING", "ING:",
    "DENI", "NIED", "APPR", "OVED", "NEED", "EEDS", "REVI", "VIEW",
)
_GATE_TRANS = str.maketrans("1|!05826", "IIIOSBZG")


@dataclass(frozen=True)
class FlagsCandidate:
    flags: frozenset[str]
    parsed_ok: bool
    tier: float
    source: Source
    conf: float


@dataclass(frozen=True)
class Finding:
    label: str
    reason: str
    source: Source
    conf: float
    # Reason-template-only findings (MIB_REASON_ADJ=1) never join N0
    # note semantics; policy consults them only on under-determined paths.
    template_only: bool = False
    template: str | None = None


@dataclass
class CaseEvidence:
    """Reconciled per-field evidence feeding policy + output row."""
    values: dict[str, str | None] = dc_field(default_factory=dict)
    known: dict[str, bool] = dc_field(default_factory=dict)
    conf: dict[str, float] = dc_field(default_factory=dict)
    flags: set[str] = dc_field(default_factory=set)
    flags_known: bool = False
    finding: Finding | None = None
    # Two trusted findings that disagree are definitionally under-determined
    # (a genuine packet carries exactly one adjudicator note); page order
    # must never decide between them.
    finding_conflict: bool = False
    # Reason-template-only adjudication candidate (MIB_REASON_ADJ=1; None
    # otherwise). Kept apart from `finding`: it must never outrank N0 or
    # any positive field decision.
    template_finding: Finding | None = None
    # The manual's date rule keys on the intake form: a registry date never
    # substitutes (verified: registry shows a clean digital date on gold
    # NEEDS_REVIEW cases whose intake arrival is UNREADABLE/blank).
    arrival_on_intake: bool = False

    def value(self, fld: str) -> str | None:
        return self.values.get(fld)

    def is_known(self, fld: str) -> bool:
        return self.known.get(fld, False)


# Body labels that identify a document type when its header is unreadable.
# Distinctive labels appear on exactly one template and count double.
_TYPE_LABELS: dict[str, set[str]] = {
    "intake": {"Case ID", "Applicant", "Species Code", "Home World",
               "Visa Class", "Sponsor ID", "Arrival Date", "Declared Purpose"},
    "registry": {"Registry Name", "Home World", "Species Code",
                 "Registry Status", "Arrival Date"},
    "fee_receipt": {"Case ID", "Fee Status", "Amount", "Waiver Code"},
    "biometric": {"Case ID", "Applicant", "Species Match", "Observed flags"},
    "attestation": {"Sponsor ID", "Applicant", "Purpose", "Visa Class"},
}
_TYPE_DISTINCTIVE: dict[str, set[str]] = {
    "intake": {"Declared Purpose"},
    "registry": {"Registry Name", "Registry Status"},
    "fee_receipt": {"Fee Status", "Amount", "Waiver Code"},
    "biometric": {"Species Match", "Observed flags"},
    "attestation": set(),
}


def _labels_on(page: Page) -> set[str]:
    found = set()
    for line in page.lines:
        anchored = _match_anchor(line.text)
        if anchored is not None:
            found.add(anchored[1])
    return found


def _type_from_labels(page: Page) -> str | None:
    found = _labels_on(page)
    scores = {
        doc: len(found & labels) + len(found & _TYPE_DISTINCTIVE[doc])
        for doc, labels in _TYPE_LABELS.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if ranked[0][1] >= 3 and ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return None


_ALL_DISTINCTIVE = frozenset().union(*_TYPE_DISTINCTIVE.values())


def _finding_eligible(page: Page) -> bool:
    """Only Adjudicator-Note pages may carry a trusted Finding.

    Pages typed as another template — and untyped pages showing another
    template's distinctive labels — must never have a "Finding:" line
    honored: a single visible forged line on, say, a registry extract would
    otherwise override every other piece of evidence. Genuine notes whose
    scanned header defeats typing are sparse pages with no foreign labels,
    so they stay eligible.
    """
    if page.doc_type == "note":
        return True
    if page.doc_type is not None:
        return False
    return not (_labels_on(page) & _ALL_DISTINCTIVE)


def detect_doc_type(page: Page) -> tuple[str | None, float]:
    """Identify the page's document type.

    Header aliases are compared window-wise (head and tail of the line):
    digital line assembly merges the "MIB-XXXXXX | MIB Eyes Only" corner tag
    into the title line, and OCR both appends debris and drops header words,
    so whole-line distance would reject genuine headers. Pages whose header
    never OCRs fall back to identification by the set of body labels found.
    """
    best: tuple[str | None, float] = (None, _UNKNOWN_TIER)
    best_dist = 1.0
    for line in page.lines[:10]:
        text = line.text.strip().strip(" .|-_~:;,'\"`")
        if len(text) < 8:
            continue
        for alias, doc_type, tier in _DOC_ALIASES:
            span = len(alias) + 3
            for window in (text[:span], text[-span:]):
                dist = vocab.weighted_distance(window, alias) / len(alias)
                if dist <= 0.42 and dist < best_dist:
                    best, best_dist = (doc_type, tier), dist
    if best[0] is None:
        doc_type = _type_from_labels(page)
        if doc_type is not None:
            return doc_type, DOC_TIERS[doc_type]
    return best


def _date_word_in(text: str) -> bool:
    return any(
        vocab.weighted_distance(tok.strip(":;.,").lower(), "date")
        / max(len(tok), 4) <= _DATE_WORD_MAX_DIST
        for tok in text.split() if 3 <= len(tok) <= 7
    )


def _flags_word_rescue(text: str) -> tuple[set[str], bool] | None:
    """Recover an Observed-flags read whose label is too damaged to anchor.

    Requires a token near "observed"/"flags"; the value part must then parse
    as known flag names or "none" (parse_flags is junk-safe by construction).
    """
    tokens = [t.strip(":;.,|").lower() for t in text.split()]
    hit_idx = None
    for i, t in enumerate(tokens):
        if 4 <= len(t) <= 9 and (
            vocab.weighted_distance(t, "flags") <= 1.2
            or vocab.weighted_distance(t, "observed") <= 1.6
        ):
            hit_idx = i
    if hit_idx is None or hit_idx + 1 >= len(tokens):
        return None
    value = " ".join(text.split()[hit_idx + 1:]).lstrip(":;.,- ")
    flags, ok = vocab.parse_flags(vocab.strip_captions(value))
    return (flags, ok) if ok else None


def _fee_word_rescue(text: str) -> str | None:
    """Recover a fee status from a line whose label anchor is too damaged
    to match ("| Fee waived |"): requires a token close to "fee" plus a
    near-exact fee-status vocabulary token."""
    tokens = [t.strip(":;.,|").lower() for t in text.split()]
    if not any(vocab.weighted_distance(t, "fee") <= 1.0
               for t in tokens if 2 <= len(t) <= 5):
        return None
    for t in tokens:
        if 3 <= len(t) <= 8:
            hit = vocab.match_vocab(t, vocab.FEE_STATUSES, 0.15)
            if hit:
                return hit
    return None


def _match_anchor(text: str) -> tuple[str, str, str] | None:
    """Match a line against label anchors; return (field, label, value)."""
    words = text.split()
    while words and not any(ch.isalnum() for ch in words[0]):
        words = words[1:]   # ruled-line debris OCRs as leading "|" tokens
    if not words:
        return None
    for fld, label in _ANCHORS:
        label_words = label.split()
        k = len(label_words)
        if len(words) < k:
            continue
        prefix = " ".join(words[:k])
        # keep a trailing colon on the prefix from counting as a mismatch
        dist = vocab.weighted_distance(prefix.rstrip(":;.,"), label)
        if dist / max(len(prefix), len(label)) <= _ANCHOR_MAX_DIST:
            value = " ".join(words[k:]).lstrip(":;.,- ").strip()
            return fld, label, value
    # No-space fallback: the PP-OCR recognizer emits "SponsorID:SPN-8468";
    # compare the squashed label against the line's leading characters.
    squashed = "".join(words)
    for fld, label in _ANCHORS:
        flat = label.replace(" ", "")
        head = squashed[: len(flat) + 1]
        dist = vocab.weighted_distance(head.rstrip(":;.,"), flat)
        if dist / len(flat) <= _ANCHOR_MAX_DIST - 0.08:
            value = squashed[len(flat):].lstrip(":;.,- ").strip()
            if value:
                return fld, label, value
    return None


def _correct(fld: str, raw: str) -> str | None:
    raw = vocab.strip_captions(raw)
    if not raw:
        return None
    if fld == "species_code":
        return vocab.match_vocab(raw, vocab.SPECIES_CODES, 0.34)
    if fld == "home_world":
        return vocab.match_vocab(raw, vocab.HOME_WORLDS, 0.30)
    if fld == "visa_class":
        return vocab.match_vocab(raw, vocab.VISA_CLASSES, 0.30)
    if fld == "declared_purpose":
        return vocab.match_vocab(raw, vocab.PURPOSES, 0.34)
    if fld == "fee_status":
        # 0.34 admits real damage ("carved" -> waived); the runner-up margin
        # test plus the authoritative amount check guard paid vs unpaid
        return vocab.match_vocab(raw.lower(), vocab.FEE_STATUSES, 0.34)
    if fld == "sponsor_id":
        return vocab.repair_sponsor_id(raw)
    if fld == "arrival_date":
        return vocab.repair_date(raw)
    if fld == "case_id":
        return vocab.repair_case_id(raw)
    if fld == "applicant_name":
        corrected = vocab.correct_name(raw)
        return corrected if corrected else None
    if fld == "registry_status":
        return vocab.match_vocab(raw.upper(), ["CLEAR", "EMBARGO REVIEW"], 0.35)
    if fld == "waiver_code":
        return raw
    return None


def _parse_correction(text: str) -> tuple[str, str] | None:
    m = _CORRECTION_RE.search(text)
    if not m:
        return None
    body = m.group(1).strip().rstrip(".")
    lowered = body.lower()
    for keyword, fld in _CORRECTION_FIELDS:
        if keyword in lowered.split(" is ")[0]:
            _, _, value = body.partition(" is ")
            if value.strip():
                return fld, value.strip()
    return None


def _reason_continuation(lines: list[Line], idx: int) -> str:
    """Unanchored text on the next lines, continuing a Finding's reason."""
    reason = ""
    for nxt in lines[idx + 1: idx + 3]:
        if _match_anchor(nxt.text) is not None or _FINDING_RE.search(nxt.text):
            continue
        if FOOTER_ID_RE.search(nxt.text):
            continue
        tokens = [t for t in nxt.text.split() if any(ch.isalnum() for ch in t)]
        if tokens and len(tokens) <= 2 and vocab.match_vocab(
                "_".join(tokens).upper(), _ADJUDICATIONS, 0.3):
            # the 20pt stamp word rendered between reason lines is not
            # reason text; glued on, it poisons the flag-vocab match
            continue
        reason += " " + nxt.text
    return reason


def _parse_finding(lines: list[Line], idx: int) -> Finding | None:
    line = lines[idx]
    # Tier-1 findings are printed in near-black ink; a light-grey "Finding:"
    # that survived the hidden-span filter is a forgery, not evidence.
    # Sub-trusted engine lines (tier1_ok=False) never mint findings.
    if not line.dark or not line.tier1_ok:
        return None
    m = _FINDING_RE.search(line.text + ".")
    if not m:
        return None
    label = vocab.match_vocab(m.group(1).strip().replace(" ", "_"),
                              _ADJUDICATIONS, 0.3)
    if not label:
        return None
    reason = line.text[m.end():] + _reason_continuation(lines, idx)
    return Finding(label=label, reason=reason.strip(), source=line.source,
                   conf=line.conf)


_STAMP_WORDS = {"APPROVED": "APPROVED", "DENIED": "DENIED",
                "REVIEW": "NEEDS_REVIEW"}


def _stamp_finding(page: Page) -> Finding | None:
    """Recover a note whose "Finding:" body is washed out.

    Damaged scanned notes often lose the small Finding line while the 20pt
    stamp word and the "Reason:" line survive. Both are required together:
    a standalone stamp word alone would match the red DENIED/APPROVED decoy
    stamps on other templates, but those never print a Reason line.
    """
    stamp: tuple[str, float] | None = None
    reason: tuple[str, float] | None = None
    for line in page.lines:
        if not line.dark or not line.tier1_ok:
            continue
        tokens = [t for t in line.text.replace(".", " ").split()
                  if any(ch.isalnum() for ch in t)]
        if 1 <= len(tokens) <= 2:
            hit = vocab.match_vocab(tokens[0].upper(), list(_STAMP_WORDS), 0.25)
            if hit and (stamp is None or line.conf > stamp[1]):
                stamp = (_STAMP_WORDS[hit], line.conf)
        first = tokens[0].strip(":;.,").lower() if tokens else ""
        if first and vocab.weighted_distance(first, "reason") <= 1.2 and \
                (reason is None or line.conf > reason[1]):
            reason = (line.text, line.conf)
    if stamp is not None and reason is None:
        # The first-token test rejects 2+-error reads of "Reason" itself
        # ("Reosan:"). Fall back to whole-phrase matching against the
        # closed reason-template vocabulary — identification only: the
        # template never supplies the label (2/18 concrete templates are
        # organizer-seeded multi-label traps), the stamp still carries the
        # verdict, so the two-signal requirement stands.
        for line in page.lines:
            if not line.dark or not line.tier1_ok:
                continue
            if _reason_template_line(line.text) and \
                    (reason is None or line.conf > reason[1]):
                reason = (line.text, line.conf)
    if stamp is not None and reason is not None:
        # the two corroborating signals carry the evidence jointly: a washed
        # 20pt stamp word reads at low confidence even when unmistakable
        return Finding(label=stamp[0], reason=reason[0], source=Source.OCR,
                       conf=(stamp[1] + reason[1]) / 2)
    return None


def _reason_template_line(text: str) -> bool:
    """True when a line reads as one of the closed Reason templates."""
    norm = " ".join(text.upper().split())
    if not norm or len(norm) > 100:
        return False
    if max(vocab.partial_ratio_bound(norm, tmpl)
           for tmpl in _REASON_TEMPLATES_UC) < _REASON_MIN_SCORE:
        return False
    if FOOTER_ID_RE.search(text) or _match_anchor(text) is not None:
        return False
    tokens = [t for t in norm.split() if any(ch.isalnum() for ch in t)]
    if len(tokens) <= 2:    # stamp words and fragments, never a sentence
        return False
    # Asymmetric on purpose: the template is the needle and must fit
    # INSIDE the line. A short line covering only a template's tail
    # ("iF Home World:" vs "Embargo home world:") must score as the
    # whole-string mismatch it is, not as a clean substring.
    scored = sorted(
        ((vocab.partial_ratio_alignment(tmpl, norm)[0], tmpl)
         for tmpl in _REASON_TEMPLATES_UC),
        reverse=True,
    )
    (best, tmpl), runner_up = scored[0], scored[1][0]
    return (best >= _REASON_MIN_SCORE
            and best - runner_up >= _REASON_MIN_MARGIN
            and len(norm) >= _PHRASE_MIN_COVER * len(tmpl))


def _template_adjudication(page: Page) -> Finding | None:
    """Reason-template-only adjudication candidate (MIB_REASON_ADJ=1 only).

    A note whose Finding line AND stamp are destroyed can still carry a
    near-perfect read of its Reason sentence. When that read clears the
    strict whole-phrase bar against ALL 13 template stems and the winner is
    one of the seven single-label templates, the mined label becomes a
    candidate — consumed by policy strictly below every positive decision.
    The traps and excluded templates compete in the scoring, so a read that
    looks at all like one of them loses the winner-take-all or the margin
    and never mints (measured trap-corruption ceiling: 44.4 vs the 90 bar).
    """
    hits: list[tuple[float, float, str, Line]] = []
    for line in page.lines:
        if not line.dark or not line.tier1_ok:
            continue
        norm = " ".join(line.text.upper().split())
        if not norm or len(norm) > 100:
            continue
        if max(vocab.partial_ratio_bound(norm, t)
               for t in _REASON_ADJ_UC) < _REASON_ADJ_MIN_SCORE:
            continue
        if FOOTER_ID_RE.search(line.text) or _match_anchor(line.text) is not None:
            continue
        tokens = [t for t in norm.split() if any(ch.isalnum() for ch in t)]
        if len(tokens) <= 2:    # stamp words and fragments, never a sentence
            continue
        scored = sorted(
            ((vocab.partial_ratio_alignment(tmpl, norm)[0], tmpl)
             for tmpl in _REASON_TEMPLATES_UC),
            reverse=True,
        )
        (best, tmpl), runner_up = scored[0], scored[1][0]
        if tmpl not in _REASON_ADJ_UC:
            continue    # a trap/excluded template won the read: never mint
        if best < _REASON_ADJ_MIN_SCORE or \
                best - runner_up < _REASON_ADJ_MIN_MARGIN:
            continue
        if len(norm) < _REASON_ADJ_MIN_COVER * len(tmpl):
            continue
        hits.append((best, line.conf, _REASON_ADJ_UC[tmpl], line))
    if not hits:
        return None
    if len({_REASON_ADJ_LABELS[canon] for _, _, canon, _ in hits}) > 1:
        return None     # two trusted sentences disagree: under-determined
    _, conf, canon, line = max(hits, key=lambda h: (h[1], h[0]))
    return Finding(label=_REASON_ADJ_LABELS[canon], reason=line.text,
                   source=line.source, conf=conf, template_only=True,
                   template=canon)


def _fuzzy_phrase_finding(page: Page) -> Finding | None:
    """Recover a Finding whose printed line is garbled past the regex.

    Damaged scans misread the phrase wholesale ("FNDING: DEN1ED.")
    where `_parse_finding`'s literal "finding" token never surfaces.
    Accept only a whole-phrase read: the best label scores >= 78
    partial-ratio points AND >= 3 over the runner-up label AND the line
    is long enough to cover the phrase. Two qualifying lines naming
    different labels are a conflict, not a vote — abstain. Anything
    below threshold degrades to the current behavior (no finding).
    """
    hits: list[tuple[float, float, int, str, str]] = []
    for idx, line in enumerate(page.lines):
        if not line.dark or not line.tier1_ok:
            continue
        text = " ".join(line.text.split())
        # merged prose past this length is junk territory for a 15-21
        # char phrase; the regex path still handles long genuine lines
        if not text or len(text) > 80:
            continue
        upper = text.upper()
        gated = upper.translate(_GATE_TRANS)
        if not any(gram in gated for gram in _PHRASE_GRAMS):
            continue
        if _match_anchor(text) is not None or FOOTER_ID_RE.search(text):
            continue
        scored = sorted(
            ((*vocab.partial_ratio_alignment(phrase, upper), label, phrase)
             for label, phrase in _FINDING_PHRASES.items()),
            reverse=True,
        )
        score, end, label, phrase = scored[0]
        if score < _PHRASE_MIN_SCORE:
            continue
        if len(upper) < _PHRASE_MIN_COVER * len(phrase):
            continue
        # The runner-up margin is symmetric: a crop/damage-truncated read
        # of a LONGER competitor phrase scores badly against that phrase
        # whole-string ("Finding: NEEDS_F N" vs NEEDS_REVIEW) while a
        # shorter label ("DENIED") can sneak past the threshold — the
        # competitor must get credit for the line being its prefix.
        runner_up = max(vocab.partial_ratio(other, upper)
                        for lab, other in _FINDING_PHRASES.items()
                        if lab != label)
        if score - runner_up < _PHRASE_MIN_MARGIN:
            continue
        hits.append((score, line.conf, idx, label, text[end:].lstrip(" .,:;-")))
    if not hits or len({label for _, _, _, label, _ in hits}) > 1:
        return None
    _, conf, idx, label, tail = max(hits)
    line = page.lines[idx]
    reason = (tail + _reason_continuation(page.lines, idx)).strip()
    return Finding(label=label, reason=reason, source=line.source, conf=conf)


def _foreign_page(page: Page, case_id: str) -> bool:
    """True when the page's own digital footer names ANOTHER case.

    Every generated page carries a vector-text "Packet MIB-XXXXXX / page N"
    footer; a page appended from a different case leaks that case's values
    and verdict, so it is excluded from evidence entirely. Only exact
    digital spans are consulted — an OCR misread must never disqualify a
    genuine page.
    """
    for span in page.visible_spans:
        m = FOOTER_ID_RE.search(span.text)
        if m and m.group(1) != case_id:
            return True
    return False


def _trim_decoy_lines(page: Page, case_id: str) -> list[Line]:
    """Cut off an "archived adjacent applicant" block.

    The decoy is a sub-section appended to an otherwise genuine page: the
    lines above the marker carry real values for the active applicant, the
    lines from the marker on (foreign Case ID such as MIB-000000, planted
    adjudication) are bait. Only digital text is trusted for the cut — an
    OCR misread of a digit must never truncate a genuine page.
    """
    for idx, line in enumerate(page.lines):
        if line.source != Source.DIGITAL:
            continue
        if _DECOY_MARKER in line.text.casefold():
            return page.lines[:idx]
        anchored = _match_anchor(line.text)
        if anchored and anchored[0] == "case_id":
            found = vocab.repair_case_id(anchored[2])
            if found and found != case_id:
                return page.lines[:idx]
    return page.lines


def _prose_attestation_candidates(page: Page, tier: float) -> list[FieldCandidate]:
    """Parse the prose attestation letter variant."""
    text = "\n".join(line.text for line in page.lines)
    if "attest" not in text.casefold():
        return []
    conf = min((line.conf for line in page.lines), default=0.99)
    found: list[tuple[str, str]] = []
    if m := _ATTEST_SPONSOR_RE.search(text):
        found.append(("sponsor_id", f"SPN-{m.group(1)}"))
    if m := _ATTEST_NAME_RE.search(text):
        found.append(("applicant_name", m.group(1).replace("\n", " ")))
    if m := _ATTEST_PURPOSE_RE.search(text.replace("\n", " ")):
        found.append(("declared_purpose", m.group(1)))
    if m := _ATTEST_VISA_RE.search(text):
        found.append(("visa_class", m.group(1)))
    return [
        FieldCandidate(fld=fld, raw=raw, value=_correct(fld, raw), tier=tier,
                       page_index=page.index, source=Source.DIGITAL
                       if all(l.source == Source.DIGITAL for l in page.lines)
                       else Source.OCR, conf=conf)
        for fld, raw in found
    ]


def collect_candidates(
    pages: list[Page], case_id: str,
) -> tuple[list[FieldCandidate], list[FlagsCandidate], list[Finding]]:
    """Parse every trusted line on every non-decoy page into candidates."""
    candidates: list[FieldCandidate] = []
    flag_candidates: list[FlagsCandidate] = []
    findings: list[Finding] = []

    for page in pages:
        if _foreign_page(page, case_id):
            continue
        page.lines = _trim_decoy_lines(page, case_id)
        doc_type, tier = detect_doc_type(page)
        page.doc_type = doc_type
        finding_ok = _finding_eligible(page)
        if doc_type == "attestation":
            candidates.extend(_prose_attestation_candidates(page, tier))
        page_findings_before = len(findings)
        for idx, line in enumerate(page.lines):
            correction = _parse_correction(line.text)
            if correction is not None:
                fld, raw = correction
                candidates.append(FieldCandidate(
                    fld=fld, raw=raw, value=_correct(fld, raw),
                    tier=_CORRECTION_TIER, page_index=page.index,
                    source=line.source, conf=line.conf,
                ))
                continue
            if finding_ok:
                finding = _parse_finding(page.lines, idx)
                if finding is not None:
                    findings.append(finding)
                    continue
            anchored = _match_anchor(line.text)
            if anchored is None:
                # Arrival rescue: damaged labels ("Antvel Date: 2026-03-15")
                # defeat anchoring, but a fuzzy "Date" token next to a valid
                # ISO date is unambiguous — no other dated line exists on
                # these templates.
                if line.source == Source.OCR and _date_word_in(line.text):
                    value = vocab.repair_date(line.text)
                    if value is not None:
                        candidates.append(FieldCandidate(
                            fld="arrival_date", raw=line.text, value=value,
                            tier=tier, page_index=page.index,
                            source=line.source, conf=line.conf,
                        ))
                elif line.source == Source.OCR:
                    fee = _fee_word_rescue(line.text)
                    if fee is not None:
                        candidates.append(FieldCandidate(
                            fld="fee_status", raw=line.text, value=fee,
                            tier=tier, page_index=page.index,
                            source=line.source, conf=line.conf * 0.9,
                        ))
                    flags_read = _flags_word_rescue(line.text)
                    if flags_read is None and doc_type == "biometric":
                        # OCR sometimes splits the flags row, leaving the
                        # value as its own line; flag names are distinctive
                        # enough to accept standalone on the slip itself.
                        found, ok = vocab.parse_flags(line.text)
                        if ok and found:
                            flags_read = (found, True)
                    if flags_read is not None:
                        flag_candidates.append(FlagsCandidate(
                            flags=frozenset(flags_read[0]), parsed_ok=True,
                            tier=tier, source=line.source,
                            conf=line.conf * 0.9,
                        ))
                    # SPN-#### is self-identifying: harvest it from any
                    # visible line even when the Sponsor ID label is gone.
                    sponsor = vocab.repair_sponsor_id(line.text, loose=True)
                    if sponsor is not None:
                        candidates.append(FieldCandidate(
                            fld="sponsor_id", raw=line.text, value=sponsor,
                            tier=tier, page_index=page.index,
                            source=line.source, conf=line.conf * 0.85,
                        ))
                continue
            fld, _, raw = anchored
            if fld == "risk_flags":
                flags, ok = vocab.parse_flags(raw)
                flag_candidates.append(FlagsCandidate(
                    flags=frozenset(flags), parsed_ok=ok, tier=tier,
                    source=line.source, conf=line.conf,
                ))
                continue
            candidates.append(FieldCandidate(
                fld=fld, raw=raw, value=_correct(fld, raw), tier=tier,
                page_index=page.index, source=line.source, conf=line.conf,
            ))
        if finding_ok and len(findings) == page_findings_before:
            recovered = _stamp_finding(page) or _fuzzy_phrase_finding(page)
            if recovered is None and _reason_adj_enabled():
                recovered = _template_adjudication(page)
            if recovered is not None:
                findings.append(recovered)
    return candidates, flag_candidates, findings


DOC_TIERS = {doc: tier for doc, tier in _DOC_HEADERS.values()}


# Voting weights: conf * (1 + tier bonus). Bonuses are small so cross-page
# agreement (a second supporting page) always beats any single page —
# digital-conflict audits show strict majority matches the label 13/13.
# Tie-break orders are field-specific where the audits show the intake
# losing 2-way ties (registry wins arrival/name, attestation wins sponsor).
_TIER_BONUS = {1.0: .5, _CORRECTION_TIER: .45, 2.0: .3, 2.5: .3, 3.0: .25,
               4.0: .2, 5.0: .15}
_REGISTRY_FIRST = {1.0: .5, _CORRECTION_TIER: .45, 5.0: .35, 4.0: .3,
                   3.0: .25, 2.0: .2, 2.5: .2}
_ATTESTATION_FIRST = {1.0: .5, _CORRECTION_TIER: .45, 4.0: .35, 5.0: .3,
                      3.0: .25, 2.0: .2, 2.5: .2}
_FIELD_TIER_BONUS = {
    "arrival_date": _REGISTRY_FIRST,
    "applicant_name": _REGISTRY_FIRST,
    "sponsor_id": _ATTESTATION_FIRST,
}
_KNOWN_MIN_OCR_CONF = 0.55
_FLAG_MIN_CONF = 0.40
_NONE_MIN_CONF = 0.50

SCHEMA_FIELDS = [
    "applicant_name", "species_code", "home_world", "visa_class",
    "sponsor_id", "arrival_date", "declared_purpose", "fee_status",
]


def _weight(candidate: FieldCandidate) -> float:
    bonus = _FIELD_TIER_BONUS.get(candidate.fld, _TIER_BONUS)
    return candidate.conf * (1.0 + bonus.get(candidate.tier, 0.1))


def _parse_amount(raw: str) -> int | None:
    m = _AMOUNT_RE.search(raw)
    if not m:
        return None
    digits = m.group(1).translate(vocab._DIGIT_REPAIRS)
    return int(digits) if digits.isdigit() else None


def _receipt_verdict(status: str | None, amount: int | None,
                     waiver: str | None) -> str | None:
    """True fee status for one receipt page.

    Verified against every digital fee receipt in train (449 pages, zero
    exceptions): the amount is authoritative over the printed status line —
    $809.00 is always a paid fee (297/297, including receipts whose status
    line says "unpaid"), $0.00 with a waiver code is always waived
    (106/106), and $0.00 with waiver N/A resolves to the printed status
    except that a paid/waived claim without money or waiver is the
    generator's "inconsistent receipt" pattern for a literal unknown.
    The $809 geometry observation originates from MIT-licensed public
    solutions (attributed in the memo); the full table was re-derived and
    verified here. None means the page under-determines the fee.
    """
    waiver_state = None
    if waiver is not None:
        waiver_state = "present" if "WAIVER" in waiver.upper() else "na"

    if amount == 809:
        return "paid"
    if amount == 0:
        if waiver_state == "present":
            return "waived"
        if status == "unpaid":
            return "unpaid"
        if status in ("paid", "waived", "unknown"):
            return "unknown"
        return "unknown"
    # amount unread (or an implausible misread): status line decides
    if status == "waived" and waiver_state == "na":
        return "unknown"
    return status


def _resolve_fee(candidates: list[FieldCandidate]) -> tuple[str | None, float]:
    by_page: dict[int, dict[str, FieldCandidate]] = {}
    for c in candidates:
        if c.fld in ("fee_status", "fee_amount", "waiver_code"):
            slot = by_page.setdefault(c.page_index, {})
            held = slot.get(c.fld)
            # a parsed value always beats an unparsed read of the same row
            if held is None or (c.value is not None, c.conf) > \
                    (held.value is not None, held.conf):
                slot[c.fld] = c

    scores: dict[str, float] = {}
    confs: dict[str, float] = {}
    for view in by_page.values():
        status = view.get("fee_status")
        amount_cand = view.get("fee_amount")
        amount = _parse_amount(amount_cand.raw) if amount_cand else None
        waiver = view.get("waiver_code")
        verdict = _receipt_verdict(status.value if status else None, amount,
                                   waiver.value if waiver else None)
        if verdict is None:
            continue
        anchor = status or amount_cand
        weight = _weight(anchor) if anchor else 0.5
        scores[verdict] = scores.get(verdict, 0.0) + weight
        confs[verdict] = max(confs.get(verdict, 0.0),
                             anchor.conf if anchor else 0.5)
    if not scores:
        return None, 0.0
    value = max(scores, key=scores.get)
    return value, confs[value]


_NAME_LABEL_TIERS = (2.0, 3.0, 5.0)
_NAME_NEAR_CUT = 0.15


def _digital_name_choice(pool: list[FieldCandidate], voted: str) -> str:
    """Let an exact digital name beat the OCR vote only when that is safe.

    Blanket digital-first fixes 15 names but breaks 3 on train: planted
    wrong-name digital label rows (MIB-000064/000564) and a wrong-applicant
    digital attestation (MIB-000175). Guards: attestation prose (tier 4)
    never outvotes a digital label row; agreeing digital sources win
    outright; a SINGLE digital line wins only over an OCR-confusable
    variant of itself — a far digital-vs-vote mismatch is the planted-lie
    signature, so the cross-page vote stands.
    """
    digital = [c for c in pool if c.source == Source.DIGITAL]
    label_rows = [c for c in digital if c.tier in _NAME_LABEL_TIERS]
    if label_rows:
        digital = label_rows
    if not digital:
        return voted
    values = {c.value for c in digital}
    if len(values) != 1:
        return voted
    value = next(iter(values))
    if value == voted:
        return voted
    if len(digital) >= 2:
        return value
    norm = vocab.weighted_distance(value.lower(), voted.lower()) \
        / max(len(value), len(voted))
    return value if norm <= _NAME_NEAR_CUT else voted


_REASON_SPONSOR_RE = re.compile(r"[Rr]evoked sponsor:?\s*(SPN[\s\-–—._:]*[0-9]{4})")
_REASON_WORLD_RE = re.compile(r"[Ee]mbargo home world:?\s*([A-Za-z0-9][A-Za-z0-9 \-]*?)[.,;]")


def _mine_reason_fields(ev: CaseEvidence) -> None:
    """Field values stated by a trusted note's Reason line (tier-1 evidence).

    Verified against every digital train note (162 Finding+Reason lines):
    fee unpaid 6/6, fee unknown 3/3, revoked sponsor 8/8, embargo world 7/7,
    transit-class 12/12 match gold. The finding has already passed the
    dark-ink / note-page / conflict / foreign-footer checks, so its Reason
    carries the same trust as the Finding label (297/297 corpus-wide).
    """
    reason = " ".join(ev.finding.reason.split())
    conf = ev.finding.conf

    def _override(fld: str, value: str) -> None:
        ev.values[fld] = value
        ev.known[fld] = True
        ev.conf[fld] = max(ev.conf.get(fld, 0.0), conf)

    tokens = [t.strip(".,;:").lower() for t in reason.split()]
    if any(vocab.weighted_distance(t, "fee") <= 0.8 for t in tokens):
        if any(vocab.weighted_distance(t, "unpaid") <= 1.2 for t in tokens):
            _override("fee_status", "unpaid")
        elif any(vocab.weighted_distance(t, "unknown") <= 1.2 for t in tokens):
            _override("fee_status", "unknown")
    m = _REASON_SPONSOR_RE.search(reason)
    if m:
        spn = vocab.repair_sponsor_id(m.group(1))
        if spn:
            _override("sponsor_id", spn)
    m = _REASON_WORLD_RE.search(reason)
    if m:
        world = vocab.match_vocab(m.group(1), vocab.HOME_WORLDS, 0.35)
        if world:
            _override("home_world", world)
    if "transit class cannot" in reason.lower():
        _override("visa_class", "TRANSIT-7")


def reconcile(
    candidates: list[FieldCandidate],
    flag_candidates: list[FlagsCandidate],
    findings: list[Finding],
) -> CaseEvidence:
    ev = CaseEvidence()

    for fld in SCHEMA_FIELDS:
        pool = [c for c in candidates if c.fld == fld and c.value is not None]
        corrections = [c for c in pool if c.tier == _CORRECTION_TIER]
        if corrections:
            best = max(corrections, key=_weight)
            ev.values[fld] = best.value
            ev.known[fld] = True
            ev.conf[fld] = best.conf
            continue
        if fld == "fee_status":
            value, conf = _resolve_fee(candidates)
            ev.values[fld] = value
            ev.known[fld] = value is not None
            ev.conf[fld] = conf
            continue
        if not pool:
            ev.values[fld] = None
            ev.known[fld] = False
            ev.conf[fld] = 0.0
            continue
        if fld in ("visa_class", "sponsor_id", "arrival_date"):
            # an exact vector-text read beats any OCR vote-sum: the same
            # font misreads identically on 2-3 scan pages and outvotes the
            # one digital line (train: 29 fixes / 0 breaks on these fields)
            digital = [c for c in pool if c.source == Source.DIGITAL]
            if digital:
                pool = digital
        if fld == "applicant_name":
            # names are two lexicon tokens; a fragment read ("Zamora") must
            # never outvote a full read ("Solul Zamora")
            full = [c for c in pool if len(c.value.split()) >= 2]
            pool = full or pool
        scores: dict[str, float] = {}
        for c in pool:
            scores[c.value] = scores.get(c.value, 0.0) + _weight(c)
        value = max(scores, key=scores.get)
        if fld == "applicant_name":
            value = _digital_name_choice(pool, value)
        supporters = [c for c in pool if c.value == value]
        best = max(supporters, key=lambda c: c.conf)
        ev.values[fld] = value
        ev.known[fld] = (best.source == Source.DIGITAL
                         or best.conf >= _KNOWN_MIN_OCR_CONF)
        ev.conf[fld] = best.conf

    ev.arrival_on_intake = any(
        c.fld == "arrival_date" and c.value is not None
        and c.tier in (_CORRECTION_TIER, 2.0)
        for c in candidates
    )

    for fc in flag_candidates:
        if not fc.parsed_ok or fc.conf < _FLAG_MIN_CONF:
            continue
        if fc.flags:
            ev.flags |= set(fc.flags)
            ev.flags_known = True
        elif fc.source == Source.DIGITAL or fc.conf >= _NONE_MIN_CONF:
            ev.flags_known = True

    # Template-only findings (MIB_REASON_ADJ=1; empty list otherwise) stay
    # outside N0 semantics: they neither vote on `finding` nor raise a
    # conflict, and they never feed the Reason field/flag miners.
    template_findings = [f for f in findings if f.template_only]
    findings = [f for f in findings if not f.template_only]
    if template_findings and \
            len({f.label for f in template_findings}) == 1:
        ev.template_finding = max(template_findings, key=lambda f: f.conf)

    if findings:
        labels = {f.label for f in findings}
        if len(labels) > 1:
            ev.finding_conflict = True
        else:
            ev.finding = max(findings, key=lambda f: f.conf)
            reason_seg = ev.finding.reason.rpartition(":")[2]
            # the flag name ends at the first period; anything after it
            # (second sentence, residual OCR junk) poisons the vocab match
            reason_flags, _ = vocab.parse_flags(reason_seg.partition(".")[0])
            if not reason_flags:
                reason_flags, _ = vocab.parse_flags(reason_seg)
            if reason_flags:
                ev.flags |= reason_flags
                ev.flags_known = True
            _mine_reason_fields(ev)

    return ev
