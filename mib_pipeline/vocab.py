"""Closed vocabularies and OCR-aware fuzzy matching.

All enumerations below are field vocabularies mined from the public
FIELD_MANUAL.md and the distribution of values in data/train_labels.csv.
They are vocabularies of the document generator, not per-case answers.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache

SPECIES_CODES = [
    "ALPHA_DRACONIAN", "ANDROMEDAN", "AQUARIAN_MANTIS", "ARCTURIAN",
    "CENTAURI_SYNTH", "JOVIAN_GASFORM", "KAIJU_MICRO", "LUNA_SECURID",
    "ORION_GRAYS", "SIRIUS_AVIAN", "TRIANGULAN", "VENUSIAN_MYCELIAL",
]

HOME_WORLDS = [
    "Barnard-c", "Eris Relay", "Europa Station", "Gliese-581g",
    "Kepler-186f", "Luyten-b", "Mars Dome-7", "Proxima-b",
    "Sirius Outpost", "Titan Freeport", "TRAPPIST-1e", "Wolf-1061c",
    "Zeta Reticuli",
]

VISA_CLASSES = ["XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"]

PURPOSES = [
    "archive audit", "cultural exchange", "diplomatic", "field repair",
    "medical consult", "reactor maintenance", "research", "transit",
    "translation", "xenobotany",
]

FEE_STATUSES = ["paid", "waived", "unpaid", "unknown"]

RISK_FLAGS = [
    "active_warrant", "biohazard_red", "identity_conflict",
    "illegible_biometrics", "memory_tampering", "planetary_embargo",
    "rescinded_denial", "sponsor_mismatch",
]

# The generator's closed set of Manual Adjudicator Note "Reason:" sentence
# stems, mined from the printed text of every digital train note (162
# Finding+Reason lines; variable tails — flag name, SPN, world — omitted).
# These identify that a line IS a Reason line; they must never supply the
# Finding label: 2 of the 18 concrete templates are organizer-seeded
# multi-label traps ("Review-only …" is NEEDS_REVIEW 20 / DENIED 2,
# "Revoked sponsor: …" is DENIED 6 / NEEDS_REVIEW 2 — see
# research/12-package-b-spec.md; reason-only adjudication was evaluated
# and rejected).
REASON_TEMPLATES = [
    "Ambiguous packet.",
    "Approval supported by surviving visible evidence and exception notes.",
    "Arrival date missing from trusted visible evidence.",
    "Clean or exception-qualified packet.",
    "Denial supported by damaged registry evidence and visible policy notes.",
    "Disqualifying risk flag:",
    "Embargo home world:",
    "Fee status unknown.",
    "Mandatory fee unpaid.",
    "Packet contains damaged or contradictory visible evidence.",
    "Review-only risk flag present:",
    "Revoked sponsor:",
    "Transit class cannot authorize declared work.",
]

# Applicant names are two tokens drawn from a syllable grammar
# (12 prefixes x 24 suffixes observed across all 1,000 train labels).
_NAME_PREFIXES = [
    "Ari", "Ixo", "Lu", "Mir", "Nex", "Ori",
    "Qor", "Sol", "Tek", "Vee", "Xan", "Za",
]
_NAME_SUFFIXES = [
    "adane", "aix", "akesh", "amora", "anax", "aquell", "arix", "atari",
    "aul", "avara", "avoss", "azarn", "dane", "ix", "kesh", "mora",
    "nax", "quell", "rix", "tari", "ul", "vara", "voss", "zarn",
]
NAME_TOKENS = [p + s for p in _NAME_PREFIXES for s in _NAME_SUFFIXES]

# The cartesian list above doubles every real token with a phantom
# edit-distance-1 neighbour: the generator's actual grammar is 12 true
# prefixes x 12 base suffixes, where the 24 printed suffixes are just the
# 12 base forms plus their a-linked variants and only "Mir" takes the
# a-link ("Miradane", never "Mirdane"; every other prefix takes the bare
# form — "Nexdane", never "Nexadane"). Verified against all 1,000 train
# gold names: exactly these 144 tokens are attested, in both first and
# last position; the other 144 cartesian tokens never occur. Used by the
# MIB_JOINTNAME joint decoder only — the base matcher keeps the wide list.
_NAME_TRUE_PREFIXES = [
    "Ari", "Ixo", "Lu", "Mira", "Nex", "Ori",
    "Qor", "Sol", "Tek", "Vee", "Xan", "Za",
]
_NAME_BASE_SUFFIXES = [
    "dane", "ix", "kesh", "mora", "nax", "quell",
    "rix", "tari", "ul", "vara", "voss", "zarn",
]
NAME_TOKENS_ATTESTED = [p + s for p in _NAME_TRUE_PREFIXES
                        for s in _NAME_BASE_SUFFIXES]

# Characters OCR frequently swaps on this generator's Helvetica output.
_CONFUSION_GROUPS = [
    "0oO", "1iIl|!", "5sS", "8B", "2zZ", "6G", "9g", "e c", "rn m", "u n",
    ".-_–—", "'`’", ",.", ":;",
]
_CHEAP_SUBST: dict[tuple[str, str], float] = {}
for group in _CONFUSION_GROUPS:
    chars = group.replace(" ", "")
    for a in chars:
        for b in chars:
            if a != b:
                _CHEAP_SUBST[(a, b)] = 0.2

_CHEAP_INDEL = set(" .,:;|'`-_")


def _snapfix_enabled() -> bool:
    """One env gate (MIB_SNAPFIX=1, default OFF) for the three text-level
    decode-repair mechanisms: fusion-bridged vocabulary snaps
    (_fusion_rematch below), the flag truncation-prefix (parse_flags), and
    the cross-page sponsor digit vote (fields.sponsor_digit_vote)."""
    return os.environ.get("MIB_SNAPFIX") == "1"


# --- Fusion (2-gram <-> 1-gram) OCR confusions (MIB_SNAPFIX) ----------------
# One glyph OCR'd as two strokes, or two glyphs welded into one, on this
# generator's Helvetica output. A per-character DP charges a full indel +
# substitute for these (>= 1.2 even with the cheap-confusion table, 2.0
# for the pairs the table misses), pushing genuinely-close fused reads past
# every snap threshold. The reference implementations charge one reduced
# cost (mib-intake lexicon.py _MULTI_CONFUSIONS at 0.4; balawal fuzzy.py
# LIGATURE_CONFUSIONS — both MIT, see ATTRIBUTION.md).
_FUSION_PAIRS = [
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"),
    ("ii", "n"), ("n", "ii"), ("vv", "w"), ("w", "vv"),
    ("li", "h"), ("h", "li"), ("nn", "m"), ("m", "nn"),
    ("ri", "n"), ("n", "ri"),
]
# Single-glyph stroke confusions the degradation also produces, admitted
# only on the fusion path (balawal's bridged set): they are too generous
# for the ordinary cheap-substitution table.
_FUSION_SINGLES = frozenset({("c", "o"), ("o", "c"), ("e", "a"), ("a", "e")})
_FUSION_COST = 0.4
# Fusion-bridged matches accept only at a TIGHTER threshold than raw
# matches (balawal's stricter-re-accept discipline: replacing a rejected
# read needs near-exact vocabulary agreement under the repair moves).
_FUSION_TIGHTEN = 0.75


def fusion_distance(a: str, b: str) -> float:
    """weighted_distance plus fusion moves, each charged _FUSION_COST once.

    Full-matrix DP: the 2-gram fusions reach back two rows/columns, which
    the two-row weighted_distance cannot express. Base costs match
    weighted_distance (the head-row init is indel-aware, never dearer), so
    fusion_distance(a, b) <= weighted_distance(a, b) always; only
    match_vocab's stricter fusion-bridged re-accept consumes the gap.
    """
    if a == b:
        return 0.0
    low_a, low_b = a.lower(), b.lower()
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + (0.35 if a[i - 1] in _CHEAP_INDEL else 1.0)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + (0.35 if b[j - 1] in _CHEAP_INDEL else 1.0)
    for i in range(1, m + 1):
        ca, cal = a[i - 1], low_a[i - 1]
        del_cost = 0.35 if ca in _CHEAP_INDEL else 1.0
        row, prev = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            cb, cbl = b[j - 1], low_b[j - 1]
            if ca == cb or cal == cbl:
                sub = 0.0
            else:
                sub = _CHEAP_SUBST.get((cal, cbl), 1.0)
                if sub > _FUSION_COST and (cal, cbl) in _FUSION_SINGLES:
                    sub = _FUSION_COST
            ins_cost = 0.35 if cb in _CHEAP_INDEL else 1.0
            best = min(prev[j] + del_cost, row[j - 1] + ins_cost,
                       prev[j - 1] + sub)
            for x, y in _FUSION_PAIRS:
                lx, ly = len(x), len(y)
                if i >= lx and j >= ly and low_a[i - lx:i] == x \
                        and low_b[j - ly:j] == y:
                    cand = dp[i - lx][j - ly] + _FUSION_COST
                    if cand < best:
                        best = cand
            row[j] = best
    return dp[m][n]


def _fusion_rematch(raw: str, vocab: list[str], max_norm_dist: float,
                    min_margin: float) -> str | None:
    """Stricter re-accept of a distance-rejected read via fusion moves.

    Called by match_vocab (MIB_SNAPFIX=1) only after the raw pass failed
    on DISTANCE — an ambiguity rejection stands, fused or not. Acceptance
    is tighter than the raw pass (_FUSION_TIGHTEN) and keeps the same
    runner-up margin test: an ambiguous fused read stays unknown.
    """
    cutoff = max_norm_dist * _FUSION_TIGHTEN
    scored = []
    for v in vocab:
        longer = max(len(raw), len(v))
        # cheap lower bound: a length gap alone costs >= 0.35/char, so
        # entries that cannot even reach the margin band are skipped
        if 0.35 * abs(len(raw) - len(v)) / longer > cutoff + min_margin:
            continue
        scored.append((fusion_distance(raw, v) / longer, v))
    if not scored:
        return None
    scored.sort()
    best_dist, best = scored[0]
    if best_dist > cutoff:
        return None
    if len(scored) > 1 and scored[1][0] - best_dist < min_margin \
            and best_dist > 0:
        return None
    return best


def weighted_distance(a: str, b: str) -> float:
    """Levenshtein distance with reduced cost for known OCR confusions."""
    if a == b:
        return 0.0
    m, n = len(a), len(b)
    prev = [i * 1.0 for i in range(n + 1)]
    for i in range(1, m + 1):
        ca = a[i - 1]
        del_cost = 0.35 if ca in _CHEAP_INDEL else 1.0
        cur = [prev[0] + del_cost]
        for j in range(1, n + 1):
            cb = b[j - 1]
            if ca == cb or ca.lower() == cb.lower():
                sub = 0.0
            else:
                sub = _CHEAP_SUBST.get((ca.lower(), cb.lower()), 1.0)
            ins_cost = 0.35 if cb in _CHEAP_INDEL else 1.0
            cur.append(min(prev[j] + del_cost, cur[j - 1] + ins_cost, prev[j - 1] + sub))
        prev = cur
    return prev[n]


@lru_cache(maxsize=1024)
def _char_counts(s: str) -> Counter:
    return Counter(s)


def partial_ratio_bound(a: str, b: str) -> float:
    """Cheap upper bound on partial_ratio, from character-multiset overlap.

    Any aligned window shares at most `common` characters with the needle,
    so its ratio is at most 200*common/(len(shorter)+common). Lets callers
    skip the quadratic matcher on lines that cannot reach their threshold.
    """
    if not a or not b:
        return 0.0
    common = sum((_char_counts(a) & _char_counts(b)).values())
    if not common:
        return 0.0
    return 200.0 * common / (min(len(a), len(b)) + common)


def partial_ratio_alignment(needle: str, haystack: str) -> tuple[float, int]:
    """Best-substring similarity of needle within haystack.

    fuzzywuzzy-style partial ratio (0..100) built on stdlib difflib:
    candidate windows of len(needle) are anchored at the matching blocks
    (plus the string head), and the best window's ratio wins. Returns
    (score, window_end) so a caller can take the text after the matched
    phrase. When needle is longer than haystack the score is the plain
    whole-string ratio — a fragment can never claim to contain the phrase.
    """
    if not needle or not haystack:
        return 0.0, 0
    if len(needle) > len(haystack):
        sm = SequenceMatcher(None, needle, haystack, autojunk=False)
        return sm.ratio() * 100.0, len(haystack)
    sm = SequenceMatcher(None, needle, haystack, autojunk=False)
    starts = {0} | {max(j - i, 0) for i, j, _ in sm.get_matching_blocks()}
    best, best_start = 0.0, 0
    for start in sorted(starts):
        window = haystack[start:start + len(needle)]
        ratio = SequenceMatcher(None, needle, window, autojunk=False).ratio()
        if ratio > best:
            best, best_start = ratio, start
    return best * 100.0, min(best_start + len(needle), len(haystack))


def partial_ratio(a: str, b: str) -> float:
    """Symmetric partial ratio: the shorter string slides over the longer."""
    needle, haystack = (a, b) if len(a) <= len(b) else (b, a)
    return partial_ratio_alignment(needle, haystack)[0]


def match_vocab(raw: str, vocab: list[str], max_norm_dist: float,
                min_margin: float = 0.08) -> str | None:
    """Best vocabulary entry for an OCR'd value, or None.

    Accepts only when the best match clears the distance threshold AND the
    runner-up is clearly worse: an ambiguous read must stay unknown rather
    than be forced onto a decision-bearing enum value.
    """
    raw = raw.strip()
    if not raw:
        return None
    scored = sorted(
        (weighted_distance(raw, v) / max(len(raw), len(v)), v) for v in vocab
    )
    best_dist, best = scored[0]
    if best_dist > max_norm_dist:
        # Fusion-bridged second chance (MIB_SNAPFIX=1 only; flag unset
        # keeps this branch dead code): retry with the 2-gram<->1-gram
        # fusion moves at a TIGHTER acceptance threshold.
        if _snapfix_enabled():
            return _fusion_rematch(raw, vocab, max_norm_dist, min_margin)
        return None
    if len(scored) > 1 and scored[1][0] - best_dist < min_margin and best_dist > 0:
        return None
    return best


_DIGIT_REPAIRS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6", "g": "9",
    "A": "4", "T": "7", "e": "8",
})

_SPONSOR_RE = re.compile(r"SPN\s*[-–—._:]?\s*([0-9]{4})(?![0-9])")
# damaged scans eat leading glyphs of the prefix ("N-5809"); the dash is
# then mandatory so bare 4-digit numbers never qualify
_SPONSOR_LOOSE_RE = re.compile(r"(?<![A-Za-z0-9])[SP]{0,2}N\s*[-–—]\s*([0-9]{4})(?![0-9])")
_DATE_RE = re.compile(r"(20[0-9]{2})\s*[-–—._/]\s*([0-9]{2})\s*[-–—._/]\s*([0-9]{2})")
_CASE_RE = re.compile(r"MIB\s*[-–—._:]?\s*([0-9]{6})(?![0-9])")


def repair_sponsor_id(raw: str, loose: bool = False) -> str | None:
    """Normalize an OCR'd sponsor value to SPN-#### or None."""
    text = raw.strip()
    m = _SPONSOR_RE.search(text)
    if not m:
        repaired = _repair_pattern_text(text, "SPN")
        m = _SPONSOR_RE.search(repaired)
    if not m and loose:
        m = _SPONSOR_LOOSE_RE.search(text)
    if not m:
        return None
    return f"SPN-{m.group(1)}"


def repair_case_id(raw: str) -> str | None:
    m = _CASE_RE.search(raw) or _CASE_RE.search(_repair_pattern_text(raw, "MIB"))
    return f"MIB-{m.group(1)}" if m else None


# Every arrival in the corpus falls in these years (packet receipt epoch is
# mid-2026 and staleness reaches back into 2025). An OCR'd year outside the
# range is a misread: snap it when exactly one valid year is a plausible
# single-digit confusion, otherwise reject rather than guess.
_VALID_YEARS = ("2025", "2026")


def _repair_year(year: str) -> int | None:
    if year in _VALID_YEARS:
        return int(year)
    # 2027-2029 is a last-digit misread of 2026: arrivals cannot postdate
    # the mid-2026 receipt epoch by years, and 94.8% of arrivals are 2026
    # (verified corpus-wide; the printed month/day is preserved).
    if year in ("2027", "2028", "2029"):
        return 2026
    scored = sorted((weighted_distance(year, v), v) for v in _VALID_YEARS)
    if scored[0][0] <= 1.0 and scored[1][0] - scored[0][0] >= 0.5:
        return int(scored[0][1])
    return None


def repair_date(raw: str) -> str | None:
    """Normalize an OCR'd date to strict ISO YYYY-MM-DD or None."""
    text = raw.strip()
    m = _DATE_RE.search(text)
    if not m:
        m = _DATE_RE.search(text.translate(_DIGIT_REPAIRS))
        if not m:
            return None
    y = _repair_year(m.group(1))
    if y is None:
        return None
    try:
        return date(y, int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None


def _repair_pattern_text(text: str, keep_prefix: str) -> str:
    """Map confusable letters to digits everywhere except the known prefix."""
    idx = text.upper().find(keep_prefix)
    if idx < 0:
        return text.translate(_DIGIT_REPAIRS)
    head = text[: idx + len(keep_prefix)]
    tail = text[idx + len(keep_prefix):].translate(_DIGIT_REPAIRS)
    return head + tail


# Page decorations the generator prints near values; OCR line merging can
# glue them onto the value text (e.g. "AQUARIAN MANTIS SCAN IMAGE").
_CAPTION_RE = re.compile(
    r"\s*[|\[\](){}~™—–-]*\s*(SCAN IMAGE|PASSPORT IMAGE|REGISTRY IMAGE|"
    r"MIB EYES ONLY|COPY ARTIFACT|CASEWORK|COPY|FILED|ARCHIVE|MIB INTAKE)"
    r"\s*[|\[\](){}~™.—–-]*\s*$"
)


def strip_captions(raw: str) -> str:
    text = raw.strip()
    while True:
        stripped = _CAPTION_RE.sub("", text).strip()
        if stripped == text:
            return text
        text = stripped


def correct_name(raw: str) -> str | None:
    """Fuzzy-correct a name against the syllable-grammar lexicon.

    Names are two lexicon tokens. When at least two tokens match, keep only
    the matches — dropping OCR line-merge debris such as page captions
    ("PASSPORT IMAGE"). A value with no lexicon support at all is rejected:
    it is OCR junk or a damage placeholder ("[NAME CUT OUT]"), and letting
    it through would outvote genuine low-confidence reads from other pages.
    """
    tokens = [t for t in re.split(r"[^A-Za-z]+", raw) if len(t) >= 2]
    corrected = [
        (match_vocab(tok, NAME_TOKENS, max_norm_dist=0.34), tok) for tok in tokens
    ]
    matched = [fixed for fixed, _ in corrected if fixed]
    if len(matched) >= 2:
        return " ".join(matched[:2])
    if not matched:
        return None
    return " ".join(fixed if fixed else tok for fixed, tok in corrected)


def is_grammar_name(value: str | None) -> bool:
    """True for a legal two-token name from the syllable grammar."""
    tokens = str(value or "").split()
    return len(tokens) == 2 and all(t in NAME_TOKENS for t in tokens)


# Joint-decode acceptance (MIB_JOINTNAME): normalized whole-name distance
# ceiling and runner-up margin, calibrated on the 640 label-anchored name
# reads harvested from the scan-OCR corpus (see ctc-branch-VALIDATION.md;
# measured at this point: 11 mints, 7 gold-right, and truncated-suffix
# ties abstain). The shortlist size bounds the joint space; the
# partial_ratio_bound prefilter guards each weighted_distance call.
_JOINT_MAX_NORM = 0.40
_JOINT_MIN_MARGIN = 0.01
_JOINT_SHORTLIST = 12
_JOINT_BOUND_MIN = 55.0


def correct_name_joint(raw: str) -> str | None:
    """Joint two-token decode of a garbled name over the attested grammar.

    Idea from the MIT-licensed moonshots public solution (see
    ATTRIBUTION.md): score whole candidate names against the whole read,
    so a clean token carries its garbled partner ("Aririx lozan" still
    finds "Aririx Ixozarn") instead of both tokens snapping — or failing —
    independently. Their rapidfuzz joint scan is replaced by per-position
    weighted_distance shortlists over the attested 144-token lexicon with
    the partial_ratio_bound prefilter in front of every full distance
    computation. Returns a grammar-legal name or None; callers only invoke
    it after correct_name failed to produce a legal two-token name, so a
    clean read can never be rewritten.
    """
    parts = [t for t in re.split(r"[^A-Za-z]+", raw) if len(t) >= 2]
    if len(parts) != 2:
        return None

    def shortlist(token: str) -> list[str]:
        scored = sorted(
            (weighted_distance(token, cand) / max(len(token), len(cand)), cand)
            for cand in NAME_TOKENS_ATTESTED
        )
        return [cand for _, cand in scored[:_JOINT_SHORTLIST]]

    read = f"{parts[0]} {parts[1]}"
    best: tuple[float, str] | None = None
    runner: tuple[float, str] | None = None
    for first in shortlist(parts[0]):
        for last in shortlist(parts[1]):
            cand = f"{first} {last}"
            if partial_ratio_bound(read, cand) < _JOINT_BOUND_MIN:
                continue
            dist = weighted_distance(read, cand) / max(len(read), len(cand))
            if best is None or dist < best[0]:
                if best is not None and best[1] != cand:
                    runner = best
                best = (dist, cand)
            elif (runner is None or dist < runner[0]) and \
                    (best is None or cand != best[1]):
                runner = (dist, cand)
    if best is None or best[0] > _JOINT_MAX_NORM:
        return None
    if runner is not None and runner[0] - best[0] < _JOINT_MIN_MARGIN:
        return None
    return best[1]


def parse_flags(raw: str, flag_context: bool = False) -> tuple[set[str], bool]:
    """Parse an 'Observed flags' value into (flags, parsed_ok).

    parsed_ok is False when the value was non-empty but nothing in it could
    be confidently read as 'none' or a known flag — an unreadable flags line
    must never count as an affirmative clean read.

    flag_context=True marks text taken from the VALUE side of a matched
    'Observed flags' label; it licenses the truncation-prefix decode
    (MIB_SNAPFIX=1, _truncated_flag). Free-text callers keep the strict
    rule: without the label context a clipped prefix would invent flags
    out of ordinary words ('sponsor' -> sponsor_mismatch).
    """
    text = raw.strip().strip(".")
    if not text:
        return set(), False
    if match_vocab(text.lower(), ["none"], max_norm_dist=0.34):
        return set(), True
    flags: set[str] = set()
    for part in re.split(r"[,|;/]+", text):
        token = "_".join(p for p in re.split(r"[^a-z]+", part.lower()) if p)
        if not token:
            continue
        # flags are long snake_case tokens with distinctive shapes, so a
        # generous cut still resolves unambiguously under the margin test
        hit = match_vocab(token, RISK_FLAGS, max_norm_dist=0.35)
        if hit is None and flag_context and _snapfix_enabled():
            hit = _truncated_flag(token)
        if hit:
            flags.add(hit)
    return flags, bool(flags)


_FLAG_CANON = {f: f.replace("_", "") for f in RISK_FLAGS}


def _truncated_flag(token: str) -> str | None:
    """Unique-prefix decode of a column-clipped flag token (MIB_SNAPFIX).

    Truncation is a distinct failure from substitution and edit distance
    handles it badly: a scan clipped at the column edge yields 'resc' for
    rescinded_denial — ~0.75 normalized distance, hopeless against the
    0.35 cap, yet unambiguous. Licensed ONLY via parse_flags's
    flag_context (the value side of an Observed-flags label), where
    context already guarantees the token is a flag. Rule per the
    mib-intake reference (lexicon.py snap_flag, MIT — see ATTRIBUTION.md):
    a >= 6-char exact unique prefix accepts; a >= 3-char fuzzy prefix
    accepts at weighted-distance <= 1.0 with near-tie rejection (a
    confidently wrong flag is worse than none).
    """
    obs = token.replace("_", "")
    if len(obs) >= 6:
        matches = [f for f, canon in _FLAG_CANON.items()
                   if canon.startswith(obs)]
        if len(matches) == 1:
            return matches[0]
    if len(obs) >= 3:
        scored = sorted(
            (weighted_distance(obs, canon[:len(obs)]), f)
            for f, canon in _FLAG_CANON.items() if len(obs) <= len(canon)
        )
        scored = [s for s in scored if s[0] <= 1.0]
        if scored and (len(scored) == 1 or scored[1][0] - scored[0][0] >= 0.5):
            return scored[0][1]
    return None
