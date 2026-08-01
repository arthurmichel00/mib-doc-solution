"""Closed vocabularies and OCR-aware fuzzy matching.

All enumerations below are field vocabularies mined from the public
FIELD_MANUAL.md and the distribution of values in data/train_labels.csv.
They are vocabularies of the document generator, not per-case answers.
"""
from __future__ import annotations

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


def parse_flags(raw: str) -> tuple[set[str], bool]:
    """Parse an 'Observed flags' value into (flags, parsed_ok).

    parsed_ok is False when the value was non-empty but nothing in it could
    be confidently read as 'none' or a known flag — an unreadable flags line
    must never count as an affirmative clean read.
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
        if hit:
            flags.add(hit)
    return flags, bool(flags)
