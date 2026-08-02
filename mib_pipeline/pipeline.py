"""Per-PDF orchestration: load -> filter -> OCR -> extract -> adjudicate."""
from __future__ import annotations

import os
import re
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path as FsPath

import cv2
import numpy as np
import pymupdf

from . import (crnn, decision, diagnostics, discharge, fields, ocr, policy,
               stamp_rescue, vocab, writer)
from .calibration import clamp_confidence, path_stats
from .fields import CaseEvidence
from .model import PageKind


def _grammar_name(value: str | None) -> bool:
    """True for a legal two-token name from the 12x24 syllable grammar."""
    tokens = str(value or "").split()
    return len(tokens) == 2 and all(t in vocab.NAME_TOKENS for t in tokens)

# --- Ship defaults for the env-gated escalation levers (esc-views-gate) ---
# The Docker image runs with NO custom environment, so these constants ARE
# the shipped behavior — the final-rebuild decision sets them. Env overrides
# exist for A/B work: MIB_ESC_VIEWS / MIB_ROT_PROBE ("1" forces on, "0"
# forces off, unset falls back to the constant).
#
# ESC_VIEWS: multi-view rotation-aware escalation (best_rot + 180 hedge +
# border pad). Off = the shipped image's single-raw-view behavior; the
# multi-view variant costs extra OCR passes on every escalated case and is
# the prime suspect for the 6.06-6.32 s/PDF docker_check breach (ship 5.39).
# ROT_PROBE: the A2 anchor-scored rotation probe (selection layer).
ESC_VIEWS_DEFAULT = False
ROT_PROBE_DEFAULT = False
# BGSUB: median-background-subtraction escalation variants appended to the
# ladder tail (see ocr.BGSUB_VARIANTS). Candidate lever from Arthur's
# 2026-07-30 wall review (MIB-000051 faint-under-overlay class); same lazy
# under-determined + soft-budget gating as the shipped variants.
BGSUB_DEFAULT = False
# NOTE_RESCUE: quarantined native-resolution re-read of the note header
# band (ocr.note_band_lines) feeding ONLY the N1 reason-template probe
# (fields.note_template_finding); rescue lines never join page.lines.
# Recovers a destroyed note's Reason sentence that the 288-DPI 2x upsample
# welds into unreadable micro-text (MIB-001000). Fires only on cases with
# no finding evidence of any kind; mints only via the MIB_REASON_ADJ=1
# template channel, strictly below every positive decision.
NOTE_RESCUE_DEFAULT = False


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


# Hard per-case wall-clock deadline, far above the observed per-case
# ceiling (~29 s on the heaviest train packet): a pathological input must
# degrade to the calibrated fallback row, never stall a worker.
CASE_DEADLINE_SECONDS = 180

# Soft budget for the OPTIONAL escalation engines (rapid/variants/CRNN/weld):
# past this, the case adjudicates with the evidence in hand. Prevents the
# stacked engines from driving a heavy packet into the hard deadline, where
# the fallback row would DISCARD already-won evidence (observed: 6 correct
# note-verdict cases lost to FALLBACK_error on the first combined run).
ESCALATION_SOFT_BUDGET_SECONDS = 70


@contextmanager
def _deadline(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _timeout(signum, frame):
        raise TimeoutError(f"case exceeded {seconds}s deadline")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


_CASE_ID_RE = re.compile(r"^MIB-[0-9]{6}$")


def _resolve_case_id(pdf_path: str) -> str:
    """Filename stem, or the packet's own footer id if the stem is invalid.

    The entrypoint only feeds MIB-######.pdf files, so this is belt and
    suspenders: a schema-invalid case_id would invalidate the whole row.
    """
    stem = FsPath(pdf_path).stem
    if _CASE_ID_RE.match(stem):
        return stem
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                m = fields.FOOTER_ID_RE.search(page.get_text())
                if m:
                    return m.group(1)
    except Exception:
        pass
    return stem


def process_pdf(pdf_path: str, engine: ocr.OcrEngine | None = None) -> dict:
    """Produce one schema-valid prediction row for one packet."""
    case_id = _resolve_case_id(pdf_path)
    try:
        with _deadline(CASE_DEADLINE_SECONDS):
            row = _process(pdf_path, engine or ocr.default_engine(), case_id)
        row["case_id"] = case_id
        return row
    except Exception:
        return _fallback_row(case_id)


def _under_determined(ev: CaseEvidence) -> bool:
    """True when a decision-relevant input is still unread."""
    visa = ev.value("visa_class")
    return (
        ev.value("fee_status") is None
        or not ev.flags_known
        or not ev.arrival_on_intake
        or visa is None
        or ev.value("home_world") is None
        or (visa != "DIP-1" and ev.value("sponsor_id") is None)
    )


# --- A2: anchor-scored rotation probe (rot-probe-integrator) ---
# A rotation-SELECTION layer for the escalation views, gated behind
# MIB_ROT_PROBE=1 (default OFF: with the flag unset every path below is
# dead code and the pipeline is byte-identical to the shipped build, so
# the pending A1 mini A/B stays uncontaminated). Idea from tylergibbs1
# 028ba78 (credited to naidx0) + kirtandesai's memo (both MIT; see
# ATTRIBUTION.md). Motivation: the load-time orientation estimator picks
# from a downscaled probe without anchor evidence and mis-selects on
# exactly the sparse pages that need escalation (19/20 B-band adjudication
# misses have >=1 rotated/unreadable page — research/14-miss-audit.md).

_ROT_PROBE_ORDER = (0, 1, 3, 2)     # upright first, 180 last (rarest bake-in)
_ROT_PROBE_SCALE = 0.5              # probe reads a half-resolution render

# Form-anchor vocabulary: label phrases every template prints. Word-bounded
# so FORM never matches inside PERFORMANCE nor ARRIVAL inside ARRIVALS.
_ROT_ANCHOR_RES = tuple(
    re.compile(r"\b" + phrase.replace(" ", r"\s+") + r"\b")
    for phrase in (
        "FORM", "APPLICANT", "SPECIES", "SPONSOR", "ARRIVAL", "FEE STATUS",
        "REGISTRY", "BIOMETRIC", "OBSERVED FLAGS", "FINDING",
    )
)

# The vector footer is stamped upright on every render, so it OCRs cleanly
# at whichever probe turn shows it right-side-up — scoring it would let a
# rotated page win on its readable footer. Phrase regex first, then any
# leftover footer tokens (OCR sometimes splits or partially reads the
# phrase), mirroring ocr._content_words.
_ROT_FOOTER_PHRASE_RE = re.compile(
    r"(?i)packet\s+mib[-\s.:]*[0-9]{6}\s*/?\s*page(\s+[0-9]+(\s+of\s+[0-9]+)?)?"
    r"|synthetic\s+hiring\s+challenge\s+document"
)
_ROT_FOOTER_WORDS = frozenset(
    "packet page synthetic hiring challenge document".split()
)
_ROT_CASE_ID_TOKEN_RE = re.compile(r"(?i)^mib[-\s.:]?[0-9]{6}$")

_ROT_TRIGGER_CHARS = 100   # trigger: footer-stripped text below this ...
_ROT_TRIGGER_ANCHORS = 2   # ... AND fewer than this many distinct anchors
_ROT_EXIT_ANCHORS = 2      # early-exit: this many anchors ...
_ROT_EXIT_CHARS = 80       # ... AND at least this much content
_ROT_FLOOR_CHARS = 40      # candidates below floor (and anchorless) are noise
_ROT_LEX_CAP = 20          # recognizable-word credit saturates here


def _build_rot_lexicon() -> frozenset[str]:
    """Lowercased words a genuine document orientation can read.

    Closed field vocabularies + template label words + the name-grammar
    tokens. Used for the probe's content score (kirtandesai's variant):
    ranking by recognizable WORDS instead of characters won the fixture
    benchmark 10/11 vs 5/11 — wrong rotations emit large volumes of
    confident junk characters, but junk almost never forms real words.
    """
    phrases = (
        list(vocab.SPECIES_CODES) + list(vocab.HOME_WORLDS)
        + list(vocab.VISA_CLASSES) + list(vocab.PURPOSES)
        + list(vocab.FEE_STATUSES) + list(vocab.RISK_FLAGS)
        + list(vocab.NAME_TOKENS)
        + ["FORM", "APPLICANT", "SPECIES", "SPONSOR", "ARRIVAL", "FEE STATUS",
           "REGISTRY", "BIOMETRIC", "OBSERVED FLAGS", "FINDING",
           "visa class", "home world", "purpose", "amount", "waiver",
           "status", "receipt", "intake", "attestation", "adjudicator",
           "manual note", "case", "name", "date", "entry", "denied",
           "approved", "review", "signature", "reason"]
    )
    words = set()
    for phrase in phrases:
        for tok in re.split(r"[^A-Za-z]+", str(phrase)):
            if len(tok) >= 3:
                words.add(tok.lower())
    return frozenset(words)


_ROT_LEXICON = _build_rot_lexicon()


@dataclass(frozen=True)
class _RotChoice:
    k: int                 # selected np.rot90 turn count
    confident: bool        # True when the probe early-exited on anchors


def _rot_probe_enabled() -> bool:
    return _env_flag("MIB_ROT_PROBE", ROT_PROBE_DEFAULT)


def _esc_views_enabled() -> bool:
    return _env_flag("MIB_ESC_VIEWS", ESC_VIEWS_DEFAULT)


def _note_rescue_enabled() -> bool:
    return _env_flag("MIB_NOTE_RESCUE", NOTE_RESCUE_DEFAULT)


def _strip_footer(text: str) -> str:
    text = _ROT_FOOTER_PHRASE_RE.sub(" ", text)
    kept = [
        tok for tok in text.split()
        if tok.lower() not in _ROT_FOOTER_WORDS
        and not _ROT_CASE_ID_TOKEN_RE.match(tok)
        and tok != "/"
    ]
    return " ".join(kept)


def _rot_anchor_count(text: str) -> int:
    upper = text.upper()
    return sum(1 for rx in _ROT_ANCHOR_RES if rx.search(upper))


def _rot_probe_trigger(lines) -> bool:
    """True when the primary ladder read essentially nothing off the page."""
    text = _strip_footer(" ".join(l.text for l in lines if l.conf > 0.5))
    return (len(text) < _ROT_TRIGGER_CHARS
            and _rot_anchor_count(text) < _ROT_TRIGGER_ANCHORS)


def _rot_candidate_stats(words) -> tuple[float, int, int]:
    """(score, anchors, chars) for one probed rotation's OCR words.

    Anchors dominate (2.0 each vs. a combined 2.0 ceiling for content and
    confidence): wrong rotations emit plenty of confident short junk, but
    only the true orientation reads label vocabulary. Content is counted
    in recognizable document WORDS, not characters (kirtandesai variant;
    chose it on fixture evidence — see _build_rot_lexicon). A candidate
    whose footer-stripped text is empty scores 0 — footer confidence
    alone is not evidence of anything.
    """
    kept = [w for w in words if w.conf > 0.5]
    text = _strip_footer(" ".join(w.text for w in kept))
    anchors = _rot_anchor_count(text)
    chars = len(text)
    if not chars:
        return 0.0, 0, 0
    hits = sum(1 for tok in re.split(r"[^A-Za-z]+", text)
               if len(tok) >= 3 and tok.lower() in _ROT_LEXICON)
    conf = float(np.mean([w.conf for w in kept]))
    score = 2.0 * anchors + min(hits, _ROT_LEX_CAP) / _ROT_LEX_CAP + conf
    return score, anchors, chars


def _probe_rotation(gray: np.ndarray, engine: ocr.OcrEngine) -> _RotChoice | None:
    """Anchor-score the np.rot90 turns of one page; None when inconclusive.

    One half-resolution OCR pass per turn in _ROT_PROBE_ORDER, early-exiting
    the moment a turn reads like a form (>=2 anchors and >=80 chars). Ties
    resolve to the earliest turn probed, so the selection is deterministic.
    """
    small = cv2.resize(gray, None, fx=_ROT_PROBE_SCALE, fy=_ROT_PROBE_SCALE,
                       interpolation=cv2.INTER_AREA)
    best: tuple[float, int] | None = None
    for k in _ROT_PROBE_ORDER:
        img = np.ascontiguousarray(np.rot90(small, k=k)) if k else small
        score, anchors, chars = _rot_candidate_stats(engine.words(img))
        if anchors >= _ROT_EXIT_ANCHORS and chars >= _ROT_EXIT_CHARS:
            return _RotChoice(k=k, confident=True)
        if anchors == 0 and chars < _ROT_FLOOR_CHARS:
            continue  # junk-level read; never worth selecting on
        if best is None or score > best[0]:
            best = (score, k)
    return None if best is None else _RotChoice(k=best[1], confident=False)


def _rot_probe_selections(scans, engine: ocr.OcrEngine,
                          budget_left) -> dict[int, _RotChoice]:
    """Probe every triggered scan page once, within the escalation budget.

    Trigger = near-empty page OR the load ladder had to estimate the
    orientation. The near-empty criterion alone (the published A2 recipe)
    fires on 1/51 rotation-family pages here, because unlike its source
    pipeline our ladder pools counter-rotation passes into result.lines,
    so even a sideways page usually carries >100 chars of pooled text.
    Estimator-run pages are exactly where the view selection is a guess —
    including pages the estimator scored upright that are not
    (MIB-000321 p3: upright-classified, actually k=1; the probe fixes the
    view where the legacy single-upright-view cannot).
    """
    if not _rot_probe_enabled():
        return {}
    selections: dict[int, _RotChoice] = {}
    for index, result in scans.items():
        if not budget_left():
            break
        if not (result.orientation_estimated
                or _rot_probe_trigger(result.lines)):
            continue
        choice = _probe_rotation(result.gray, engine)
        if choice is not None:
            selections[index] = choice
    return selections


def _escalation_views(result, probe: _RotChoice | None = None) -> list[np.ndarray]:
    """Images the escalation engines should read for one scan page.

    Upright pages: the render as-is. Non-upright pages: the estimated
    orientation AND its 180-degree complement, border-padded — feeding the
    sideways original produced guaranteed junk (MIB-000802: a legible
    'Finding: APPROVED' note missed because every escalation engine saw it
    sideways), and the orientation estimator picks the wrong direction
    often enough that the complement is a cheap hedge.

    When the A2 probe selected a rotation (MIB_ROT_PROBE=1, triggered page)
    its choice overrides the load-time estimate: a confident selection gets
    a single view (saving the hedge's OCR passes), an unconfident one keeps
    the 180-degree hedge.
    """
    if probe is not None:
        ks = (probe.k,) if probe.confident else (probe.k, (probe.k + 2) % 4)
        views = []
        for k in dict.fromkeys(ks):
            rot = np.ascontiguousarray(np.rot90(result.gray, k=k)) if k \
                else result.gray
            views.append(ocr.pad_for_ocr(rot))
        return views
    # MIB_ESC_VIEWS off: the shipped image's behavior — every escalation
    # engine reads the render as-is, sideways or not. The multi-view branch
    # below is what the gate buys (and what it costs in OCR passes).
    if result.upright or not _esc_views_enabled():
        return [result.gray]
    views = []
    for k in {result.best_rot or 1, ((result.best_rot or 1) + 2) % 4}:
        rot = np.ascontiguousarray(np.rot90(result.gray, k=k)) if k else result.gray
        views.append(ocr.pad_for_ocr(rot))
    return views


def _note_rescue_candidate(pages, scans, engine: ocr.OcrEngine,
                           budget_left) -> fields.Finding | None:
    """Agreed N1 candidate from quarantined note-band re-reads, or None.

    Each note-eligible scan page gets one native-resolution band re-read
    (ocr.note_band_lines) fed to the N1-only probe
    (fields.note_template_finding). The rescue lines are never attached to
    page.lines or any other consumer, so this pass can mint a
    reason-template candidate and nothing else. Two pages minting
    different labels are definitionally under-determined (a genuine packet
    carries exactly one adjudicator note): abstain outright.
    """
    candidate = None
    for index in sorted(scans):
        if not budget_left():
            break
        page = pages[index]
        if not fields.finding_eligible(page):
            continue
        lines = ocr.note_band_lines(scans[index].gray, index, engine)
        rescue = fields.note_template_finding(page, lines)
        if rescue is None:
            continue
        if candidate is not None and rescue.label != candidate.label:
            return None
        if candidate is None or rescue.conf > candidate.conf:
            candidate = rescue
    return candidate


def _process(pdf_path: str, engine: ocr.OcrEngine, case_id: str) -> dict:
    from .pdf_loader import load_pages

    started = time.monotonic()

    def _budget_left() -> bool:
        return time.monotonic() - started < ESCALATION_SOFT_BUDGET_SECONDS

    scans: dict[int, ocr.ScanOcrResult] = {}
    with pymupdf.open(pdf_path) as doc:
        pages = load_pages(doc)
        # Diagnostic-only stamp counter: logs to stderr/diag file, returns
        # None, mutates nothing — emitted rows must stay byte-identical.
        diagnostics.log_stamp_scan(doc, pages, case_id)
        for page in pages:
            if page.kind == PageKind.SCAN:
                gray = ocr.render_gray(doc[page.index], page)
                result = ocr.ocr_scan_page(gray, page.index, engine)
                page.lines = result.lines
                scans[page.index] = result

    candidates, flag_candidates, findings = fields.collect_candidates(pages, case_id)
    evidence = fields.reconcile(candidates, flag_candidates, findings)
    crnn_page_lines: dict[int, list] = {}

    # Second-opinion recognizer, only when the Tesseract ladder left a
    # decision-relevant input unread: PP-OCR (~1.2 s/page) survives damage
    # profiles that defeat Tesseract, and its lines rejoin the exact same
    # anchoring/vocabulary/reconciliation path — never the tier-1 logic.
    if scans and _under_determined(evidence) and _budget_left():
        for index, result in scans.items():
            extra = ocr.rapid_lines(result.gray, index)
            if not result.upright:
                for k in (1, 3):
                    rotated = np.ascontiguousarray(np.rot90(result.gray, k=k))
                    extra.extend(ocr.rapid_lines(rotated, index))
            # dedupe against the page's existing lines: the rotation passes
            # re-read the same rows, and duplicate candidates would
            # vote-sum a single fallback read past a better primary one
            best = {" ".join(l.text.lower().split()): l
                    for l in sorted(pages[index].lines + extra,
                                    key=lambda l: l.conf)}
            pages[index].lines = list(best.values())
        candidates, flag_candidates, findings = fields.collect_candidates(
            pages, case_id)
        evidence = fields.reconcile(candidates, flag_candidates, findings)
    # Preprocessing escalation, only when PP-OCR also left a decision input
    # unread: the four variants recover rows washed past both engines
    # (train: 3 receipts, 0 wrong verdicts on the whole R8 population).
    # Lazy — each variant runs only while an input is still unread; lines
    # rejoin the ordinary anchoring/vocabulary/reconciliation path.
    if scans and _under_determined(evidence) and _budget_left():
        pre = evidence
        # A2 rotation selections (MIB_ROT_PROBE=1 only; {} when the flag is
        # off, making every .get() below None = the shipped behavior).
        # Computed once per case so variants/CRNN/weld read the same views.
        rot_probe = _rot_probe_selections(scans, engine, _budget_left)
        ladder = ocr.ESCALATION_VARIANTS
        # MIB_BGSUB: "1" = raw ladder append; "2" = fill-only (a bgsub line
        # may resolve a still-unread field but can never outvote/flip a value
        # that was already read — kills the observed 619-style corruption
        # while keeping the 724-style recovery). Unset/0 = off.
        bgsub_mode = os.environ.get("MIB_BGSUB", "1" if BGSUB_DEFAULT else "0")
        if bgsub_mode in ("1", "2"):
            ladder = ladder + ocr.BGSUB_VARIANTS
        for variant in ladder:
            fill_only_guard = None
            if variant in ocr.BGSUB_VARIANTS and bgsub_mode == "2":
                fill_only_guard = (
                    {f: v for f, v in evidence.values.items() if v is not None},
                    dict(evidence.conf), set(evidence.flags), evidence.flags_known,
                )
            for index, result in scans.items():
                extra = []
                for view in _escalation_views(result, rot_probe.get(index)):
                    extra.extend(ocr.escalation_lines(view, index, variant,
                                                      engine))
                best = {" ".join(l.text.lower().split()): l
                        for l in sorted(pages[index].lines + extra,
                                        key=lambda l: l.conf)}
                pages[index].lines = list(best.values())
            candidates, flag_candidates, findings = fields.collect_candidates(
                pages, case_id)
            evidence = fields.reconcile(candidates, flag_candidates, findings)
            if fill_only_guard is not None:
                prev_values, prev_conf, prev_flags, prev_known = fill_only_guard
                for f, v in prev_values.items():
                    if evidence.values.get(f) != v:
                        evidence.values[f] = v
                        if f in prev_conf:
                            evidence.conf[f] = prev_conf[f]
                if prev_known and evidence.flags != prev_flags:
                    evidence.flags.clear()
                    evidence.flags.update(prev_flags)
            if not _under_determined(evidence) or not _budget_left():
                break
        # --- CRNN block (crnn-integrator; weld edits go elsewhere) ---
        # Candidate-trained CRNN, last of all engines: it only sees pages
        # everything else failed on, and its lines carry tier1_ok=False so
        # a misread can anchor a field candidate at worst, never a Finding.
        # The CRNN's measured value is flags rows and grammar-valid names
        # (v4 audit); firing it on every under-determined case costs ~7
        # CPU-s/case corpus-wide and broke the 6 s/PDF Docker budget
        # (measured 8.34). Gate it to the fields it actually recovers.
        need_flags = not evidence.flags_known
        need_name = not _grammar_name(evidence.value("applicant_name"))
        if (need_flags or need_name) and _budget_left():
            # Page-target the CRNN to where its value fields live: flags
            # print on the B-13 slip, names on intake/registry label rows.
            # Untyped pages (headers destroyed) stay eligible for both.
            wanted: set = set()
            if need_flags:
                wanted |= {"biometric"}
            if need_name:
                wanted |= {"intake", "registry"}
            # The strict sponsor muzzle lives at the SOURCE (crnn.py drops
            # sponsor-bearing lines before they exist): every later
            # reconcile pass — including the weld block's — stays clean of
            # CRNN sponsor evidence regardless of block ordering.
            for index, result in scans.items():
                if not _budget_left():
                    break
                if pages[index].doc_type is not None \
                        and pages[index].doc_type not in wanted:
                    continue
                extra = []
                for view in _escalation_views(result, rot_probe.get(index)):
                    extra.extend(crnn.crnn_lines(view, index))
                crnn_page_lines[index] = extra
                if not extra:
                    continue
                best = {" ".join(l.text.lower().split()): l
                        for l in sorted(pages[index].lines + extra,
                                        key=lambda l: l.conf)}
                pages[index].lines = list(best.values())
            candidates, flag_candidates, findings = fields.collect_candidates(
                pages, case_id)
            evidence = fields.reconcile(candidates, flag_candidates, findings)
        # Sponsor-only cut-strip weld (displaced-strip damage model, found by
        # human review): fires only while the sponsor is still unread; the
        # weld emits nothing but well-formed non-revoked SPN lines, so its
        # worst case is the field staying unread (never a minted R4 denial).
        if evidence.value("sponsor_id") is None and _budget_left():
            # The weld's whole measured value is a few sponsor fills; its
            # band scan must never dominate a case. 20 s allowance, also
            # bounded by the case's global soft budget.
            weld_deadline = min(started + ESCALATION_SOFT_BUDGET_SECONDS,
                                time.monotonic() + 20)
            for index, result in scans.items():
                if not _budget_left():
                    break
                extra = []
                for view in _escalation_views(result, rot_probe.get(index)):
                    extra.extend(ocr.weld_sponsor_lines(view, index, engine,
                                                        deadline=weld_deadline))
                if not extra:
                    continue
                best = {" ".join(l.text.lower().split()): l
                        for l in sorted(pages[index].lines + extra,
                                        key=lambda l: l.conf)}
                pages[index].lines = list(best.values())
            candidates, flag_candidates, findings = fields.collect_candidates(
                pages, case_id)
            evidence = fields.reconcile(candidates, flag_candidates, findings)
        # The escalation exists to FILL unread inputs; its aggressive
        # variants read junk on healthy rows, so anything the ordinary
        # ladders already read affirmatively is restored, not re-voted.
        for fld in fields.SCHEMA_FIELDS:
            if pre.known.get(fld):
                evidence.values[fld] = pre.values[fld]
                evidence.known[fld] = True
                evidence.conf[fld] = pre.conf[fld]
        if pre.flags_known:
            evidence.flags = pre.flags
            evidence.flags_known = True
        if pre.finding is not None or pre.finding_conflict:
            evidence.finding = pre.finding
            evidence.finding_conflict = pre.finding_conflict
    # Quarantined note-band rescue (MIB_NOTE_RESCUE=1, default OFF — flag
    # unset keeps this block dead code): when the case carries NO finding
    # evidence of any kind, re-read each note-eligible scan page's header
    # band at native resolution and let the N1-only probe mint a
    # reason-template candidate. Consumed by policy strictly below every
    # positive decision (the same template_finding channel as the ordinary
    # MIB_REASON_ADJ path); disagreeing pages abstain inside the helper.
    if (scans and _note_rescue_enabled() and evidence.finding is None
            and not evidence.finding_conflict
            and evidence.template_finding is None and _budget_left()):
        rescued = _note_rescue_candidate(pages, scans, engine, _budget_left)
        if rescued is not None:
            evidence.template_finding = rescued
    # Name-challenge, AFTER the restore guard and for applicant_name only:
    # a grammar-valid CRNN name may replace a pre-known read ONLY when that
    # read fails the 12x24 name grammar (i.e. no engine produced a legal
    # name). Names carry zero classification signal, so this arbitration is
    # adjudication-inert by construction; no other field gets a challenge.
    if crnn_page_lines and evidence.known.get("applicant_name") and \
            not _grammar_name(evidence.values.get("applicant_name")):
        challengers = []
        for ls in crnn_page_lines.values():
            for line in ls:
                if "applicant" not in line.text.lower() or ":" not in line.text:
                    continue
                fixed = vocab.correct_name(line.text.split(":", 1)[1].strip())
                if fixed and _grammar_name(fixed):
                    challengers.append((line.conf, fixed))
        if challengers:
            conf, best_name = max(challengers)
            evidence.values["applicant_name"] = best_name
            evidence.conf["applicant_name"] = conf
    # Hard-embargo home worlds carry the planetary_embargo flag in every
    # observed truth row (50/50 corpus-wide), whether or not the packet
    # prints it — inferring it recovers extraction on silent-flag packets.
    if evidence.value("home_world") in policy.HARD_EMBARGO_WORLDS:
        evidence.flags.add("planetary_embargo")
    policy_label, path = policy.adjudicate(evidence)
    # Structural discharge heads (MIB_DISCHARGE=1, default OFF — flag unset
    # keeps this block dead code): one targeted re-read of the review path's
    # blocking evidence item, then the unchanged cascade re-runs. Lives
    # inside the same escalation soft budget as every optional engine.
    fired = None
    if scans and discharge.any_enabled() and _budget_left():
        fired = discharge.run_discharge(
            pdf_path=pdf_path, case_id=case_id, pages=pages, scans=scans,
            evidence=evidence, path=path, engine=engine,
            budget_left=_budget_left)
        if fired is not None:
            policy_label, path = policy.adjudicate(evidence)
    adjudication, confidence = decision.decide(policy_label, path)
    if fired is not None:
        confidence = min(confidence, discharge.CONF_CAP)
    # Green-stamp rescue (MIB_STAMP_RESCUE=1, default OFF — flag unset
    # keeps this block dead code): a detected green APPROVED stamp promotes
    # a could-not-read review to APPROVED, strictly below every deny path
    # (precedence proof in stamp_rescue.py). The scan doubles as the
    # review-case diagnostic sweep, which is then skipped below.
    rescue_scanned = False
    if (stamp_rescue.enabled() and adjudication == policy.NEEDS_REVIEW
            and stamp_rescue.eligible(evidence, path)):
        stamped = diagnostics.log_stamp_scan_review(pdf_path, case_id)
        rescue_scanned = True
        if stamped:
            adjudication, confidence = policy.APPROVED, stamp_rescue.CONF
            path = stamp_rescue.PATH
            diagnostics.log_rescue(case_id, stamped)
    row = writer.build_row(case_id, evidence, adjudication, confidence)
    # Diagnostic-only stamp sweep of review cases (the row is already
    # built; scan-page images are color-capable, see diagnostics.py).
    if adjudication == "NEEDS_REVIEW" and not rescue_scanned:
        diagnostics.log_stamp_scan_review(pdf_path, case_id)
    row["_path"] = path  # stripped before writing; used by dev tooling
    if fired is not None:
        row["_discharge"] = fired.as_dict()  # stripped like _path
    return row


def _fallback_row(case_id: str) -> dict:
    stats = path_stats("FALLBACK_error")
    row = writer.build_row(
        case_id, CaseEvidence(), "NEEDS_REVIEW",
        clamp_confidence(stats.accuracy),
    )
    row["_path"] = "FALLBACK_error"
    return row
