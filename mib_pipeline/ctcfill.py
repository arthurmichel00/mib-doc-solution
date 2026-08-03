"""Closed-menu CTC fill (MIB_CTCFILL=1, default OFF): score every legal
menu candidate directly against the bundled PP-OCRv6 recognizer's frame
posteriors with the exact CTC forward algorithm, instead of trusting an
argmax decode that the damage already defeated.

Mechanism adapted from the MIT-licensed "moonshots" public solution
(mib/ctcscore.py: exact CTC forward P(candidate | strip) over PP-OCR rec
posteriors; see ATTRIBUTION.md). Adaptations for this pipeline:

- PP-OCRv6 rec.onnx (the second-opinion recognizer we already bundle).
  v6 ships NO ``character`` metadata (their v5 did): the head charset is
  rebuilt from models/ppocrv6/rec_keys.txt as [blank] + keys + [space],
  verified against the model's 18710-dim output at load.
- v6 exports a dynamic input width; strips keep their aspect ratio up to
  640 px instead of being squeezed into the reference's fixed 320.
- Candidate strings are LABEL-INCLUSIVE ("Home World: Titan Freeport"):
  the strip is cropped from the label's own box, so a mislocated strip
  fails the shared label prefix and every candidate scores low — the
  scorer double-checks the locator.
- The locator is the OCR ladder's own word-box geometry FIRST
  (ScanOcrResult.words — passes the pipeline already runs, zero extra
  cost). Only when no pass ever read the field's label — the deep-damage
  slice where every recognizer goes blind but pixel correlation still
  sees the form — a slimmed single-size intake-grid fit (adapted from
  our menu-sweep locator, ~6x cheaper than the swept version that was
  killed on cost) locates the row on intake-typed/untyped pages. Both
  paths feed the same label-inclusive scorer, whose shared label prefix
  is itself a locator check.

Fill-only contract (enforced in pipeline.py, tested in test_ctcfill.py):
fires only on decision-relevant closed-menu fields that are STILL UNREAD
after both engines and the whole escalation ladder + restore guard; never
fee_status (79% mode base rate, measured no-win), never sponsor digits
(measured chance-level), never anything feeding Finding/notes. Accepted
values are extraction candidates only: confidence is capped below the
affirmative-read threshold (fields._KNOWN_MIN_OCR_CONF), so ``known``
stays False and no policy rule can consume them. Hard-embargo worlds are
never accepted — a reconstructed read must not feed the pipeline's
planetary_embargo inference (same principle as the weld's revoked-sponsor
drop).

Second-resolution margin gates (MIB_CTCFILL_MARGIN=1, default OFF,
layered on MIB_CTCFILL=1) add a purely restrictive confirmation pass on
top of all of the above: see the SECOND_RES_* / MARGIN2_FLOOR block.

Likelihood fusion (MIB_CTCFILL_FUSION=1, default OFF) is layered on top
as additive helpers at the bottom of this module: instead of gating each
preprocessing variant / page independently and vetoing on disagreement,
it AVERAGES the per-candidate log-likelihoods across every view of the
field and gates the fused list once — how likely is a candidate under
ALL the evidence, not under one witness. Zero extra inference: the same
posteriorgrams the flag-off path already computes. Flag unset keeps
every code path below byte-identical.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

from . import row_restore, vocab
from .ocr import RENDER_DPI, ScanOcrResult, _models_dir

# ---------------------------------------------------------------- fields
# Decision-relevant closed menus only. fee_status and sponsor_id are
# excluded BY CONSTRUCTION (see module docstring); risk flags / findings
# are never touched — the lever emits field values, not tier-1 evidence.
FIELDS = ("species_code", "home_world", "visa_class", "declared_purpose")

_MENUS = {
    "species_code": tuple(vocab.SPECIES_CODES),
    "home_world": tuple(vocab.HOME_WORLDS),
    "visa_class": tuple(vocab.VISA_CLASSES),
    "declared_purpose": tuple(vocab.PURPOSES),
}

# Printed label token sequences that anchor each field's value strip.
# "Species Match" (biometric) is excluded: its value row carries extra
# glyphs the menu strings do not model. Single-token "Purpose:" (the
# attestation letter's label) is allowed — the label-inclusive prefix
# check keeps a mislocated hit from minting anything.
_LABELS: dict[str, tuple[tuple[str, ...], ...]] = {
    "species_code": (("Species", "Code"),),
    "home_world": (("Home", "World"),),
    "visa_class": (("Visa", "Class"),),
    "declared_purpose": (("Declared", "Purpose"), ("Purpose",)),
}

# Values the fill must never emit: pipeline.py infers planetary_embargo
# from evidence.value("home_world") regardless of `known`, so a wrong
# hard-embargo fill could mint an R1 denial. Worst case here = the field
# stays unread, exactly as it was.
_EXCLUDED_VALUES = frozenset({"TRAPPIST-1e", "Eris Relay"})

# ------------------------------------------------------------------ gates
# Locator: every matched label token needs this word confidence.
LOC_MIN_CONF = 0.60
# Acceptance, calibrated on the menu-sweep population (96 slots + gold;
# see ctc-branch-VALIDATION.md; started from moonshots' flagread floor
# -3.5 / margin 2.0 nats and re-derived on our dev data): a
# length-normalized log-prob floor for the winner, a margin over the
# runner-up candidate, and a margin over the null hypothesis (label with
# no legible value printed). The grid-fallback path gets its own floor /
# margin — its native-scale strips score systematically lower than the
# word-box path's 288-DPI crops.
SCORE_FLOOR = -3.5
MARGIN_RUNNER_UP = 0.30
MARGIN_NONE = 0.00
GRID_SCORE_FLOOR = -4.0
GRID_MARGIN_RUNNER_UP = 0.20
# Extraction-only confidence cap: strictly below the affirmative-read
# threshold (fields._KNOWN_MIN_OCR_CONF = 0.55), mirroring the weld cap —
# a CTC fill can populate the output row but never sets known=True.
CONF_CAP = 0.50

# ----------------------------------------- second-resolution margin gates
# MIB_CTCFILL_MARGIN=1 (default OFF), layered on MIB_CTCFILL=1. Idea from
# the MIT-licensed ShreyShingala public solution (mib/row_restore.py:
# choose_equal_length returns the best-minus-runner-up margin as the
# confidence signal, and consensus() accepts a value only when two
# independently rendered views agree on it and both clear per-field
# floors; see ATTRIBUTION.md). Purely restrictive here: the primary
# resolution's gate is untouched and runs first, so every fill this layer
# admits is one the shipped lever already admits — it can only remove.
#
# "Second resolution" is a second rec-model input resize of the SAME
# crop. The reference re-rendered the page at two DPIs; our strips are
# cut from one 288-DPI render and re-rendering a page costs orders of
# magnitude more than a second rec pass, so the two views are two
# _preprocess widths. Only a DOWN-scale carries information: a word-box
# strip (680 px wide) saturates the _REC_MAX_W = 640 dynamic-width cap,
# so an up-scale is clamped straight back to 640 and hands back the
# primary view unchanged.
#
# 0.80 is the measured optimum of the trade curve below over 0.60-0.85:
# scale 0.60 removes 4 of the 5 wrong fires but gives up 7.9% of the
# right ones, which on the field audit's ~10:1 right:wrong population is
# break-even at best, while 0.80 gives up 2.0% for 40% of the wrong
# fires — a bounded downside (~0.6 fields if it removes nothing real)
# for a comparable upside.
SECOND_RES_SCALE = 0.80
# The absolute log-prob-per-char scale shifts DOWN at the smaller resize
# (fewer CTC frames over the same string). Re-applying the primary
# SCORE_FLOOR unshifted would reject on that resize artifact rather than
# on evidence — at scale 0.60 it caused 19 of 48 rejections, none of them
# wrong fires. At 0.80 the measured shift over accepted strips is mean
# -0.007, p05 -0.29, min -0.53 nats/char, so one nat of slack clears the
# worst case about twice over and this floor does NOT bind on the
# calibration population (removing it entirely reproduces the same
# verdicts). It stands as a guard rail against a degenerate second view,
# not as a working gate; test_ctcfill_margin.py pins that it still fires.
SECOND_RES_FLOOR_SLACK = 1.0
# Cost. ONE extra rec pass per strip the primary gate ACCEPTS — not per
# strip examined, and not per PDF: measured 52-56 ms here against
# 131-139 ms for the primary pass's two variants. The shipped lever fills
# ~40 fields over the 1000-PDF train corpus, so even at the
# _MAX_STRIPS_PER_FIELD ceiling of accepted strips per fill the average
# sits near 2-6 ms/PDF; it would take 0.36 accepted strips per PDF, an
# order of magnitude above the measured fill rate, to reach 20 ms/PDF.

# PER-FIELD RUNNER-UP MARGIN FLOORS, second resolution only (nats/char).
#
# Scale. score_strip divides the CTC log-prob of the whole
# label-inclusive candidate by its character count, so a margin is nats
# of evidence PER CHARACTER and the total evidence separating winner from
# runner-up is margin x L. Mean label-inclusive length differs 1.7x
# across the four menus (visa_class 17.4, home_world 23.2, species_code
# 26.5, declared_purpose 30.2), so the single global MARGIN_RUNNER_UP
# demands only 5.2 total nats on visa_class against 9.1 on
# declared_purpose — and visa_class is exactly the menu whose closest
# pair is ONE glyph apart (XW-1/XW-2; every other menu's closest pair is
# >= 5 edits). Each floor below is MARGIN_RUNNER_UP x L_max / L_field:
# the same TOTAL evidence the global floor already demands of the longest
# field, asked of every field. The longest field is therefore unchanged
# and every floor is >= MARGIN_RUNNER_UP by construction, so this can
# only restrict.
#
# Level. Calibrated on 1120 synthetic strips (four fields x every menu
# value x 14 damage conditions x both variants) scored through the real
# rec posteriors: the primary path accepts 394 right / 5 wrong, this
# layer keeps 386 / 3 — precision 0.9875 -> 0.9923, giving up 2.0% of the
# right fires to remove 40% of the wrong ones. Scaling these floors up
# (1.5x, 2x, 3x were measured) removes at most one further wrong fire and
# costs 8-55 more right ones, so the equalized level is kept as derived
# rather than tuned up. Tightening the PRIMARY margin instead measured
# net negative (4 right lost, 0 wrong removed): the wrong fires are NOT
# near-ties — their margins run 0.44-1.36, inside the bulk of the right
# fires — they are occlusion prefix fits, where a shorter candidate
# explains the ink that survived, and only the change of resolution
# breaks them.
MARGIN2_FLOOR = {
    "species_code":     0.34,
    "home_world":       0.39,
    "visa_class":       0.52,
    "declared_purpose": 0.30,
}
# The grid path reads coarser native-scale strips and already runs a
# lower base margin; hold its second-resolution floors at the same ratio
# to that base (GRID_MARGIN_RUNNER_UP / MARGIN_RUNNER_UP = 2/3) instead
# of importing the word-box levels wholesale.
GRID_MARGIN2_FLOOR = {
    "species_code":     0.23,
    "home_world":       0.26,
    "visa_class":       0.35,
    "declared_purpose": 0.20,
}

# Strip geometry (in the 288-DPI render space of ScanOcrResult.gray):
# label box top/bottom padding and value-region width from the label's
# left edge — 680 px = the menu-sweep's 340 px native row window at 2x.
_STRIP_Y_PAD = 6
_STRIP_W = 680
_MAX_STRIPS_PER_FIELD = 3

# --- grid-fallback locator (native embedded-image scale = 0.5x render) ---
# The intake template prints its eight label rows on a fixed-pitch grid
# (measured 31.22 +- 0.46 px across 158 cleanly-OCR'd pages). A joint fit
# of (x0, y0, pitch) over all eight labels is far harder to fool than one
# label alone. Slimmed against the menu-sweep version: ONE template size
# (12, the modal choice on 67/96 located slots) x two dilations instead
# of 3x2, everything else identical.
_GRID_LABELS = ("Case ID:", "Applicant:", "Species Code:", "Home World:",
                "Visa Class:", "Sponsor ID:", "Arrival Date:",
                "Declared Purpose:")
_GRID_ROW_OF = {"species_code": 2, "home_world": 3, "visa_class": 4,
                "declared_purpose": 7}
_GRID_PITCHES = (30.8, 31.2, 31.6)
_GRID_SIZE = 12
_GRID_DILS = (3, 4)
_GRID_BLOCK = (40, 480, 40, 470)     # x0, x1, y0, y1 native search window
_GRID_MIN_NCC = 0.35                 # junk-page floor; the CTC label
                                     # prefix is the real locator check
_GRID_ROW_W = 346                    # native row window width (label+value)

_REC_H = 48                 # PP-OCR rec input height
_REC_MAX_W = 640            # dynamic-width cap (2x the reference's 320)
_NEG = -1e9

_SESSION = None
_CHARS: list[str] | None = None
_CHAR_TO_IX: dict[str, int] | None = None


def _session():
    """Lazy per-process ONNX session + charset; single-threaded (the
    pipeline's CPU discipline). Raises if the bundle is absent — callers
    that must not fail use available()."""
    global _SESSION, _CHARS, _CHAR_TO_IX
    if _SESSION is None:
        import onnxruntime as ort

        base = _models_dir() / "ppocrv6"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(str(base / "rec.onnx"), so,
                                       providers=["CPUExecutionProvider"])
        keys = [line.rstrip("\n")
                for line in open(base / "rec_keys.txt", encoding="utf-8")]
        chars = ["<blank>"] + keys + [" "]
        # Verify the assumed CTC head layout against the exported model:
        # v6's output is (N, T, len(keys) + 2) = blank + dict + space.
        out_dim = session.get_outputs()[0].shape[-1]
        if isinstance(out_dim, int) and out_dim != len(chars):
            raise RuntimeError(
                f"rec.onnx head dim {out_dim} != charset {len(chars)}")
        _SESSION, _CHARS, _CHAR_TO_IX = session, chars, {
            c: i for i, c in enumerate(chars)}
    return _SESSION


def _margin_gates() -> bool:
    """Both flags, read together: MIB_CTCFILL_MARGIN=1 layers on top of
    MIB_CTCFILL=1 and neither alone does anything. One reader for the
    pair keeps the layering a single fact rather than two that can drift
    apart. Unset (the default) leaves every path below byte-identical to
    the shipped lever — that is the A/B contract."""
    return (os.environ.get("MIB_CTCFILL") == "1"
            and os.environ.get("MIB_CTCFILL_MARGIN") == "1")


def available() -> bool:
    """True when the rec bundle loads (models are not shipped to every
    dev checkout); the pipeline block abstains quietly when False."""
    try:
        _session()
        return True
    except Exception:
        return False


def _preprocess(gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Grayscale strip -> the rec model's (1, 3, 48, W) normalized tensor.
    PP-OCR rec normalization is (x/255 - 0.5) / 0.5 across generations.

    `scale` != 1.0 rescales the width the default view WOULD use, not the
    strip's raw aspect width: a 680-px word-box strip already saturates
    _REC_MAX_W, so scaling the aspect width first would hand the second
    view the identical 640 px and make the consensus vacuous exactly on
    the widest strips. Scaling the clamped width instead keeps the two
    views a fixed ratio apart on every strip. Scale 1.0 skips the branch
    outright, so the default path is untouched.
    """
    h, w = gray.shape
    tw = min(_REC_MAX_W, max(16, int(round(_REC_H * w / h))))
    if scale != 1.0:
        tw = min(_REC_MAX_W, max(16, int(round(tw * scale))))
    img = cv2.resize(gray, (tw, _REC_H), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.repeat(x[None, None], 3, axis=1)


def posteriorgram(gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """(T, C) log-probability frames for one strip at one rec resize."""
    sess = _session()
    y = sess.run(None,
                 {sess.get_inputs()[0].name: _preprocess(gray, scale)})[0][0]
    return np.log(np.maximum(y, 1e-12))


@lru_cache(maxsize=4096)
def _label_ixs(text: str) -> tuple[int, ...] | None:
    """Candidate string -> class indices, or None if out-of-dict."""
    _session()
    out = []
    for ch in text:
        ix = _CHAR_TO_IX.get(ch)
        if ix is None:
            return None
        out.append(ix)
    return tuple(out)


def _ctc_logp(logp: np.ndarray, labels: tuple[int, ...]) -> float:
    """Log P(labels | frames) by the CTC forward recursion (vectorized
    over the extended label sequence)."""
    T = logp.shape[0]
    ext = [0]
    for l in labels:
        ext += [l, 0]
    S = len(ext)
    if S > 2 * T + 1:
        return _NEG
    ext = np.asarray(ext)
    allow_skip = np.zeros(S, bool)
    allow_skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    alpha = np.full(S, _NEG)
    alpha[0] = logp[0, 0]
    if S > 1:
        alpha[1] = logp[0, ext[1]]
    for t in range(1, T):
        shift1 = np.concatenate(([_NEG], alpha[:-1]))
        best = np.logaddexp(alpha, shift1)
        if S > 2:
            shift2 = np.concatenate(([_NEG, _NEG], alpha[:-2]))
            best = np.where(allow_skip, np.logaddexp(best, shift2), best)
        alpha = best + logp[t, ext]
    tail = alpha[-1] if S == 1 else np.logaddexp(alpha[-1], alpha[-2])
    return float(tail)


def score_strip(strip: np.ndarray, label_text: str,
                values: tuple[str, ...],
                scale: float = 1.0) -> list[tuple[float, str]]:
    """Rank menu values by length-normalized CTC log-prob of the
    label-inclusive candidate string; "<none>" = the label printed with
    no legible value (the null hypothesis). Best first."""
    logp = posteriorgram(strip, scale)
    out = []
    for value in list(values) + ["<none>"]:
        text = label_text if value == "<none>" else f"{label_text} {value}"
        ixs = _label_ixs(text)
        if ixs is None:
            continue
        lp = _ctc_logp(logp, ixs)
        out.append((lp / max(1, len(ixs)), value))
    out.sort(reverse=True)
    return out


# ---------------------------------------------------------------- locator

def _clean_token(text: str) -> str:
    return text.strip().strip("|:;.,'\"`!‘’“”[](){}").lower()


def locate_strips(result: ScanOcrResult, field: str
                  ) -> list[tuple[np.ndarray, str, float]]:
    """Label-anchored value strips for one field on one scan page.

    Scans the page's stashed word boxes (ScanOcrResult.words — geometry
    the ladder already produced, no new OCR pass) for the field's printed
    label token run: tokens must match exactly after punctuation
    stripping, sit on one row, read left-to-right with a sane gap, and
    each carry conf >= LOC_MIN_CONF — the locator-confidence gate the
    menu-sweep lesson demands. Multiple passes re-reading the same label
    are deduped by row. Returns [(strip, label_text, locator_conf)];
    empty list = abstain (no full-page fallback search by design).
    """
    # No upright gate: the stash was read on the UNROTATED render, so a
    # confident label token run found there is genuine upright content
    # even when the (documented-unreliable on sparse pages) orientation
    # estimator guessed a rotation.
    words = result.words or []
    if not words:
        return []
    gray = result.gray
    hits: dict[int, tuple[float, tuple]] = {}
    for pattern in _LABELS[field]:
        n = len(pattern)
        label_text = " ".join(pattern) + ":"
        for i in range(len(words) - n + 1):
            run = words[i:i + n]
            if any(w.h <= 0 or w.w <= 0 for w in run):
                continue    # legacy stash without geometry
            if any(_clean_token(w.text) != t.lower()
                   for w, t in zip(run, pattern)):
                continue
            if any(w.conf < LOC_MIN_CONF for w in run):
                continue
            ys = [w.y for w in run]
            hs = [w.h for w in run]
            if max(ys) - min(ys) > max(hs):
                continue    # not one row
            ok = True
            for a, b in zip(run, run[1:]):
                if not (a.x + a.w * 0.5 < b.x < a.x + a.w + 3.0 * a.h):
                    ok = False
                    break
            if not ok:
                continue
            conf = min(w.conf for w in run)
            y0 = int(max(0, min(ys) - _STRIP_Y_PAD))
            y1 = int(min(gray.shape[0], max(w.y + w.h for w in run)
                         + _STRIP_Y_PAD))
            x0 = int(max(0, run[0].x - 2))
            x1 = int(min(gray.shape[1], run[0].x + _STRIP_W))
            if y1 - y0 < 8 or x1 - x0 < 60:
                continue
            key = (y0 + y1) // 24    # dedupe re-reads of the same row
            held = hits.get(key)
            if held is None or conf > held[0]:
                hits[key] = (conf, (y0, y1, x0, x1, label_text))
    out = []
    for conf, (y0, y1, x0, x1, label_text) in sorted(
            hits.values(), reverse=True)[:_MAX_STRIPS_PER_FIELD]:
        strip = np.ascontiguousarray(gray[y0:y1, x0:x1])
        out.append((strip, label_text, conf))
    return out


# ------------------------------------------------- grid-fallback locator

_TMPL_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _label_tmpl(text: str, dil: int) -> np.ndarray:
    """Helvetica-Bold glyphs as ink on white, tight-cropped and bloated —
    the generator's own font family, so the template is a faithful
    un-degraded version of the page's label ink (menu-sweep renderer)."""
    key = (text, dil)
    if key in _TMPL_CACHE:
        return _TMPL_CACHE[key]
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=2000, height=200)
    page.insert_text((20, 120), text, fontsize=_GRID_SIZE, fontname="hebo")
    pm = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pm.samples, np.uint8).reshape(
        pm.height, pm.width).copy()
    doc.close()
    ink = a < 200
    ys, xs = np.nonzero(ink)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if dil > 0:
        p = dil + 2
        a = cv2.copyMakeBorder(a, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
        m = (a < 200).astype(np.uint8) * 255
        m = cv2.dilate(m, np.ones((2, 2), np.uint8), iterations=dil)
        a = 255 - m
        ink = a < 200
        ys, xs = np.nonzero(ink)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _TMPL_CACHE[key] = a
    return a


def _grid_fit(native: np.ndarray):
    """Joint 8-label intake-grid fit on one native-scale page.

    Returns (mean_ncc, x0, y0, pitch, label_h) or None. A page that is
    not an intake form scores low everywhere because no single
    (x0, y0, pitch) explains eight labels at once.
    """
    x0b, x1b, y0b, y1b = _GRID_BLOCK
    blk = np.ascontiguousarray(native[y0b:y1b, x0b:x1b])
    if blk.shape[0] < 60 or blk.shape[1] < 120:
        return None
    best = None
    for dil in _GRID_DILS:
        maps = []
        ok = True
        for lab in _GRID_LABELS:
            t = _label_tmpl(lab, dil)
            if t.shape[0] >= blk.shape[0] or t.shape[1] >= blk.shape[1]:
                ok = False
                break
            maps.append(cv2.matchTemplate(blk, t, cv2.TM_CCOEFF_NORMED))
        if not ok:
            continue
        H = min(m.shape[0] for m in maps)
        W = min(m.shape[1] for m in maps)
        stack = np.stack([m[:H, :W] for m in maps])
        for pitch in _GRID_PITCHES:
            ymax = H - int(round(7 * pitch)) - 1
            if ymax <= 0:
                continue
            offs = [int(round(k * pitch)) for k in range(8)]
            acc = np.zeros((ymax, W))
            for k, off in enumerate(offs):
                acc += stack[k, off:off + ymax, :]
            acc /= 8.0
            iy, ix = np.unravel_index(int(acc.argmax()), acc.shape)
            sc = float(acc[iy, ix])
            if best is None or sc > best[0]:
                best = (sc, x0b + int(ix), y0b + int(iy), pitch,
                        _label_tmpl(_GRID_LABELS[0], dil).shape[0])
    if best is None or best[0] < _GRID_MIN_NCC:
        return None
    return best


def _main_blob(row: np.ndarray) -> tuple[int, int]:
    """Widest ink run of the row (menu-sweep row trimming): drops table
    borders and stray marks left/right of the field text."""
    prof = (row < 200).sum(0)
    runs, s, inside = [], 0, False
    for i, v in enumerate(prof):
        if v > 0 and not inside:
            s, inside = i, True
        elif v == 0 and inside:
            runs.append((s, i))
            inside = False
    if inside:
        runs.append((s, len(prof)))
    merged, cur = [], None
    for a, b in runs:
        if cur and a - cur[1] < 6:
            cur = (cur[0], b)
        else:
            if cur:
                merged.append(cur)
            cur = (a, b)
    if cur:
        merged.append(cur)
    if not merged:
        return 0, row.shape[1]
    return max(merged, key=lambda ab: ab[1] - ab[0])


def grid_strips(result: ScanOcrResult, field: str,
                fit=None) -> list[tuple[np.ndarray, str, float]]:
    """Grid-fallback value strip for one field on one page, native scale.

    Used ONLY when no OCR pass ever produced the field's label word boxes
    (locate_strips returned nothing on every page): the deep-damage slice
    where recognition is blind but the form's fixed geometry still shows.
    Same return shape as locate_strips; locator_conf is the grid's mean
    NCC (a joint 8-label fit, so a confident value is hard to fake). The
    fit always reads the unrotated render: on a genuinely rotated page it
    scores below _GRID_MIN_NCC and abstains, whereas trusting the sparse
    orientation estimate would skip exactly the destroyed-intake pages
    this fallback exists for.
    """
    if fit is None:
        native = cv2.resize(result.gray, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_AREA)
        fit = _grid_fit(native)
        if fit is None:
            return []
    else:
        native = cv2.resize(result.gray, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_AREA)
    ncc, gx, gy, pitch, lab_h = fit
    k = _GRID_ROW_OF[field]
    y_top = gy + int(round(k * pitch))
    ya, yb = y_top - 3, y_top + lab_h + 3
    xa = max(0, gx - 6)
    xb = min(native.shape[1], gx + _GRID_ROW_W)
    if ya < 0 or yb > native.shape[0] or xb - xa < 40:
        return []
    row = native[ya:yb, xa:xb]
    s, e = _main_blob(row)
    y0, y1 = max(0, ya - 4), min(native.shape[0], yb + 4)
    x0, x1 = max(0, xa + s - 8), min(native.shape[1], xa + e + 8)
    if y1 - y0 < 8 or x1 - x0 < 30:
        return []
    strip = np.ascontiguousarray(native[y0:y1, x0:x1])
    label_text = _GRID_LABELS[k]
    return [(strip, label_text, ncc)]


def page_grid_fit(result: ScanOcrResult):
    """One cached-by-caller grid fit for a page (serves all four fields)."""
    native = cv2.resize(result.gray, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    return _grid_fit(native)


# ------------------------------------------------------------- acceptance

def _variants(strip: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    """The two cheap views scored per strip: the raw crop and a
    contrast-stretched one (faint washed ink sits a few grey levels under
    paper; MINMAX stretch is the flagread-verified lift)."""
    stretched = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX)
    return (("raw", strip), ("stretch", stretched))


def _score_margin(scored: list[tuple[float, str]]
                  ) -> tuple[str | None, float, float, float]:
    """(winner, top score, runner-up margin, margin over <none>) for one
    scored variant — the winner carried together with the confidence
    signal that qualifies it. The runner-up margin ignores "<none>"
    (which the separate null-hypothesis margin covers) and is +inf when
    the menu holds no other candidate."""
    if not scored:
        return None, _NEG, float("inf"), float("inf")
    by = {value: s for s, value in scored}
    top_score, winner = scored[0]
    others = [s for s, v in scored if v not in (winner, "<none>")]
    margin = top_score - max(others) if others else float("inf")
    return winner, top_score, margin, top_score - by.get("<none>", _NEG)


def _gated_winner(scored: list[tuple[float, str]],
                  floor: float = SCORE_FLOOR,
                  m_ru: float = MARGIN_RUNNER_UP) -> str | None:
    """Apply the calibrated acceptance gate to one scored variant."""
    winner, top_score, margin, m_none = _score_margin(scored)
    if winner is None or winner == "<none>":
        return None
    if top_score < floor or margin < m_ru or m_none < MARGIN_NONE:
        return None
    return winner


def _mode_default(field: str) -> str:
    """The writer's train-mode fallback for the field: emitting it is a
    no-op (the output row prints it anyway when the field is unread), and
    default-accepts were the ONLY wrong-fire mode the calibration
    measured — so the fill never emits it."""
    from . import writer

    return writer._DEFAULTS[field]


def _strip_winner(strip: np.ndarray, label_text: str, field: str,
                  floor: float, m_ru: float) -> tuple[str, float] | None:
    """One strip's variant-consensus read, as (value, margin), or None.

    A strip accepts only when its variants AGREE on the winner and at
    least one variant passes the full gate (calibration: requiring every
    variant to pass costs measured hits to sub-0.02-nat margin noise
    while agreement already screens the junk). The returned margin is the
    widest runner-up gap among the variants — the confidence signal that
    qualified the winner.
    """
    gated, tops, margins = [], [], []
    for _, img in _variants(strip):
        scored = score_strip(img, label_text, _MENUS[field])
        tops.append(scored[0][1] if scored else None)
        margins.append(_score_margin(scored)[2])
        gated.append(_gated_winner(scored, floor, m_ru))
    passing = [w for w in gated if w is not None]
    if not passing or len(set(tops)) != 1:
        return None
    value = passing[0]
    if len(set(passing)) != 1 or value != tops[0]:
        return None
    return value, max(margins)


def _second_resolution_confirms(strip: np.ndarray, label_text: str,
                                field: str, value: str, floor: float,
                                m2: dict[str, float]) -> bool:
    """MIB_CTCFILL_MARGIN: re-score the crop at SECOND_RES_SCALE and
    require the second resolution to name the same winner while clearing
    its own slackened score floor and the field's margin floor. A genuine
    read survives the resize; the occlusion prefix fit behind the
    measured wrong fires does not.

    One rec pass on the raw crop, not both variants: the contrast stretch
    exists to rescue faint ink and the primary pass has already made the
    two views agree, so re-running it here changed no verdict on the
    calibration population (identical right/wrong over all 399 accepted
    strips) and would double this layer's cost.
    """
    scored = score_strip(strip, label_text, _MENUS[field],
                         scale=SECOND_RES_SCALE)
    return _gated_winner(scored, floor - SECOND_RES_FLOOR_SLACK,
                         m2[field]) == value


def _accept_strips(strips: list[tuple[np.ndarray, str, float]],
                   field: str, floor: float = SCORE_FLOOR,
                   m_ru: float = MARGIN_RUNNER_UP,
                   m2: dict[str, float] | None = None
                   ) -> tuple[str, float] | None:
    """Gated acceptance over one page's strips: (value, confidence) or
    None. Strips disagreeing on the winner abstain outright — two label
    instances on one page reading differently is under-determination, not
    evidence. Under MIB_CTCFILL_MARGIN each strip the primary resolution
    accepts must additionally be confirmed at the second resolution; the
    embargo/mode-default drops run first so a rejected value never pays
    for the extra rec pass."""
    if fusion_enabled():
        return _accept_strips_fused(strips, field, floor, m_ru, m2)
    accepted: list[tuple[str, float]] = []
    for strip, label_text, loc_conf in strips:
        got = _strip_winner(strip, label_text, field, floor, m_ru)
        if got is None:
            continue
        value = got[0]
        if value in _EXCLUDED_VALUES or value == _mode_default(field):
            continue
        if _margin_gates() and not _second_resolution_confirms(
                strip, label_text, field, value, floor,
                m2 if m2 is not None else MARGIN2_FLOOR):
            continue
        accepted.append((value, loc_conf))
    if not accepted:
        return None
    values = {value for value, _ in accepted}
    if len(values) != 1:
        return None
    value = values.pop()
    conf = min(CONF_CAP, max(conf for _, conf in accepted))
    return value, conf


def read_field(result: ScanOcrResult, field: str) -> tuple[str, float] | None:
    """One page's word-box-anchored gated read for one field."""
    return _accept_strips(locate_strips(result, field), field)


# --------------------------------------------- frame-registered channel
# MIB_ROWRESTORE routes here (see row_restore.py). A horizontally
# displaced band is a GEOMETRY defect: where the cut runs through a text
# row's x-height the glyphs are split across two x offsets, which no
# threshold or contrast variant can undo and which leaves the row's label
# unread — so the ladder produces no word boxes for it and the ordinary
# word-box locator has nothing to anchor on. That is precisely the slice
# the grid locator exists for, and registering the band puts the row back
# on the intake lattice the grid fit is looking for.
#
# Everything downstream is the unchanged ctcfill contract: same locators,
# same label-inclusive candidates, same calibrated floors and margins,
# same CONF_CAP, same excluded values and mode-default suppression.
#
# Scoping: only rows the REGISTRATION changed are eligible. An undamaged
# row elsewhere on the page is ordinary ctcfill's business, and letting
# this channel read it would credit MIB_ROWRESTORE for fills that have
# nothing to do with the remap — the A/B has to attribute cleanly.


# Rows a needed field can live on, in render space. The intake template
# prints its eight label rows inside _GRID_BLOCK's native y window, so
# even with no lattice fit yet (the damaged page is why we are here) the
# envelope is known: native 0.5x -> render 2x.
_NEED_ENVELOPE = (2 * _GRID_BLOCK[2], 2 * _GRID_BLOCK[3])


def located_labels(result: ScanOcrResult,
                   fields_needed: list[str]) -> list[str]:
    """Needed fields whose printed label the ladder already found here."""
    out = []
    for field in fields_needed:
        for pattern in _LABELS.get(field, ()):
            head = pattern[0].lower()
            if any(_clean_token(w.text) == head and w.h > 0
                   and w.conf >= LOC_MIN_CONF for w in result.words or []):
                out.append(field)
                break
    return out


def needed_spans(result: ScanOcrResult, fields_needed: list[str]
                 ) -> list[tuple[int, int]]:
    """Line spans worth repairing on a grid-path page.

    Arthur's rule: do not align a strip with nothing in it that anyone
    wants. The consumer is a closed-menu read of four fields on a page
    where no label was located, so the only rows that can matter are the
    template's own label block — the same window the grid locator
    searches. Everything outside it is detected and left alone.
    """
    return [_NEED_ENVELOPE]


def _in_band(y0: int, y1: int,
             spans: list[tuple[int, int]] | None) -> bool:
    """None = no row restriction (a vertical repair moved columns)."""
    if spans is None:
        return True
    return any(y0 < hi and y1 > lo for lo, hi in spans)


def _grid_row_span(fit, field: str) -> tuple[int, int]:
    """Render-space y range of a field's grid row (the fit is native 0.5x)."""
    _ncc, _gx, gy, pitch, lab_h = fit
    top = gy + int(round(_GRID_ROW_OF[field] * pitch))
    return 2 * (top - 7), 2 * (top + lab_h + 7)


def _agreed(page_reads: list[tuple[str, float]]) -> tuple[str, float] | None:
    """Cross-page agreement, same rule as fill(): disagreement abstains."""
    if not page_reads:
        return None
    values = {value for value, _ in page_reads}
    if len(values) != 1:
        return None
    return values.pop(), min(CONF_CAP, max(c for _, c in page_reads))


def restored_view(result: ScanOcrResult, fields_needed: list[str]
                  ) -> tuple[ScanOcrResult, list[tuple[int, int]]] | None:
    """(page registered on its frame, repaired spans), or None.

    Skipped on pages whose orientation the ladder had to guess: the rails
    are only rails when the page is the right way up, and a rotated form
    would have the detector measuring the wrong axis off the wrong rules.
    Abstaining is free; firing garbage is not.

    The spans returned are the rows the repair actually MOVED, which is
    what makes any fill attributable to the registration rather than to
    ctcfill's ordinary reach.
    """
    if not result.upright or result.orientation_estimated:
        return None
    # GRID-PATH PAGES ONLY. Where the ladder did read a field's label, the
    # closed-menu scorer already survives the displacement unaided —
    # measured: the correct value comes back off a row split in two, at
    # offsets from 30 to 240 px, registered or not. Registering such a
    # page buys nothing and costs a gate, so the eligible population is
    # exactly the pages where no label was found and the lattice is the
    # only locator left.
    if located_labels(result, fields_needed):
        return None
    wanted = needed_spans(result, fields_needed)
    reg = row_restore.register(result.gray, RENDER_DPI, wanted=wanted)
    if reg is None and row_restore.vertical_enabled():
        reg = row_restore.register(result.gray, RENDER_DPI, wanted=None,
                                   axis=row_restore.VERTICAL)
    if reg is None:
        return None
    horizontal = reg.axis == row_restore.HORIZONTAL
    view = ScanOcrResult(
        lines=[], gray=reg.image, upright=result.upright,
        best_rot=result.best_rot,
        orientation_estimated=result.orientation_estimated,
        # No carried-over geometry: this page reached here precisely
        # because no needed label was located on it.
        words=[],
    )
    # A vertical repair moves COLUMN bands, so its spans are not row spans
    # and cannot restrict which lattice row is eligible: None = no row
    # restriction. The horizontal case keeps its exact attribution.
    return view, (reg.spans if horizontal else None)


def fill_restored(scans: dict[int, ScanOcrResult], fields_needed: list[str],
                  budget_left=lambda: True,
                  page_types: dict[int, str | None] | None = None,
                  ) -> dict[str, tuple[str, float]]:
    """Gated fills read off frame-registered pages (MIB_ROWRESTORE).

    The GRID path only, by construction: restored_view admits a page just
    when no needed field's label was located on it, which is exactly the
    condition under which fill() would fall through to the lattice. The
    word-box path is not reachable here and is not attempted — on a page
    whose labels WERE read, the closed-menu scorer already survives the
    displacement without help, so registering it is pure cost.

    Same contract as fill() otherwise: same locator, same calibrated grid
    floor and margin, same CONF_CAP, same cross-page agreement. Pages
    whose gate does not fire cost one strided frame trace and never reach
    the recognizer.
    """
    if not fields_needed or not scans or not available():
        return {}
    views: dict[int, tuple[ScanOcrResult, list[tuple[int, int]] | None]] = {}
    for index in sorted(scans):
        if not budget_left():
            return {}
        view = restored_view(scans[index], fields_needed)
        if view is not None:
            views[index] = view
    if not views:
        return {}

    fills: dict[str, tuple[str, float]] = {}
    fits: dict[int, object] = {}
    for field in fields_needed:
        page_reads = []
        for index in sorted(views):
            if not budget_left():
                return fills
            if page_types is not None and \
                    page_types.get(index) not in (None, "intake"):
                continue
            view, spans = views[index]
            if index not in fits:
                fits[index] = page_grid_fit(view)
            fit = fits[index]
            # The lattice row must fall inside a span the repair MOVED,
            # or the fill is ctcfill's ordinary reach and not ours.
            if fit is None or not _in_band(*_grid_row_span(fit, field),
                                           spans):
                continue
            got = _accept_strips(
                grid_strips(view, field, fit=fit), field,
                floor=GRID_SCORE_FLOOR, m_ru=GRID_MARGIN_RUNNER_UP,
                m2=GRID_MARGIN2_FLOOR)
            if got is not None:
                page_reads.append(got)
        agreed = _agreed(page_reads)
        if agreed is not None:
            fills[field] = agreed
    return fills


def fill(scans: dict[int, ScanOcrResult], fields_needed: list[str],
         budget_left=lambda: True,
         page_types: dict[int, str | None] | None = None,
         ) -> dict[str, tuple[str, float]]:
    """Gated fills for every still-unread menu field, across scan pages.

    Word-box anchoring first (free — the ladder already produced the
    geometry). Fields the word-box path did not ACCEPT fall through to
    the grid locator, one cached fit per intake-typed/untyped page: a
    readable-but-abstaining label instance elsewhere (a registry row,
    say) must not veto the intake-grid attempt on the destroyed page —
    measured identical precision/net to the never-located-only trigger,
    and it is what recovers MIB-000016's species fill in the full
    pipeline. Cross-page disagreement on a field abstains (same rule as
    within a page). Bounded by the caller's escalation soft budget.
    """
    if not fields_needed or not scans or not available():
        return {}
    if fusion_enabled():
        return _fill_fused(scans, fields_needed, budget_left, page_types)
    fills: dict[str, tuple[str, float]] = {}
    unlocated: list[str] = []
    for field in fields_needed:
        page_reads: list[tuple[str, float]] = []
        for index in sorted(scans):
            if not budget_left():
                return fills
            got = _accept_strips(locate_strips(scans[index], field), field)
            if got is not None:
                page_reads.append(got)
        if not page_reads:
            unlocated.append(field)
            continue
        values = {value for value, _ in page_reads}
        if len(values) != 1:
            continue
        fills[field] = (values.pop(),
                        min(CONF_CAP, max(c for _, c in page_reads)))
    # Grid fallback for the fields the word-box path accepted nothing on
    # (the deep-damage slice lives here: measured, every recognizer goes
    # blind on exactly these pages while pixel correlation still sees
    # the form). One grid fit per eligible page serves every such field;
    # a page confidently typed as another template is skipped (the grid
    # is intake geometry).
    if unlocated:
        fits: dict[int, object] = {}
        for field in unlocated:
            page_reads = []
            for index in sorted(scans):
                if not budget_left():
                    return fills
                if page_types is not None and \
                        page_types.get(index) not in (None, "intake"):
                    continue
                result = scans[index]
                if index not in fits:
                    fits[index] = page_grid_fit(result)
                if fits[index] is None:
                    continue
                got = _accept_strips(
                    grid_strips(result, field, fit=fits[index]), field,
                    floor=GRID_SCORE_FLOOR, m_ru=GRID_MARGIN_RUNNER_UP,
                    m2=GRID_MARGIN2_FLOOR)
                if got is not None:
                    page_reads.append(got)
            if not page_reads:
                continue
            values = {value for value, _ in page_reads}
            if len(values) != 1:
                continue
            fills[field] = (values.pop(),
                            min(CONF_CAP, max(c for _, c in page_reads)))
    return fills


# ------------------------------------------------------ likelihood fusion
# MIB_CTCFILL_FUSION=1 (default OFF). Everything below is additive: with
# the flag unset the two call sites above fall straight through and no
# function here ever runs.
#
# The flag-off scorer treats each view of a field (two preprocessing
# variants per strip, up to _MAX_STRIPS_PER_FIELD label instances per
# page, every page carrying the field) as a separate witness that must
# INDEPENDENTLY clear the gate, and vetoes the field when two witnesses
# name different values. That throws away the one thing the exact CTC
# forward gives us for free: every witness scores the SAME closed menu on
# a common scale, so they can be pooled before any decision is taken.
# Under fusion a candidate's views are averaged into one score and the
# gate fires once, on the pooled evidence. Two views that disagree now
# argue on the same scale instead of cancelling: the candidate that is
# consistently second-best across views beats the one that wins a single
# view and collapses in the next.


def fusion_enabled() -> bool:
    """One env gate (MIB_CTCFILL_FUSION=1, default OFF) for likelihood
    fusion across views. Unset/0 => the flag-off paths run untouched."""
    return os.environ.get("MIB_CTCFILL_FUSION") == "1"


def fuse_scores(score_lists: list[list[tuple[float, str]]]
                ) -> list[tuple[float, str]]:
    """Combine per-view (score, value) lists into one, best first.

    AVERAGE, not sum. Every view reports the same quantity — a
    length-normalized CTC log-prob — so the mean of N views lands on
    exactly the scale one view produces, and SCORE_FLOOR /
    MARGIN_RUNNER_UP / MARGIN_NONE / the GRID_* pair stay valid on the
    fused list with no re-derivation. A sum would scale both the floor
    and every margin by N and silently invalidate all four constants.

    A candidate missing from some view (out-of-dict for that view's
    label text, so score_strip skipped it) is averaged over the views
    that did score it: absence is not evidence against a candidate, and
    the result is still on the single-view scale. Ties break by value,
    exactly as score_strip's own sort does.
    """
    pooled: dict[str, list[float]] = {}
    for scored in score_lists:
        for score, value in scored:
            pooled.setdefault(value, []).append(score)
    fused = [(sum(scores) / len(scores), value)
             for value, scores in pooled.items()]
    fused.sort(reverse=True)
    return fused


def _strip_scores(strip: np.ndarray, label_text: str,
                  field: str) -> list[tuple[float, str]]:
    """One strip's multi-view fused score list.

    Scores exactly the views _variants() already yields and exactly once
    each — the flag-off path computes these same posteriorgrams, so
    fusion adds zero inference; it only changes what is done with the
    numbers.
    """
    return fuse_scores([score_strip(img, label_text, _MENUS[field])
                        for _, img in _variants(strip)])


def _page_scores(strips: list[tuple[np.ndarray, str, float]], field: str
                 ) -> tuple[list[tuple[float, str]], float,
                            list[tuple[np.ndarray, str]]] | None:
    """One page's fused score list, its best locator confidence, and the
    (crop, label_text) pairs that went into it — or None when the page
    contributed no view at all. The crops are carried out so the
    second-resolution confirmation can re-score exactly the evidence the
    fused verdict was taken on, without locating anything twice.

    Strips are pooled here, BEFORE the cross-page pool, so a page that
    happens to print the label three times does not outvote a page that
    prints it once: every page enters the cross-page average with weight
    one. A page whose strip failed to locate returns None and drops out
    of the average entirely — no penalty, it simply saw nothing.
    """
    lists, confs, views = [], [], []
    for strip, label_text, loc_conf in strips:
        scored = _strip_scores(strip, label_text, field)
        if not scored:
            continue
        lists.append(scored)
        confs.append(loc_conf)
        views.append((strip, label_text))
    if not lists:
        return None
    return fuse_scores(lists), max(confs), views


def _accept_fused(fused: list[tuple[float, str]], field: str,
                  floor: float, m_ru: float) -> str | None:
    """The unchanged safety gates, applied once to the fused winner:
    the calibrated floor/margins via _gated_winner (so a sibling lever's
    changes to those constants flow through here too), then the
    hard-embargo embargo list and mode-default suppression.

    A fused winner that is embargoed or is the writer's mode default
    abstains outright rather than promoting the runner-up: the runner-up
    is not what the pooled evidence favours, and the worst case of
    abstaining is the field staying unread, exactly as it was.
    """
    winner = _gated_winner(fused, floor, m_ru)
    if winner is None:
        return None
    if winner in _EXCLUDED_VALUES or winner == _mode_default(field):
        return None
    return winner


def _second_resolution_confirms_fused(
        views: list[tuple[np.ndarray, str]], field: str, value: str,
        floor: float, m2: dict[str, float]) -> bool:
    """MIB_CTCFILL_MARGIN applied to a FUSED winner: re-score every crop
    the fused verdict pooled at SECOND_RES_SCALE, fuse those second-
    resolution lists the same way, and require the fused second
    resolution to name the same winner while clearing the slackened
    score floor and the field's second-resolution margin floor.

    Confirmed on the pooled second resolution, not on one crop. The
    fused winner is by construction the candidate best supported by ALL
    the views; asking a single crop to name it as that crop's own argmax
    — against a floor TIGHTER than the primary gate's — asks a question
    the fused verdict never claimed to answer, and would reject exactly
    the fills fusion exists to add (the pooled winner is typically the
    consistent runner-up, so each individual view still favours its own
    candidate). Pooling both sides keeps the comparison like-for-like.

    Restrictive, exactly as the single-resolution layer is: it runs only
    after _accept_fused has already passed, so it can remove a fill and
    never add one.

    Raw crop only, one rec pass per pooled crop — the layer's cost
    contract. On the single-strip case, which is the common one and the
    one the layer was calibrated on, this IS the single-resolution
    behaviour: one crop in, one rec pass, the same verdict.
    """
    fused2 = fuse_scores([score_strip(strip, label_text, _MENUS[field],
                                      scale=SECOND_RES_SCALE)
                          for strip, label_text in views])
    return _gated_winner(fused2, floor - SECOND_RES_FLOOR_SLACK,
                         m2[field]) == value


def _accept_strips_fused(strips: list[tuple[np.ndarray, str, float]],
                         field: str, floor: float, m_ru: float,
                         m2: dict[str, float] | None = None
                         ) -> tuple[str, float] | None:
    """_accept_strips under fusion: pool every view this page offers,
    gate once. Same (value, confidence) contract, same CONF_CAP. Under
    MIB_CTCFILL_MARGIN the fused winner must additionally survive the
    second resolution; the embargo/mode-default drops inside
    _accept_fused run first, so a rejected value never pays for the
    extra rec pass."""
    got = _page_scores(strips, field)
    if got is None:
        return None
    fused, loc_conf, views = got
    winner = _accept_fused(fused, field, floor, m_ru)
    if winner is None:
        return None
    if _margin_gates() and not _second_resolution_confirms_fused(
            views, field, winner, floor,
            m2 if m2 is not None else MARGIN2_FLOOR):
        return None
    return winner, min(CONF_CAP, loc_conf)


def _fill_fused(scans: dict[int, ScanOcrResult], fields_needed: list[str],
                budget_left, page_types: dict[int, str | None] | None,
                ) -> dict[str, tuple[str, float]]:
    """fill() under MIB_CTCFILL_FUSION=1.

    Identical locator, budget accounting, grid-fallback trigger and
    safety gates; the one change is WHERE the decision is taken. Flag
    off, each page is gated alone and the field is vetoed when two pages
    name different values. Here every page's views are pooled into one
    score list per field and the gate fires once on it, so the
    cross-page AGREEMENT veto is replaced by a single verdict on the
    pooled evidence — a page that saw nothing contributes nothing, and
    two pages that disagree are resolved on the merits instead of
    cancelling each other out.

    Word-box and grid views are never pooled together: their strips live
    at different scales (288-DPI crops vs the native render) and carry
    their own calibrated floors, so the grid pass runs as its own pooled
    decision under GRID_SCORE_FLOOR / GRID_MARGIN_RUNNER_UP — and, when
    MIB_CTCFILL_MARGIN is on, under GRID_MARGIN2_FLOOR rather than the
    word-box MARGIN2_FLOOR.

    Grid fall-through keeps the flag-off trigger: a field the word-box
    pass did not ACCEPT falls through, whether it located nothing or
    located views that lost at the gate (flag off, a located-but-
    rejected field falls through the same way — a readable-but-
    abstaining registry row must not veto the intake-grid attempt on the
    destroyed page).
    """
    fills: dict[str, tuple[str, float]] = {}
    unlocated: list[str] = []
    for field in fields_needed:
        page_lists, confs, views = [], [], []
        for index in sorted(scans):
            if not budget_left():
                return fills
            got = _page_scores(locate_strips(scans[index], field), field)
            if got is None:
                continue
            page_lists.append(got[0])
            confs.append(got[1])
            views.extend(got[2])
        if page_lists:
            winner = _accept_fused(fuse_scores(page_lists), field,
                                   SCORE_FLOOR, MARGIN_RUNNER_UP)
            if winner is not None and (
                    not _margin_gates()
                    or _second_resolution_confirms_fused(
                        views, field, winner, SCORE_FLOOR,
                        MARGIN2_FLOOR)):
                fills[field] = (winner, min(CONF_CAP, max(confs)))
                continue
        unlocated.append(field)
    if unlocated:
        fits: dict[int, object] = {}
        for field in unlocated:
            page_lists, confs, views = [], [], []
            for index in sorted(scans):
                if not budget_left():
                    return fills
                if page_types is not None and \
                        page_types.get(index) not in (None, "intake"):
                    continue
                result = scans[index]
                if index not in fits:
                    fits[index] = page_grid_fit(result)
                if fits[index] is None:
                    continue
                got = _page_scores(
                    grid_strips(result, field, fit=fits[index]), field)
                if got is None:
                    continue
                page_lists.append(got[0])
                confs.append(got[1])
                views.extend(got[2])
            if not page_lists:
                continue
            winner = _accept_fused(fuse_scores(page_lists), field,
                                   GRID_SCORE_FLOOR, GRID_MARGIN_RUNNER_UP)
            if winner is not None and (
                    not _margin_gates()
                    or _second_resolution_confirms_fused(
                        views, field, winner, GRID_SCORE_FLOOR,
                        GRID_MARGIN2_FLOOR)):
                fills[field] = (winner, min(CONF_CAP, max(confs)))
    return fills
