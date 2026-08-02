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
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from . import vocab
from .ocr import ScanOcrResult, _models_dir

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


def available() -> bool:
    """True when the rec bundle loads (models are not shipped to every
    dev checkout); the pipeline block abstains quietly when False."""
    try:
        _session()
        return True
    except Exception:
        return False


def _preprocess(gray: np.ndarray) -> np.ndarray:
    """Grayscale strip -> the rec model's (1, 3, 48, W) normalized tensor.
    PP-OCR rec normalization is (x/255 - 0.5) / 0.5 across generations."""
    h, w = gray.shape
    tw = min(_REC_MAX_W, max(16, int(round(_REC_H * w / h))))
    img = cv2.resize(gray, (tw, _REC_H), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.repeat(x[None, None], 3, axis=1)


def posteriorgram(gray: np.ndarray) -> np.ndarray:
    """(T, C) log-probability frames for one strip."""
    sess = _session()
    y = sess.run(None, {sess.get_inputs()[0].name: _preprocess(gray)})[0][0]
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
                values: tuple[str, ...]) -> list[tuple[float, str]]:
    """Rank menu values by length-normalized CTC log-prob of the
    label-inclusive candidate string; "<none>" = the label printed with
    no legible value (the null hypothesis). Best first."""
    logp = posteriorgram(strip)
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


def _gated_winner(scored: list[tuple[float, str]],
                  floor: float = SCORE_FLOOR,
                  m_ru: float = MARGIN_RUNNER_UP) -> str | None:
    """Apply the calibrated acceptance gate to one scored variant."""
    if not scored:
        return None
    by = {value: s for s, value in scored}
    top_score, winner = scored[0]
    if winner == "<none>":
        return None
    others = [s for s, v in scored if v not in (winner, "<none>")]
    if top_score < floor:
        return None
    if others and top_score - max(others) < m_ru:
        return None
    if top_score - by.get("<none>", _NEG) < MARGIN_NONE:
        return None
    return winner


def _mode_default(field: str) -> str:
    """The writer's train-mode fallback for the field: emitting it is a
    no-op (the output row prints it anyway when the field is unread), and
    default-accepts were the ONLY wrong-fire mode the calibration
    measured — so the fill never emits it."""
    from . import writer

    return writer._DEFAULTS[field]


def _accept_strips(strips: list[tuple[np.ndarray, str, float]],
                   field: str, floor: float = SCORE_FLOOR,
                   m_ru: float = MARGIN_RUNNER_UP
                   ) -> tuple[str, float] | None:
    """Gated acceptance over one page's strips: (value, confidence) or
    None. A strip accepts only when its variants AGREE on the winner and
    at least one variant passes the full gate (calibration: requiring
    every variant to pass costs measured hits to sub-0.02-nat margin
    noise while agreement already screens the junk). Strips disagreeing
    on the winner abstain outright — two label instances on one page
    reading differently is under-determination, not evidence."""
    accepted: list[tuple[str, float]] = []
    for strip, label_text, loc_conf in strips:
        gated, tops = [], []
        for _, img in _variants(strip):
            scored = score_strip(img, label_text, _MENUS[field])
            tops.append(scored[0][1] if scored else None)
            gated.append(_gated_winner(scored, floor, m_ru))
        passing = [w for w in gated if w is not None]
        if not passing or len(set(tops)) != 1:
            continue
        value = passing[0]
        if len(set(passing)) != 1 or value != tops[0]:
            continue
        if value in _EXCLUDED_VALUES or value == _mode_default(field):
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
                    floor=GRID_SCORE_FLOOR, m_ru=GRID_MARGIN_RUNNER_UP)
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
