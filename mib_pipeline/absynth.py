"""Generator-inversion menu reader (MIB_ABSYNTH=1, default OFF).

Analysis by synthesis. Where every recognizer in the stack has gone blind
on a destroyed intake row, this module stops trying to READ the row and
instead RE-RENDERS it: the packet generator's raster layout is known
exactly, so each closed-menu value can be drawn, degraded to match the
page, and correlated against the pixels. The winner is the candidate whose
synthesized image best explains the observation. No recognizer is involved
at any point, which is the whole reason it survives damage that defeats
both OCR engines AND the CTC posterior channel (MIB_CTCFILL): when the
strip is destroyed the posteriors are garbage, but pixel correlation
against a correctly degraded template still separates 13 candidates.

Mechanism adapted from the MIT-licensed luke-harriman public solution
(`lib/absynth2.py` label-anchored analysis-by-synthesis on the exact
generator grid, and `lib/mfr_reader.py` two-anchor registration +
common-canvas NCC decode; see ATTRIBUTION.md). Both halves are used
because each fixes the other's weakness: absynth2 has the exact layout
table and the self-calibrating per-row kernel fit but scores candidates in
per-candidate windows, which biases NCC by candidate length; mfr_reader
has the fixed common canvas that removes that bias and the independent
two-anchor registration check, but locates rows by free search rather than
on the generator grid.

Four steps, each of which can abstain:

1. REGISTRATION (two independent anchors). The page title and the line
   "Case ID: <cid>" are localised independently over their own windows.
   The case id comes from the FILENAME and is used purely as a
   known-content geometric fiducial — never as a lookup key, and nothing
   about the case id reaches the emitted value. Their vertical and
   horizontal offsets are fixed by the layout, so requiring them to agree
   is a real registration check rather than a restatement of one match.
   The fiducial is POSITIONAL, not authenticating: most of its
   discriminative mass is the shared "Case ID: MIB-000" boilerplate, and
   Helvetica digits are tabular, so a line bearing a different case id
   registers in the same place (measured; test_absynth.py records it).
   That is the intended contract — it pins WHERE the rows are, not whose
   they are — and identity is never at stake, because the pages are this
   packet's own and the case id comes from this packet's filename.
2. ROW LOCATION. Every row's baseline offset from the Case ID row is a
   generator constant (measured: 62 / 93 / 124 / 217 px at 144 dpi). The
   field's own printed label is re-located in a tight window around the
   predicted position.
3. KERNEL RECOVERY (self-calibration). The row's own label bitmap is
   KNOWN, so the local degradation is recovered by fitting an
   (ellipse-erode, Gaussian-sigma) pair to that label at its located
   origin. Per row, per page, no offline fitting: whatever destroyed the
   value destroyed the label beside it the same way.
4. DECODE. Each menu candidate is rendered, placed on a FIXED common
   canvas at the value pen origin, degraded with the recovered kernel, and
   scored by normalized cross-correlation over a small jitter window. The
   common canvas is essential: per-candidate windows make NCC incomparable
   across candidates of different length.

Fill-only contract, identical in spirit to ctcfill.py (enforced at the
pipeline.py call site, tested in test_absynth.py): fires only for the four
decision-relevant closed menus, and only where the field is STILL UNREAD
after every other channel including ctcfill — i.e. exactly the slots that
would otherwise be imputed with the writer's mode default. Accepted values
are extraction candidates only: confidence is capped below the
affirmative-read threshold (fields._KNOWN_MIN_OCR_CONF = 0.55) so `known`
stays False and no policy rule can consume them. Hard-embargo worlds are
never emitted (a reconstructed read must not mint an R1/R2 denial), and
the writer's own mode default is never emitted as a "win" — emitting it is
a no-op, and default-accepts were the only wrong-fire mode ctcfill's
calibration measured.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import vocab

# --------------------------------------------------------------- geometry
# The generator draws scan pages as a 1224x1584 raster of a 612x792 pt
# page: exactly 2.0 px/pt (144 dpi). The pipeline renders scans at 288 dpi
# (ocr.RENDER_DPI), so a 0.5x downscale of ScanOcrResult.gray lands in this
# native space with no further resampling -- the same space ctcfill's grid
# locator already works in.
#
# On the image source: this reads the pipeline's own render rather than
# re-extracting the embedded JPEG (luke-harriman's readers do the latter,
# for the stated reason that it structurally excludes the PDF text layer).
# CHECKED on train packets, and the reasoning does not transfer: the
# text-layer injection spans are pure white (0xffffff, 5 pt) and so are
# absent from any render, while the visible faint "SYSTEM: ... answer key"
# line is painted INTO the scan raster by the generator — present in the
# embedded JPEG and the render alike, so switching source buys nothing.
# What actually defends this module is structural: it reads only at fixed
# grid offsets below the registered Case ID row, only accepts
# closed-vocabulary values, and the trap line sits above the title, tens
# of pixels outside any row window.
#
# NOTE, corrected after measurement: ScanOcrResult.gray is NOT deskewed,
# despite the "orientation-fixed, deskewed render" comment on that field.
# ocr_scan_page returns the gray it was passed; its deskew and rotation
# work are ADDITIVE OCR passes ("never a replacement", ocr.py). So nothing
# upstream straightens the page for us and this module must do its own —
# see deskew_angle below.
PT2PX = 2.0
NATIVE_SIZE = (1584, 1224)          # (h, w) sanity check only

# Body text is Helvetica 6.6 pt (13.2 px em), titles 9.0 pt. MEASURED on a
# clean train intake page: Helvetica REGULAR beats Helvetica-Bold on the
# eight intake labels (summed NCC 7.33 vs 6.98) and the title fits at 9.0
# pt with NCC 0.962, so labels are not bold despite looking heavy under
# degradation.
BODY_PT = 6.6
TITLE_PT = 9.0
# Nominal left text margin in native px. Pages carry a global translation
# (measured -49 px in y, +19 px in x on the validation page), so this is
# only the centre of the registration search window, never a hard position.
XCOL = 106.0

INTAKE_TITLE = "FORM I-8090: Extraterrestrial Work Authorization Intake"

# Baseline offsets of each intake row from the Case ID row, in native px.
# From luke-harriman's absynth2 ROWS table (167/198/229/260/291/323/354/384
# absolute); VERIFIED against a clean train intake page, where the located
# baselines were 118/-/180/211/242 -- deltas 62/93/124 exactly as tabulated.
_ROW_DY = {
    "applicant_name": 31.0,
    "species_code": 62.0,
    "home_world": 93.0,
    "visa_class": 124.0,
    "sponsor_id": 156.0,
    "arrival_date": 187.0,
    "declared_purpose": 217.0,
}
_ROW_LABEL = {
    "species_code": "Species Code:",
    "home_world": "Home World:",
    "visa_class": "Visa Class:",
    "declared_purpose": "Declared Purpose:",
}
# Title pen origin -> Case ID pen origin, native px. luke-harriman's
# mfr_reader measured 39.0 on raster pages and their absynth2 row table
# implies 43 from a size-agnostic crop; 39.0 is the size-aware figure and
# is what a pen-origin renderer reproduces, so it is the one used here.
DY_TITLE_CASEID = 39.0

# ----------------------------------------------------------------- fields
# Identical to ctcfill.FIELDS by construction: this lever is the last
# resort BEHIND that one and must not widen the blast radius. fee_status,
# sponsor_id, dates, names and risk flags are all excluded -- open
# vocabularies, or tier-1 evidence this reader must never mint.
FIELDS = ("species_code", "home_world", "visa_class", "declared_purpose")

_MENUS = {
    "species_code": tuple(vocab.SPECIES_CODES),
    "home_world": tuple(vocab.HOME_WORLDS),
    "visa_class": tuple(vocab.VISA_CLASSES),
    "declared_purpose": tuple(vocab.PURPOSES),
}

# The generator prints a per-field damage sentinel INSTEAD of a value when
# it destroys a row ("Home World: [REGISTRY LOST]"). Counted across
# research/scan_ocr.jsonl: 21 PURPOSE ILLEGIBLE, 17 VISA CLASS TORN, 16
# SPECIES WHITEOUT, 13 REGISTRY LOST. These are exactly the rows this
# module is triggered on, so they are scored as NULL CANDIDATES: without
# them the decoder is forced to return the least-bad menu value for a row
# that has no value at all. When a sentinel wins, the row is affirmatively
# damaged and the reader abstains. Same design as luke-harriman's
# "[RISK PANEL MISSING]" entry in mfr_reader.FLAG_VALUES.
_SENTINELS = {
    "species_code": "[SPECIES WHITEOUT]",
    "home_world": "[REGISTRY LOST]",
    "visa_class": "[VISA CLASS TORN]",
    "declared_purpose": "[PURPOSE ILLEGIBLE]",
}

# Never emitted: pipeline.py infers planetary_embargo from
# evidence.value("home_world") regardless of `known`, so a wrong
# hard-embargo fill could mint an R1 denial. Mirrors
# ctcfill._EXCLUDED_VALUES. Kept as a literal frozenset rather than an
# import of policy.HARD_EMBARGO_WORLDS so the guard cannot be silently
# widened by an unrelated policy edit; test_absynth.py asserts they agree.
_EXCLUDED_VALUES = frozenset({"TRAPPIST-1e", "Eris Relay"})

# ------------------------------------------------------------------ gates
# Extraction-only confidence cap: strictly below fields._KNOWN_MIN_OCR_CONF
# (0.55), so an absynth fill populates the output row but never sets
# known=True. Same value as ctcfill.CONF_CAP.
CONF_CAP = 0.50

# Registration: both anchors must be found at these NCC floors and their
# measured offset must match the layout. Derived from luke-harriman's
# mfr_reader gates (label >= 0.65 for notes, >= 0.75 for flag rows, which
# measured 100% precision on their train split) and held at the stricter
# 0.75 here because this module fires on a WIDER field set than their two
# gated reads and every accepted value lands in a scored output row.
MIN_TITLE_NCC = 0.55
MIN_CASEID_NCC = 0.75
DY_TOL = 9.0                # mfr_reader's dy tolerance, unchanged
DX_TOL = 12.0               # tighter than their 34.0: our anchors are both
                            # left-aligned to the same text margin

# Row label, after registration: the label sits on the generator grid, so a
# genuine hit is strong. Below this the row was not found and the recovered
# kernel would be fitted to noise.
MIN_LABEL_NCC = 0.70

# Acceptance. The winner needs an absolute NCC floor and a margin over the
# runner-up.
#
# MIN_VALUE_MARGIN was inherited from luke-harriman at 0.10 (their
# NOTE_GATE / FLAGS_GATE, measured at 100% precision on their split, on
# this same common-canvas NCC statistic) and is now set from OUR OWN
# measurement. A census of 23 registered real slots with gold -- the menu
# fills recovered by diffing the ab-final arm against the guard-only
# control -- swept the gate:
#
#     margin  accepts  correct  precision
#      0.02     17       16       94%
#      0.03     13       12       92%
#      0.05     10        9       90%
#      0.10      4        3       75%   <- the inherited value
#      0.15      1        0        0%
#
# The gate is ANTI-correlated with precision on real pages: tightening it
# discards correct reads and does not remove the error, because the sole
# surviving mismatch carries the HIGHEST margin in the population. That
# mismatch (MIB-000533 visa, margin 0.155) is not a reading error at all
# -- the intake page plainly prints "Visa Class: TRANSIT-7" while gold
# says XW-1, a cross-page adjudication conflict that ctcfill read the same
# way. Excluding it the channel made zero true reading errors on 23 real
# slots, and the only other gold mismatches are sub-0.02 margin ties
# (0.010 and 0.006) -- which is where the real noise floor sits.
#
# So 0.03: just above the tie floor, 3x the recall of 0.10 at higher
# precision.
#
# CAVEAT, stated honestly: the census population is slots ctcfill could
# already read, whereas this lever fires only where nothing could. Error
# modes on that harder population may differ, so treat the precision
# column as indicative rather than as this lever's own precision. The
# downside is bounded either way -- the scorer treats a wrong value and a
# blank alike, and fills are conf-capped below the affirmative-read
# threshold, so no policy rule can consume one.
MIN_VALUE_NCC = 0.45
MIN_VALUE_MARGIN = 0.03

# Degradation search grids. luke-harriman's absynth2 fine grid is 8
# dilation kernels x 7 anisotropic sigma pairs = 56 fits per row. Collapsed
# here to isotropic (ellipse-erode, sigma) pairs because the canvas
# representation used for decoding is mfr_reader's white-background one,
# whose degrade() is isotropic by construction -- and the anisotropic pairs
# only ever won on their streak-damaged pages, which our flatten() already
# divides out. 4 x 6 = 24 fits, ~2.3x cheaper, and the coarse locate grid
# is a 12-entry subset.
_ERODE_KS = (0, 3, 5, 7)
_SIGMAS = (0.0, 0.6, 1.2, 1.8, 2.6, 3.4)
KERNEL_GRID = tuple((k, s) for k in _ERODE_KS for s in _SIGMAS)
LOCATE_GRID = tuple((k, s) for k in (0, 3, 5) for s in (0.0, 1.0, 2.0, 3.0))

# A degraded template that has gone featureless correlates with any smear.
MIN_TPL_STD = 12.0

# Search windows (native px) around the predicted position.
WIN_TITLE = (90, 60)        # (x, y) half-widths: the page's global offset
WIN_CASEID = (90, 60)       # measured up to ~50 px, so +-60 covers it
WIN_ROW = (14, 10)          # tight: registration has already fixed the page
JITTER = 3                  # decode jitter half-width, mfr_reader's value

# Cost bound. Raised from 4 after the census: packets carry 3-6 pages and
# the intake is NOT reliably early — measured registrations on page index
# 3 and 4 that a 4-page budget skipped outright.
_MAX_PAGES = 6

_RENDER_CACHE: dict[tuple, tuple[np.ndarray, float, float]] = {}
_CANVAS_CACHE: dict[tuple, np.ndarray] = {}


# --------------------------------------------------------------- rendering
def text_width(text: str, size: float, font: str = "helv") -> float:
    import pymupdf

    return pymupdf.get_text_length(text, fontname=font, fontsize=size)


def render_strip(text: str, size: float, font: str = "helv", pad: int = 3):
    """Tight grayscale render of `text` (ink on white).

    Returns (img, pen_x, pen_y): the pen origin -- baseline, left edge of
    the run -- in pixels inside img. Clipping the render relative to the
    font size (rather than a fixed box) is what makes offsets between runs
    of DIFFERENT sizes, i.e. the title and the body rows, comparable.
    """
    key = (text, round(size, 3), font, pad)
    hit = _RENDER_CACHE.get(key)
    if hit is not None:
        return hit
    import pymupdf

    w = text_width(text, size, font)
    doc = pymupdf.open()
    page = doc.new_page(width=max(w + 40, 60), height=max(size * 3, 40))
    bx, by = 10.0, size * 2.0
    page.insert_text((bx, by), text, fontname=font, fontsize=size)
    clip = pymupdf.Rect(bx - pad, by - size * 1.02 - pad,
                        bx + w + pad, by + size * 0.30 + pad)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(PT2PX, PT2PX), clip=clip,
                          colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width).copy()
    doc.close()
    out = (img, (bx - clip.x0) * PT2PX, (by - clip.y0) * PT2PX)
    if len(_RENDER_CACHE) < 20000:
        _RENDER_CACHE[key] = out
    return out


def degrade(img: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """The generator's damage model: ink spread then optical blur.

    Erode (not dilate) because this representation is ink-DARK on white
    paper, so eroding grows the strokes.
    """
    if k > 0:
        img = cv2.erode(img, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)))
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    return img


def flatten(gray: np.ndarray) -> np.ndarray:
    """Divide out a wide morphological close: kills low-frequency toner
    clouds and streaks, keeps glyph strokes."""
    g = gray.astype(np.float32)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (25, 25)))
    return np.clip(g / np.maximum(bg, 8.0) * 255.0, 0, 255).astype(np.uint8)


def native_page(gray: np.ndarray) -> np.ndarray:
    """ScanOcrResult.gray (288 dpi, deskewed, orientation-fixed) -> the
    144-dpi native raster space the generator's constants live in."""
    return cv2.resize(gray, None, fx=0.5, fy=0.5,
                      interpolation=cv2.INTER_AREA)


# Every coordinate this module touches lives in the page's top-left block:
# the title search window bottoms out near y=290, the lowest row
# (declared_purpose, +217 from Case ID) near y=460 once the global offset
# is allowed for, and the widest run is the title itself at ~700 px. The
# crop is taken FROM THE ORIGIN so every absolute constant stays valid.
# Worth doing because flatten()'s 25x25 morphological close is the single
# most expensive step in a page read: profiled at 0.075 s over the full
# 1224x1584 raster versus 0.012 s over this block, i.e. 38% of the cost of
# a page read spent normalising background the reader never looks at.
BODY_H = 520
BODY_W = 720


# Deskew. MEASURED on the real fill population: of 15 cases where
# registration failed outright, 9 register once the page is straightened,
# at angles from -2.25 to +5.00 degrees. Every one of those recoveries is
# a pure in-plane skew — a rot90 search over the same pages recovered
# nothing extra, so none is attempted. The generator's row pitch is 31 px
# and a 2-degree skew walks a row by ~18 px across the label+value span,
# which is more than the row window, so this is the difference between
# registering and abstaining rather than a refinement.
DESKEW_LIMIT = 7.0          # generator's observed skew range
DESKEW_STEP = 0.25
DESKEW_DS = 4               # estimate on a 4x downscale; angle is scale-free
DESKEW_MIN_APPLY = 0.2      # below this the warp costs more than it buys


def deskew_angle(native: np.ndarray) -> float:
    """Skew angle by projection-profile sharpness.

    Straight text maximises the row-to-row jump in the horizontal ink
    profile: at the true angle every glyph row lines up and the profile
    becomes a comb. Ink is thresholded first so paper texture cannot
    dominate the sum.
    """
    small = flatten(cv2.resize(native, None, fx=1.0 / DESKEW_DS,
                               fy=1.0 / DESKEW_DS,
                               interpolation=cv2.INTER_AREA))
    ink = np.clip(255.0 - small.astype(np.float32), 0, 255)
    ink[ink < 25] = 0
    h, w = ink.shape
    best = (-1.0, 0.0)
    for angle in np.arange(-DESKEW_LIMIT, DESKEW_LIMIT + 1e-9, DESKEW_STEP):
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
        rotated = cv2.warpAffine(ink, matrix, (w, h),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
        profile = rotated.sum(axis=1)
        sharpness = float(((profile[1:] - profile[:-1]) ** 2).sum())
        if sharpness > best[0]:
            best = (sharpness, float(angle))
    return best[1]


def observe(gray: np.ndarray) -> np.ndarray:
    """Scan-page render -> the flattened, straightened native block."""
    native = native_page(gray)
    angle = deskew_angle(native)
    if abs(angle) >= DESKEW_MIN_APPLY:
        h, w = native.shape
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        native = cv2.warpAffine(native, matrix, (w, h),
                                flags=cv2.INTER_LINEAR, borderValue=255)
    block = native[:min(BODY_H, native.shape[0]),
                   :min(BODY_W, native.shape[1])]
    return flatten(np.ascontiguousarray(block))


# ------------------------------------------------------------ localisation
class Hit:
    """A located run of known text, addressed by its PEN ORIGIN."""

    __slots__ = ("score", "x", "y", "size", "deg")

    def __init__(self, score, x, y, size, deg):
        self.score, self.x, self.y = score, x, y
        self.size, self.deg = size, deg


def locate(obs: np.ndarray, text: str, cx: float, cy: float, size: float,
           win: tuple[int, int], grid=LOCATE_GRID) -> Hit | None:
    """Best (erode, sigma) template match for `text` near the predicted pen
    origin (cx, cy). Returns the located pen origin, or None."""
    wx, wy = win
    best = None
    img, px, py = render_strip(text, size)
    h, w = img.shape
    y0 = int(max(0, cy - py - wy))
    y1 = int(min(obs.shape[0], cy - py + h + wy))
    x0 = int(max(0, cx - px - wx))
    x1 = int(min(obs.shape[1], cx - px + w + wx))
    if y1 - y0 < h or x1 - x0 < w:
        return None
    sub = obs[y0:y1, x0:x1]
    for k, sigma in grid:
        tpl = degrade(img, k, sigma)
        if float(tpl.std()) < MIN_TPL_STD:
            continue
        res = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, mxl = cv2.minMaxLoc(res)
        if best is None or mx > best.score:
            best = Hit(float(mx), x0 + mxl[0] + px, y0 + mxl[1] + py,
                       size, (k, sigma))
    return best


def fit_kernel(obs: np.ndarray, hit: Hit, text: str) -> Hit | None:
    """Recover the row's degradation from its own KNOWN label bitmap.

    Self-calibrating: the label beside the destroyed value was damaged by
    the same process, and its content is known a priori, so the (erode,
    sigma) pair that best explains the label is the pair to synthesize
    candidates with. Refines the pen origin at the same time.
    """
    refined = locate(obs, text, hit.x, hit.y, hit.size, (4, 4),
                     grid=KERNEL_GRID)
    return refined


# ------------------------------------------------------------ value decode
def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-6 else 0.0


def canvas(text: str, size: float, deg: tuple[int, float], W: int, H: int,
           pad_x: int, pad_y: int) -> np.ndarray:
    """Candidate rendered into a FIXED (H, W) white canvas with its pen
    origin at (pad_x, pad_y), then degraded.

    The common canvas is the point: per-candidate windows make NCC
    incomparable across candidates of different length, which is exactly
    how a short value loses to a long one on a smeared row.
    """
    key = (text, round(size, 3), deg, W, H, pad_x, pad_y)
    hit = _CANVAS_CACHE.get(key)
    if hit is not None:
        return hit
    img, px, py = render_strip(text, size)
    out = np.full((H, W), 255, np.uint8)
    x0, y0 = int(pad_x - px), int(pad_y - py)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(W, x0 + img.shape[1]), min(H, y0 + img.shape[0])
    if xe > xs and ye > ys:
        out[ys:ye, xs:xe] = img[ys - y0:ye - y0, xs - x0:xe - x0]
    out = degrade(out, *deg)
    if len(_CANVAS_CACHE) < 20000:
        _CANVAS_CACHE[key] = out
    return out


def decode_value(obs: np.ndarray, label_hit: Hit, label: str,
                 candidates: tuple[str, ...], jitter: int = JITTER):
    """Value-only NCC against a common canvas at the label's pen offset.

    Value-only rather than whole-line: the identical label prefix is shared
    by every candidate, so including it only dilutes the discriminative
    region. Returns (value, score, margin, ranked) or None.
    """
    size, deg = label_hit.size, label_hit.deg
    adv = text_width(label + " ", size) * PT2PX
    vx, vy = label_hit.x + adv, label_hit.y
    pad_x = 6
    pad_y = int(round(size * PT2PX * 1.05)) + 4
    W = int(max(text_width(c, size) for c in candidates) * PT2PX) + pad_x + 8
    H = pad_y + int(round(size * PT2PX * 0.35)) + 5
    tpls = [(c, canvas(c, size, deg, W, H, pad_x, pad_y)) for c in candidates]
    best = {c: -2.0 for c in candidates}
    for ddx in range(-jitter, jitter + 1):
        for ddy in range(-jitter, jitter + 1):
            ox = int(round(vx - pad_x + ddx))
            oy = int(round(vy - pad_y + ddy))
            if ox < 0 or oy < 0 or ox + W > obs.shape[1] \
                    or oy + H > obs.shape[0]:
                continue
            patch = obs[oy:oy + H, ox:ox + W]
            for c, t in tpls:
                v = ncc(patch, t)
                if v > best[c]:
                    best[c] = v
    ranked = sorted(((v, c) for c, v in best.items()), reverse=True)
    if not ranked or ranked[0][0] <= -1.9:
        return None                 # no in-bounds window: label is off-page
    margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else ranked[0][0]
    return ranked[0][1], ranked[0][0], margin, ranked


# ------------------------------------------------------------ registration
class Registration:
    """Two independently located anchors that agree with the layout."""

    __slots__ = ("title", "caseid", "dy", "dx")

    def __init__(self, title: Hit, caseid: Hit):
        self.title, self.caseid = title, caseid
        self.dy = caseid.y - title.y
        self.dx = caseid.x - title.x


def register(obs: np.ndarray, case_id: str) -> Registration | None:
    """Locate the two anchors and require them to agree geometrically.

    The case id is KNOWN CONTENT from the filename, used here purely as a
    geometric fiducial: it fixes where the rows are, and nothing about it
    reaches the emitted value. Disagreement means the page is not the
    template we think it is (or one anchor locked onto a smear), and the
    only safe response is to abstain.
    """
    title = locate(obs, INTAKE_TITLE, XCOL, 124.0, TITLE_PT, WIN_TITLE)
    if title is None or title.score < MIN_TITLE_NCC:
        return None
    caseid = locate(obs, f"Case ID: {case_id}", XCOL,
                    title.y + DY_TITLE_CASEID, BODY_PT, WIN_CASEID)
    if caseid is None or caseid.score < MIN_CASEID_NCC:
        return None
    if abs((caseid.y - title.y) - DY_TITLE_CASEID) > DY_TOL:
        return None
    if abs(caseid.x - title.x) > DX_TOL:
        return None
    return Registration(title, caseid)


# --------------------------------------------------------------- page read
def _mode_default(field: str) -> str:
    """The writer's train-mode fallback. Emitting it is a no-op (the output
    row prints it anyway when the field is unread), and default-accepts
    were the only wrong-fire mode ctcfill's calibration measured -- so this
    reader never emits it as a win."""
    from . import writer

    return writer._DEFAULTS[field]


def score_row(obs: np.ndarray, reg: Registration, field: str):
    """Locate the field's row, recover its kernel, rank the menu.

    Returns (label_hit, (winner, score, margin, ranked)) or None. This is
    the shared half of reading and vetoing: both need a confidently
    located row and a full candidate ranking, and they differ only in what
    they are allowed to do with it.
    """
    label = _ROW_LABEL[field]
    predicted_y = reg.caseid.y + _ROW_DY[field]
    coarse = locate(obs, label, reg.caseid.x, predicted_y, BODY_PT, WIN_ROW)
    if coarse is None:
        return None
    hit = fit_kernel(obs, coarse, label)
    if hit is None or hit.score < MIN_LABEL_NCC:
        return None
    candidates = _MENUS[field] + (_SENTINELS[field],)
    got = decode_value(obs, hit, label, candidates)
    if got is None:
        return None
    return hit, got


def read_field(obs: np.ndarray, reg: Registration, field: str
               ) -> tuple[str, float] | None:
    """One registered page's synthesis read for one field, or None."""
    scored = score_row(obs, reg, field)
    if scored is None:
        return None
    hit, got = scored
    value, score, margin, _ranked = got
    if value == _SENTINELS[field]:
        return None                 # row is affirmatively damaged
    if score < MIN_VALUE_NCC or margin < MIN_VALUE_MARGIN:
        return None
    if value in _EXCLUDED_VALUES:
        return None
    if value == _mode_default(field):
        return None
    return value, min(CONF_CAP, hit.score)


class PageCache:
    """Per-case reuse of the expensive per-page work.

    `observe` (crop + background flatten) and `register` (the two anchor
    searches) together are ~0.05 s and depend only on the page, not on
    which field is being asked about. When the fill and the cross-channel
    veto both run on one case they would otherwise pay it twice.
    Registration FAILURES are cached too — a page that will not register
    must not be re-attempted per field.
    """

    __slots__ = ("case_id", "_pages")

    def __init__(self, case_id: str):
        self.case_id = case_id
        self._pages: dict[int, tuple | None] = {}

    def registered(self, index: int, gray: np.ndarray):
        """-> (obs, Registration), or None when the page does not register."""
        if index not in self._pages:
            try:
                obs = observe(gray)
                reg = register(obs, self.case_id)
                self._pages[index] = None if reg is None else (obs, reg)
            except Exception:
                self._pages[index] = None
        return self._pages[index]


def read_page(gray: np.ndarray, case_id: str, fields_needed,
              cache: PageCache | None = None, index: int = 0
              ) -> dict[str, tuple[str, float]]:
    """Register one scan page, then synthesize every needed field on it."""
    got_page = (cache or PageCache(case_id)).registered(index, gray)
    if got_page is None:
        return {}
    obs, reg = got_page
    out = {}
    for field in fields_needed:
        if field not in _ROW_LABEL:
            continue
        got = read_field(obs, reg, field)
        if got is not None:
            out[field] = got
    return out


def _eligible_pages(scans, page_types, budget_left):
    """Intake-typed (or untyped) scan pages, capped and budget-bounded.

    The row table is intake geometry and the two-anchor check needs the
    Case ID row, which the registry and attestation templates do not
    print. Same scoping rule as ctcfill's grid fallback.
    """
    seen = 0
    for index in sorted(scans):
        if seen >= _MAX_PAGES or not budget_left():
            return
        if page_types is not None and \
                page_types.get(index) not in (None, "intake"):
            continue
        seen += 1
        yield index


def fill(scans, fields_needed, case_id: str,
         budget_left=lambda: True,
         page_types: dict[int, str | None] | None = None,
         cache: PageCache | None = None,
         ) -> dict[str, tuple[str, float]]:
    """Gated synthesis fills for still-unread menu fields, across pages.

    Restricted to intake-typed (or untyped) pages: the row table is intake
    geometry, and the two-anchor check needs the Case ID row, which the
    registry and attestation templates do not print. Same scoping rule as
    ctcfill's grid fallback. Cross-page disagreement on a field abstains --
    two pages reading differently is under-determination, not evidence.
    """
    if not fields_needed or not scans:
        return {}
    needed = [f for f in fields_needed if f in _ROW_LABEL]
    if not needed:
        return {}
    if cache is None:
        cache = PageCache(case_id)
    per_field: dict[str, list[tuple[str, float]]] = {f: [] for f in needed}
    for index in _eligible_pages(scans, page_types, budget_left):
        try:
            got = read_page(scans[index].gray, case_id, needed, cache, index)
        except Exception:
            continue
        for field, hit in got.items():
            per_field[field].append(hit)
    fills: dict[str, tuple[str, float]] = {}
    for field, reads in per_field.items():
        if not reads:
            continue
        values = {value for value, _ in reads}
        if len(values) != 1:
            continue
        fills[field] = (values.pop(),
                        min(CONF_CAP, max(c for _, c in reads)))
    return fills


# ------------------------------------------------------ cross-channel veto
# How much the pixel channel's winner must beat the OTHER channel's answer
# by before that answer is withdrawn.
#
# Vetoing is NOT symmetric with filling, because under the challenge scorer
# a wrong value and a blank both score zero:
#
#   veto a WRONG fill   ->  0 becomes the imputed default, gain = P(default
#                           happens to be right), a fraction of a point
#   veto a CORRECT fill ->  1 becomes 0, loss = exactly one point, always
#
# and the loss case is unavoidable rather than merely unlucky: ctcfill
# refuses to emit the writer's mode default, so whenever a vetoed fill was
# RIGHT the value replacing it is wrong by construction. With ctcfill
# filling roughly 12 of 13 correctly, a veto has to be on the order of
# forty times likelier to fire on a wrong fill than a correct one merely to
# break even. So this gate sits deliberately above the acceptance gate:
# the pixel channel must clear its own bar AND beat the contested value
# decisively. Precision here is worth far more than recall.
#
# KILLED ON EVIDENCE — MIB_XCHANNEL_VETO stays default-OFF permanently.
# Censused against the real fill population (31 slots / 27 cases recovered
# from the ab-final vs guard-only predictions diff, scored against gold):
#
#     VETO on WRONG fills    r = 0/3
#     VETO on CORRECT fills  q = 0/28
#
# The veto fires ZERO times. On the four slots where the pixel channel
# formed any opinion it AGREED with ctcfill every time; the two channels
# never decisively disagreed on a single real fill, so the mechanism has
# no signal to act on. Nor can the 48:1 bar ever be established here:
# with three wrong fills in existence, showing q < 1/48 would need ~140
# correct samples against the 28 that exist. The code is retained,
# tested and inert for the record; do not enable it.
VETO_MIN_ADVANTAGE = 0.20


def veto(scans, fills: dict[str, str], case_id: str,
         budget_left=lambda: True,
         page_types: dict[int, str | None] | None = None,
         cache: PageCache | None = None,
         ) -> set[str]:
    """Menu fills this channel decisively contradicts.

    Returns the fields whose fill should be WITHDRAWN, sending them back to
    unread and therefore to the writer's imputed default. It never returns
    a replacement: this channel may retract another channel's answer, not
    substitute its own. Doing both would make it a silent re-decode of the
    field rather than a second opinion, and would smuggle in exactly the
    reads its own acceptance gate declined to make.

    The emission guards deliberately do NOT apply here. A winner that is a
    hard-embargo world or the writer's mode default cannot be EMITTED, but
    it is still evidence that the contested fill is wrong — and withdrawing
    the fill emits neither, it falls back to the default. Only the gates
    that measure whether the read is TRUSTWORTHY apply: label quality,
    winner strength, runner-up margin, and the damage sentinel.

    A page that AGREES with the fill vetoes the veto: corroboration from
    any registered page outranks contradiction from another, since two
    pages disagreeing is under-determination, not proof the fill is wrong.
    """
    if not fills or not scans:
        return set()
    contested = [f for f in fills if f in _ROW_LABEL]
    if not contested:
        return set()
    if cache is None:
        cache = PageCache(case_id)
    agreed: set[str] = set()
    challenged: set[str] = set()
    for index in _eligible_pages(scans, page_types, budget_left):
        got_page = cache.registered(index, scans[index].gray)
        if got_page is None:
            continue
        obs, reg = got_page
        for field in contested:
            try:
                scored = score_row(obs, reg, field)
            except Exception:
                continue
            if scored is None:
                continue
            _hit, (winner, score, margin, ranked) = scored
            if winner == _SENTINELS[field]:
                continue        # row is damaged; no opinion on the value
            if score < MIN_VALUE_NCC or margin < MIN_VALUE_MARGIN:
                continue        # not confident enough to contradict anyone
            held = fills[field]
            if winner == held:
                agreed.add(field)
                continue
            by_value = {value: sc for sc, value in ranked}
            if held not in by_value:
                continue        # cannot compare: the fill is off this menu
            if score - by_value[held] >= VETO_MIN_ADVANTAGE:
                challenged.add(field)
    return challenged - agreed
