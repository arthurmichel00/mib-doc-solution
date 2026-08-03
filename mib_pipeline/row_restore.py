"""Two-rail frame registration for displaced scan bands.

Lineage (all ours; see research/LEDGER.md):
  1. The strip-realignment concept is Arthur's, 2026-07-28 — "reconstruct
     the page as if it was strips of paper ... cut and aligned back
     together". Built and lab-validated as lever W1; the narrow form
     shipped as the sponsor cut-strip weld (ocr.weld_sponsor_lines).
  2. Our Batch B safety battery then rejected runtime realignment: the
     CONTENT-BASED detector fired on 20 of 30 clean pages and cost
     120-600 s/page. LEVERS.md carried the gated variant as the clearest
     documented headroom, "blocked on a gate that is both harmless and
     useful".
  3. The frame-as-ruler gate is Arthur's too, 2026-07-30 — "the strongest
     seams are the left and right border lines, no? why wouldn't it use
     that to align?" Proposed in-house before we read any public code;
     independently confirmed in muhammadbalawal's public MIT solution on
     2026-08-01, and ShreyShingala's public MIT solution measures the
     approach paying within budget in its own pipeline. That external
     evidence informed the decision to build. The implementation here is
     original — see ATTRIBUTION.md.

Damage model: the generator translates a horizontal strip of the page
sideways (and, per Arthur, sometimes shoves a column band vertically).
Where the cut runs through a text row's x-height the glyphs are split
across two offsets, and no threshold or contrast variant can undo it —
photometry cannot fix geometry.

Why two rails. The form draws a full rectangular border, so every raster
row crosses a LEFT and a RIGHT rule. Two independent measurements of the
same row's displacement buy three things a single rail cannot: a page
that draws no vertical rule is rejected outright instead of being traced
off its text; a translation is confirmed by agreement; and a SHEARED band
— where the displacement varies across the page, so dx_near != dx_far by
construction — is both detectable and repairable, which no constant-shift
model can express. Rail agreement therefore selects a repair MODEL, it
does not veto: vetoing on disagreement would discard exactly the sheared
bands.

Acceptance is post-hoc, not predictive: after repairing a band we
re-trace the rails through it and require them to come out straight. A
repair that does not straighten the frame is discarded, whatever the
rails promised beforehand.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

ROWRESTORE_DEFAULT = False
# MIB_ROWRESTORE_FREEFORM: the first integration — feed the registered
# crop to a free-form tesseract pass and pool the lines. MEASURED NO-SHIP,
# kept behind its own dead flag so the finding stays reproducible. Over 28
# triggered bands on train renders: large bands (>160 rows) went 99 -> 75
# confident words, 8 pages degraded vs 3 improved; the small bands'
# apparent +37 words were junk tokens ("~ ~ = . > - ra 5 *" scored as 18
# confident words); no field became readable that was not already; and it
# cost ~204 ms per triggered page. The lesson: the value is not the remap
# feeding ANY reader, it is the remap feeding a CLOSED-VOCABULARY one,
# which is what MIB_ROWRESTORE enables (ctcfill.fill_restored).
FREEFORM_DEFAULT = False

# --- rail detection --------------------------------------------------------
_INK_LEVEL = 145                 # grey level below which a pixel is ink
# A frame rule survives a vertical opening this tall; a glyph stroke does
# not run unbroken for an eighth of an inch down the page.
_OPEN_FRACTION = 0.125           # of dpi, in pixels
# Search zones as fractions of width, measured from each edge. The rules
# sit ~6% in on our renders (x 126-147 and 2288-2309 on a 2448-px page);
# the zones are wide enough to hold a rule displaced by _MAX_SHIFT.
_ZONE_INNER = 0.28
_ZONE_OUTER = 0.008
# A page carries a usable rail only when some column in the zone is inked
# down this much of the page after the opening. Measured on train renders:
# real rules score 0.79-0.91, and pages drawn with horizontal rules only
# score nothing above 0.25 — that second population is what a single-rail
# text-following trace fires garbage on.
_MIN_RAIL_COVERAGE = 0.25
_MIN_OBSERVED_FRACTION = 0.12    # lines carrying the rule before we fit it

# --- robust fit ------------------------------------------------------------
# Theil-Sen: the median of pairwise slopes. No iteration, no tuning
# constant, ~29% breakdown, and unlike a trimmed least-squares refit it
# cannot be walked away from the truth by a long run of correlated
# outliers (which is exactly what a displaced band is).
_FIT_SAMPLE = 240                # observed lines sampled for the slope median

# --- displacement gate -----------------------------------------------------
# Minimum displacement, in pixels at 288 DPI, that counts as a real
# translation rather than trace noise. Matches the cut-strip weld's own
# floor for a meaningful horizontal offset (ocr._WELD_MIN_DX), derived
# from the same displaced-content damage family.
_MIN_SHIFT_PX = 12
_MAX_SHIFT_FRACTION = 0.16       # of width; beyond this the fit is nonsense
_MIN_BAND_FRACTION = 0.010       # a band is at least ~one text row tall
_MAX_BAND_FRACTION = 0.50        # more than this and the fit found the defect
_BAND_GAP_FRACTION = 0.004       # lines this close count as one run
_MIN_BAND_DENSITY = 0.5          # a strip is contiguous, not scattered
_MIN_BAND_OBSERVED = 0.60        # the rule must be SEEN across the band
_BAND_FLATNESS_PX = 3.0          # ... and read a FLAT offset across it
_MIN_BRACKET_FRACTION = 0.02     # undamaged rule above AND below the strip

# --- model selection -------------------------------------------------------
_TRANSLATION_TOL_PX = 6          # rails this close agree: pure translation
_SHEAR_MIN_SPREAD_PX = 10        # below this a "shear" is noise on a shift
_SHEAR_STABILITY_PX = 8          # a shear is a steady ramp, not a wobble
# One rail only: the page is half-blind, so demand a bigger displacement
# and a cleaner band before touching it.
_SINGLE_RAIL_SHIFT_MULTIPLIER = 2.0
_SINGLE_RAIL_MIN_OBSERVED = 0.80

# --- post-repair validation ------------------------------------------------
_VALIDATE_RESIDUAL_PX = 4.0      # rails must land this close to baseline

# The probe reads every Nth raster line at FULL cross resolution: the
# quantity measured is an offset along the other axis, so dropping lines
# costs no accuracy on it — only the band edges coarsen, which the repair
# margin absorbs.
_PROBE_STRIDE = 3
_BAND_MARGIN_INCHES = 0.25

HORIZONTAL, VERTICAL = 0, 1


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


def enabled() -> bool:
    """MIB_ROWRESTORE: detection + repair + the closed-menu CTC channel."""
    return _flag("MIB_ROWRESTORE", ROWRESTORE_DEFAULT)


def freeform_enabled() -> bool:
    """MIB_ROWRESTORE_FREEFORM: the condemned free-form re-read pass."""
    return _flag("MIB_ROWRESTORE_FREEFORM", FREEFORM_DEFAULT)


# MIB_ROWRESTORE_VERTICAL: the transposed pass. Implemented and unit
# tested, DEFAULT OFF because its corpus detections are unverified: on
# the 255-page probe set it repairs 5 pages against horizontal's 1, and
# the one case examined by hand (MIB-000705 p1) turned out to be a rule
# the tracker had swapped onto rather than a displaced band. The swap
# guard now rejects that case, but the remaining five have not been
# checked one by one, and an unverified detector does not go in an arm.
VERTICAL_DEFAULT = False


def vertical_enabled() -> bool:
    """MIB_ROWRESTORE_VERTICAL: also sweep the transposed axis."""
    return _flag("MIB_ROWRESTORE_VERTICAL", VERTICAL_DEFAULT)


def min_shift_px(dpi: float) -> float:
    """Displacement trigger in pixels, scaled to the render resolution."""
    override = os.environ.get("MIB_ROWRESTORE_MIN_SHIFT")
    base = float(_MIN_SHIFT_PX)
    if override:
        try:
            base = float(override)
        except ValueError:
            pass
    return max(2.0, base * dpi / 288.0)


@dataclass(frozen=True)
class Rail:
    """One frame rule traced down the page, with its robust baseline."""

    far: bool                # False = near edge (low coordinate)
    position: np.ndarray     # measured offset per line; NaN where unseen
    observed: np.ndarray     # lines where the rule was actually found
    baseline: np.ndarray     # Theil-Sen fit over the observed lines
    mask: np.ndarray         # rule-ink in the search zone
    zone_lo: int             # column the zone starts at

    @property
    def residual(self) -> np.ndarray:
        return self.position - self.baseline

    def inked(self, line: int, offset: float) -> bool:
        """Is there rule-ink at this offset on this line?"""
        column = int(round(offset)) - self.zone_lo
        if not (0 <= line < self.mask.shape[0]):
            return False
        lo = max(0, column - 2)
        hi = min(self.mask.shape[1], column + 3)
        return bool(lo < hi and self.mask[line, lo:hi].any())


@dataclass(frozen=True)
class Band:
    """A displaced strip and the repair model its rails imply."""

    top: int
    bottom: int
    lines: int               # displaced lines inside the span
    model: str               # "translation" | "shear" | "single"
    near_dx: float           # displacement measured at the near rail
    far_dx: float            # ... and at the far rail (NaN if unobserved)
    near_x: float            # baseline offset of the near rail
    far_x: float

    @property
    def spread(self) -> float:
        """dx_far - dx_near: zero for a translation, the shear otherwise."""
        if not (np.isfinite(self.far_dx) and np.isfinite(self.near_dx)):
            return float("nan")
        return float(self.far_dx - self.near_dx)

    @property
    def peak(self) -> float:
        return float(np.nanmax(np.abs([self.near_dx, self.far_dx])))


def _theil_sen(lines: np.ndarray, offsets: np.ndarray) -> tuple[float, float]:
    """Median of pairwise slopes, then the median intercept.

    Sampled to a bounded number of points so the pair count stays flat in
    page height; the sample is evenly spaced, so it spans the whole page
    instead of clustering wherever the rule is densest.
    """
    if len(lines) > _FIT_SAMPLE:
        pick = np.linspace(0, len(lines) - 1, _FIT_SAMPLE).astype(int)
        lines, offsets = lines[pick], offsets[pick]
    rows = np.triu_indices(len(lines), k=1)
    run = lines[rows[0]] - lines[rows[1]]
    rise = offsets[rows[0]] - offsets[rows[1]]
    usable = run != 0
    slope = float(np.median(rise[usable] / run[usable])) if usable.any() else 0.0
    return slope, float(np.median(offsets - slope * lines))


def _rule_mask(image: np.ndarray, dpi: float) -> np.ndarray:
    """Ink that survives a vertical opening: rules, not glyphs."""
    ink = (image < _INK_LEVEL).astype(np.uint8)
    tall = max(8, int(_OPEN_FRACTION * dpi))
    return cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, tall)))


def _anchor(zone: np.ndarray) -> tuple[float, float] | None:
    """(column, width) of the rule, from the page-wide coverage profile.

    The rule is the one structure that inks the SAME column down the whole
    page, so the coverage peak locates it and the shoulder around that
    peak measures how thick it is drawn. Both are then used per line to
    tell the rule apart from whatever else survived the opening.
    """
    coverage = zone.sum(axis=0).astype(np.float64)
    peak = float(coverage.max())
    if peak < _MIN_RAIL_COVERAGE * zone.shape[0]:
        return None
    column = int(coverage.argmax())
    thick = coverage >= 0.5 * peak
    lo = column
    while lo > 0 and thick[lo - 1]:
        lo -= 1
    hi = column
    while hi + 1 < len(thick) and thick[hi + 1]:
        hi += 1
    return 0.5 * (lo + hi), float(hi - lo + 1)


def _nearest_run(zone: np.ndarray, left: int, anchor: float, width: float,
                 reach: float) -> np.ndarray:
    """Centre of the run nearest the anchor on each line; NaN if none.

    Nearest — not widest. The rule can be displaced, but only by a bounded
    amount, whereas body text sits wherever the layout put it; picking by
    proximity to the page-wide anchor follows the rule into a displaced
    band without ever locking onto a column of glyphs. Runs far thicker
    than the rule is drawn are rejected outright.

    Run edges come from the column-wise difference of the padded mask, so
    the whole zone resolves in two vectorized passes, not a per-line scan.
    """
    height, span = zone.shape
    padded = np.zeros((height, span + 2), np.int8)
    padded[:, 1:-1] = zone
    edges = np.diff(padded, axis=1)
    line, start = np.nonzero(edges == 1)
    _, end = np.nonzero(edges == -1)
    out = np.full(height, np.nan)
    if line.size == 0:
        return out
    centre = left + 0.5 * (start + end - 1)
    keep = (end - start) <= max(3.0, 3.0 * width)
    distance = np.abs(centre - (left + anchor))
    keep &= distance <= reach
    if not keep.any():
        return out
    line, centre, distance = line[keep], centre[keep], distance[keep]
    order = np.lexsort((distance, line))
    ordered = line[order]
    leading = np.ones(ordered.size, bool)
    leading[1:] = ordered[1:] != ordered[:-1]
    chosen = order[leading]
    out[line[chosen]] = centre[chosen]
    return out


def _trace_rail(image: np.ndarray, dpi: float, far: bool) -> Rail | None:
    """Trace one frame rule; None when this page does not draw one.

    Page-wide column coverage decides whether a rule EXISTS at all (a
    glyph stem never inks a quarter of the page height, which is what
    stops this from following a column of text). The per-line pick then
    follows the rule wherever a defect moved it, by taking the widest
    surviving run in the zone rather than the one nearest to where the
    rule ought to be.
    """
    height, width = image.shape
    if far:
        lo = int((1.0 - _ZONE_INNER) * width)
        hi = int((1.0 - _ZONE_OUTER) * width)
    else:
        lo, hi = int(_ZONE_OUTER * width), int(_ZONE_INNER * width)
    lo, hi = max(0, lo), min(width, hi)
    if hi - lo < 16:
        return None
    zone = _rule_mask(np.ascontiguousarray(image[:, lo:hi]), dpi)
    found = _anchor(zone)
    if found is None:
        return None
    anchor, thickness = found
    position = _nearest_run(zone, lo, anchor, thickness,
                            _MAX_SHIFT_FRACTION * width)
    observed = np.isfinite(position)
    if observed.sum() < _MIN_OBSERVED_FRACTION * height:
        return None
    lines = np.flatnonzero(observed).astype(np.float64)
    slope, intercept = _theil_sen(lines, position[observed])
    baseline = slope * np.arange(height, dtype=np.float64) + intercept
    return Rail(far=far, position=position, observed=observed,
                baseline=baseline, mask=zone, zone_lo=lo)


def _candidate_spans(rails: list[Rail], height: int,
                     threshold: float) -> list[tuple[int, int, int]]:
    """Contiguous runs of lines that at least one rail reports as moved."""
    moved = np.zeros(height, bool)
    for rail in rails:
        residual = np.abs(rail.residual)
        moved |= rail.observed & (residual >= threshold)
    hits = np.flatnonzero(moved)
    if hits.size == 0 or hits.size > _MAX_BAND_FRACTION * height:
        return []
    gap = max(2, int(_BAND_GAP_FRACTION * height))
    spans = []
    for run in np.split(hits, np.flatnonzero(np.diff(hits) > gap) + 1):
        top, bottom = int(run[0]), int(run[-1]) + 1
        if run.size < max(1, int(_MIN_BAND_FRACTION * height)):
            continue
        if bottom - top > _MAX_BAND_FRACTION * height:
            continue
        if run.size < _MIN_BAND_DENSITY * (bottom - top):
            continue
        spans.append((top, bottom, int(run.size)))
    return spans


def _rail_dx(rail: Rail | None, top: int, bottom: int) -> float:
    """Median displacement this rail reports inside the span, or NaN.

    A displaced rule is still a straight segment, so its residual has to
    be FLAT across the band. Measured on train renders: when the tracker
    loses the rule and locks onto some other vertical structure in the
    zone, the residual wanders — and every one of those mis-locks
    presented as a large clean-looking displacement on one rail with the
    other rail reading zero. Requiring flatness is what tells a strip
    that moved apart from a rule the tracker lost.
    """
    if rail is None:
        return float("nan")
    seen = rail.observed[top:bottom]
    if seen.mean() < _MIN_BAND_OBSERVED:
        return float("nan")
    inside = rail.residual[top:bottom][seen]
    middle = float(np.median(inside))
    if float(np.median(np.abs(inside - middle))) > _BAND_FLATNESS_PX:
        return float("nan")
    return middle


def _bracketed(rail: Rail, top: int, bottom: int, height: int,
               threshold: float) -> bool:
    """Undamaged rule above AND below: a strip is cut OUT of the frame.

    A page edge carries evidence on one side only, which is what a footer
    below the border, or a rule that fades out halfway down the page,
    both look like.
    """
    need = max(4, int(_MIN_BRACKET_FRACTION * height))
    steady = rail.observed & (np.abs(rail.residual) < threshold)
    return bool(steady[:top].sum() >= need and steady[bottom:].sum() >= need)


def _moved_not_swapped(rail: Rail, top: int, bottom: int, dx: float) -> bool:
    """True when the rule really MOVED, rather than the tracker swapping
    onto a different structure that was always there.

    Failure mode this exists for, measured on train renders: where the
    border is simply ABSENT over part of the page, the nearest-run pick
    latches onto the next rule along and reports the gap between them as
    a displacement (MIB-000705 p1, vertical axis: the top border is
    missing over columns 1401-1643 and the tracker locked onto a table
    separator 162 px below it). The tell is that the "displaced"
    position is occupied OUTSIDE the band too — a real displacement
    vacates its origin and lands somewhere that was empty.
    """
    if abs(dx) < 1.0:
        return True
    height = rail.mask.shape[0]
    probes = [line for line in
              (top - 12, top - 40, bottom + 12, bottom + 40)
              if 0 <= line < height]
    if not probes:
        return True
    # Zero tolerance: ANY probe finding ink where the rule supposedly
    # landed means that position was already occupied, so the rule did
    # not move there — something else was always there. Hand-verification
    # of all six vertical detections separates perfectly on this count:
    # the one real displacement scores 0/4, and every mis-lock scores
    # 1/4 or 2/4. A majority vote (the first cut) passed all six.
    return not any(rail.inked(line, rail.baseline[line] + dx)
                   for line in probes)


def _classify(near: Rail | None, far: Rail | None, top: int, bottom: int,
              lines: int, height: int, threshold: float) -> Band | None:
    """Pick the repair model this span's rails support, or abstain."""
    near_dx, far_dx = _rail_dx(near, top, bottom), _rail_dx(far, top, bottom)
    if np.isfinite(near_dx) and not _moved_not_swapped(near, top, bottom,
                                                       near_dx):
        near_dx = float("nan")
    if np.isfinite(far_dx) and not _moved_not_swapped(far, top, bottom,
                                                      far_dx):
        far_dx = float("nan")
    have_near, have_far = np.isfinite(near_dx), np.isfinite(far_dx)
    if not (have_near or have_far):
        return None
    near_x = float(np.median(near.baseline[top:bottom])) if near else float("nan")
    far_x = float(np.median(far.baseline[top:bottom])) if far else float("nan")

    if have_near and have_far:
        if max(abs(near_dx), abs(far_dx)) < threshold:
            return None
        if not (_bracketed(near, top, bottom, height, threshold)
                or _bracketed(far, top, bottom, height, threshold)):
            return None
        spread = abs(far_dx - near_dx)
        if spread <= _TRANSLATION_TOL_PX:
            model = "translation"
        elif spread >= _SHEAR_MIN_SPREAD_PX:
            both = near.observed[top:bottom] & far.observed[top:bottom]
            if both.sum() < 4:
                return None
            delta = (far.residual[top:bottom][both]
                     - near.residual[top:bottom][both])
            if float(np.max(np.abs(delta - np.median(delta)))) \
                    > _SHEAR_STABILITY_PX:
                return None          # a wobble, not a ramp
            model = "shear"
        else:
            return None              # between the models: no confident fit
        return Band(top=top, bottom=bottom, lines=lines, model=model,
                    near_dx=near_dx, far_dx=far_dx, near_x=near_x,
                    far_x=far_x)

    rail = near if have_near else far
    dx = near_dx if have_near else far_dx
    if abs(dx) < threshold * _SINGLE_RAIL_SHIFT_MULTIPLIER:
        return None
    if rail.observed[top:bottom].mean() < _SINGLE_RAIL_MIN_OBSERVED:
        return None
    if not _bracketed(rail, top, bottom, height, threshold):
        return None
    return Band(top=top, bottom=bottom, lines=lines, model="single",
                near_dx=near_dx, far_dx=far_dx, near_x=near_x, far_x=far_x)


def _oriented(image: np.ndarray, axis: int) -> np.ndarray:
    """The page as the detector wants it: transposed for the vertical axis.

    A column band shoved up or down is the same defect as a row band
    shoved sideways, seen from the other side — and the form's border is
    a full rectangle, so the top and bottom rules serve the transposed
    pass exactly as the left and right ones serve the upright pass. The
    entire detector is reused; only this call differs.
    """
    return image if axis == HORIZONTAL else np.ascontiguousarray(image.T)


def detect(image: np.ndarray, dpi: float, axis: int = HORIZONTAL,
           threshold: float | None = None) -> list[Band]:
    """Displaced bands on one page, each with the repair model it supports."""
    work = _oriented(image, axis)
    if min(work.shape) < 32:
        return []
    limit = min_shift_px(dpi) if threshold is None else threshold
    near = _trace_rail(work, dpi, far=False)
    far = _trace_rail(work, dpi, far=True)
    rails = [r for r in (near, far) if r is not None]
    if not rails:
        return []
    height = work.shape[0]
    found = []
    for top, bottom, lines in _candidate_spans(rails, height, limit):
        band = _classify(near, far, top, bottom, lines, height, limit)
        if band is not None:
            found.append(band)
    return found


def _shift_map(band: Band, shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel correction for one band.

    translation / single: one offset across the strip.
    shear: the offset ramps linearly between the two rails. This is the
    model a single rail cannot express — a row whose two ends moved by
    different amounts is not a translation at any offset.
    """
    rows, width = shape
    if band.model == "shear":
        span = band.far_x - band.near_x
        if abs(span) < 1.0:
            ramp = np.full(width, band.near_dx, np.float32)
        else:
            t = (np.arange(width, dtype=np.float32) - band.near_x) / span
            ramp = (band.near_dx + t * (band.far_dx - band.near_dx)
                    ).astype(np.float32)
        return np.repeat(ramp[None, :], rows, axis=0)
    if band.model == "translation":
        dx = float(np.nanmean([band.near_dx, band.far_dx]))
    else:
        dx = band.near_dx if np.isfinite(band.near_dx) else band.far_dx
    return np.full((rows, width), dx, np.float32)


def _remap(strip: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Undo the measured displacement, white-filling what it vacates."""
    rows, width = strip.shape
    limit = _MAX_SHIFT_FRACTION * width
    map_x = np.ascontiguousarray(
        np.arange(width, dtype=np.float32)[None, :]
        + np.clip(shift, -limit, limit))
    map_y = np.ascontiguousarray(
        np.repeat(np.arange(rows, dtype=np.float32)[:, None], width, axis=1))
    return cv2.remap(strip, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _straightened(repaired: np.ndarray, dpi: float, band: Band,
                  offset: int) -> bool:
    """Do the rails come out straight through the repaired band?

    A predictive gate can only ever guess; this measures the result. A
    repair that leaves the frame bent did not model the damage, whatever
    the rails promised beforehand.
    """
    lo = max(0, band.top - offset)
    hi = min(repaired.shape[0], band.bottom - offset)
    if hi - lo < 8:
        return False
    checked = False
    for far, expected in ((False, band.near_x), (True, band.far_x)):
        if not np.isfinite(expected):
            continue
        rail = _trace_rail(repaired, dpi, far=far)
        if rail is None:
            continue
        seen = rail.observed[lo:hi]
        if seen.sum() < 4:
            continue
        checked = True
        landed = rail.position[lo:hi][seen]
        if float(np.median(np.abs(landed - expected))) > _VALIDATE_RESIDUAL_PX:
            return False
    return checked


@dataclass(frozen=True)
class Registration:
    """The outcome of one page's repair attempt."""

    image: np.ndarray            # page with the accepted bands registered
    repaired: list[Band]
    skipped_no_need: list[Band]  # detected, but no needed row inside them
    rejected: list[Band]         # repaired, then failed straightening
    axis: int = HORIZONTAL

    @property
    def spans(self) -> list[tuple[int, int]]:
        return [(b.top, b.bottom) for b in self.repaired]


def shift_at(bands: list[Band], line: int, across: float) -> float:
    """Correction this registration applied at one point, 0.0 if none.

    Lets a caller carry geometry it already measured (OCR word boxes)
    onto the repaired image instead of re-reading to rediscover it.
    """
    for band in bands:
        if not (band.top <= line < band.bottom):
            continue
        if band.model == "shear":
            span = band.far_x - band.near_x
            if abs(span) < 1.0:
                return float(band.near_dx)
            t = (across - band.near_x) / span
            return float(band.near_dx + t * (band.far_dx - band.near_dx))
        if band.model == "translation":
            return float(np.nanmean([band.near_dx, band.far_dx]))
        return float(band.near_dx if np.isfinite(band.near_dx)
                     else band.far_dx)
    return 0.0


def _intersects(top: int, bottom: int,
                wanted: list[tuple[int, int]] | None) -> bool:
    if wanted is None:
        return True
    return any(top < hi and bottom > lo for lo, hi in wanted)


def register(image: np.ndarray, dpi: float,
             wanted: list[tuple[int, int]] | None = None,
             axis: int = HORIZONTAL) -> Registration | None:
    """Repair the displaced bands that carry a line somebody needs.

    `wanted` lists the line spans the caller actually intends to read —
    for us, the rows where an unread closed-menu field can live. Bands
    intersecting none of them are detected and counted, never repaired:
    repairing a content-free strip only ever cost us junk, and the rec
    inference behind it is the expensive part. Pass None to repair every
    detected band.

    None when nothing was repaired, so callers can treat "no defect" and
    "nothing worth fixing" identically.
    """
    if not probe(image, dpi, axis=axis):
        return None          # strided pre-filter; an undamaged page ends here
    bands = detect(image, dpi, axis=axis)
    if not bands:
        return None
    work = _oriented(image, axis)
    out = work.copy()
    repaired, skipped, rejected = [], [], []
    for band in bands:
        if not _intersects(band.top, band.bottom, wanted):
            skipped.append(band)
            continue
        # EXACTLY the displaced lines. Padding the repair would drag
        # undamaged neighbours along with it and break the very lattice
        # the repair exists to restore; context padding is the reader's
        # job, at crop time.
        strip = np.ascontiguousarray(work[band.top:band.bottom])
        fixed = _remap(strip, _shift_map(band, strip.shape))
        if not _straightened(fixed, dpi, band, band.top):
            rejected.append(band)
            continue
        out[band.top:band.bottom] = fixed
        repaired.append(band)
    if not repaired:
        return None
    return Registration(image=_oriented(out, axis), repaired=repaired,
                        skipped_no_need=skipped, rejected=rejected, axis=axis)


def probe(image: np.ndarray, dpi: float, axis: int = HORIZONTAL) -> bool:
    """Cheap "is anything displaced here" test on a strided page.

    Full cross-axis resolution is kept, so the offset measurement stays
    exact; only the line sampling coarsens. An undamaged page stops here
    and never pays for the full-resolution pass.
    """
    work = _oriented(image, axis)
    if min(work.shape) < 8 * _PROBE_STRIDE:
        return False
    thin = np.ascontiguousarray(work[::_PROBE_STRIDE])
    return bool(detect(thin, dpi / _PROBE_STRIDE, axis=HORIZONTAL,
                       threshold=min_shift_px(dpi)))
