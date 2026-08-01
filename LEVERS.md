# LEVERS — what was built, what shipped, and why the rest did not

Every lever attempted during the build, with its verdict and the measurement that produced it. Derived from
the project's append-only approach ledger, which existed so no loop iteration would re-litigate a settled
dead end. Scores are on the official evaluator: train /150 over 1,000 cases, holdout = the fixed 200-case
seed-8090 subset. Referenced from MEMO.md and APPENDIX.md.

**Verdict keys.** `SHIPPED-GATED` in the tree behind a flag or decision · `SHIPPED-DORMANT` in the tree,
default off · `REJECTED-ON-SAFETY` works, but harms clean cases or cannot be gated · `NO-GO-ON-VALUE`
measured net-negative or net-zero · `DEAD-END` confirmed no headroom · `VERIFIED-SAFE` audited, no change
needed · `STAGED` built and validated, deliberately not in the frozen image · `DEFERRED` post-submission
queue. The shipped configuration itself is enumerated in MEMO.md and SUBMISSION.md.

## Shipped

| Lever | What it was | Verdict | Evidence |
|---|---|---|---|
| Fuzzy Finding/Reason recovery | Partial-ratio matching of Finding phrases and Reason templates, with a runner-up margin bar | SHIPPED-GATED | Red-team 4/4 twice; 16.4k-line OCR corpus sweep, 131/131 gold-consistent; recovers notes whose headers are garbled beyond exact match |
| Emission consistency guard | On note-decided rows, suppress deny-trigger field values the trusted-note decision ignored | SHIPPED-GATED | +0.01 train, holdout flat, CFA=1, red-team 4/4; removes all 47 rows an automated reviewer could flag as self-contradictory; adjudication decisions train-identical |
| Reason-template adjudication | Seven 100%-gold-verified Reason templates allowed to adjudicate alone; organizer-seeded trap templates hard-excluded | SHIPPED-GATED | Clean full-train A/B, both arms 0 fallbacks: train 128.35 → 128.86 (+0.51), holdout 129.38 → 129.68 (+0.30); 3 correct adjudication flips, no gold-DENIED case moved toward APPROVED; CFA=1; red-team 4/4 with the flag on |
| PP-OCR generation choice | Which bundled PaddleOCR ONNX generation serves as the second recognizer | SHIPPED-GATED (v6) | Benchmarked rather than assumed: v5 was a wash over v4 (+0.07) and was rejected; v6 measured +0.81 on the full official train set |
| Preprocessing escalation + restore guard | Divide-by-blur, Sauvola, 2× Lanczos and their composition, behind a lazy trigger | SHIPPED-GATED | Fires only when both engines leave a decision input unread; the restore guard makes escalated reads fill-only, so they can never out-vote an affirmative read |
| Candidate-trained CRNN line recognizer | ~6M-param CRNN+CTC, 17 MB ONNX, trained from scratch on synthetic renders through a measured damage chain | SHIPPED-GATED | 45.3% exact-line accuracy on reserved real crops never trained on; +0.05 measured standalone; fill-only, cannot mint a Finding, sponsor evidence dropped at source; name arbitration audited 5 corrected / 0 broken |
| Sponsor strip-weld | Detect a cut line in a text row, estimate displacement by maximizing ink continuity, re-weld and read | SHIPPED-GATED (extraction-only) | Recovers 2 sponsor ids on train, both matching gold, 0 revoked-id false fills; after external review, weld confidence is capped below the affirmative-read threshold so adjudication can never rest on a reconstructed read |
| Green-stamp rescue | Rescue-only stamp path, ranked below every deny rule | SHIPPED-GATED (insurance) | 0 flips on train (the trusted note path already covers every stamped case there); ships because the field manual makes the stamp co-top evidence, so the private set may contain the population train does not |
| Calibration provenance refit | Refit the decision-path statistics table from in-container (serve-environment) OCR | SHIPPED | Closes a fit/serve provenance gap found in self-audit; Brier improved to 0.0772; re-passed every gate (train ≥ baseline, holdout ≥ train, single designed-trap CFA) |

## Shipped dormant (in the image, default off)

| Lever | What it was | Verdict | Evidence |
|---|---|---|---|
| Green-stamp detector as a decision path | Use the adjudicator's green APPROVED stamp to decide a case | SHIPPED-DORMANT (diagnostic only) | Precision 1.000 on train (33/33) but 100% redundant with the trusted note path; ships as detect-and-log, emitted rows proven byte-identical with it on or off, 0.55% of runtime |
| Rotation probe | Anchor-scored orientation selection | SHIPPED-DORMANT | 0/4 flips, no synergy with escalation, and the worst timing of any variant measured (7.25 s/PDF combined) |
| Structural discharge heads | Targeted re-reads of the exact evidence item blocking a review, then re-run the cascade | SHIPPED-DORMANT | 0/14 readable blocked cases recovered: the blocking ink defeats every runtime engine. The frame is proven safe and activates only if reading power lands |
| Quarantined note-band re-read | Re-read the note band at native raster scale, quarantined from the shared line pool | SHIPPED-DORMANT (native-only) | Mints one additional correct approval under the dev Tesseract but not under the image's 5.3 — the same fit/serve OCR gap the calibration refit addressed. 320-case sweep: 1 mint, 0 mismatches. Stays off |

## Rejected, dead, and net-zero

| Lever | What it was | Verdict | Evidence |
|---|---|---|---|
| Fragment realignment (regional reconstruction) | Detect displaced text fragments, re-assemble them geometrically, re-read the page | REJECTED-ON-SAFETY | Detector fires candidates on 20/30 clean control pages (up to 64 on one clean page), so "gate it to damaged pages" is not a gate. The only available gate zeroes 100% of the harm and 100% of the recoveries alike. Repair produces a parser-accepted wrong value on 2/30 clean cases and degrades a correct baseline on 4/30. Detection costs 120–600 s/page against a 6.0 s budget |
| Multi-view rotation escalation | Best-rotation view + 180° hedge + border pad, pooled into the OCR stream | NO-GO-ON-VALUE | Clean 2-view full train 128.71 / 129.61 vs baseline 128.86 / 129.68 (−0.15 train, −0.07 holdout), measured twice. Per-case diff: +11 across 2 wins against −41 across 5 harms. Pooling injects plausible-but-wrong reads that outvote correct ones, landing on deny-trigger fields. A margin gate is moot: it drops the harmless hedges and keeps the harmful ones |
| Reading-coverage widening | Let escalation retry fields that were read, not only blank ones | REJECTED-ON-SAFETY | Reclaims 1.7% of missed fields (10/592, all marginal single-character) against ~20% harm on clean cases, including a visa misread of the catastrophic-false-approval class. Validates the existing design: escalate only on blank |
| Background-subtraction escalation | Median-background subtraction with residual amplification, appended to the escalation ladder | NO-GO-ON-VALUE | Three full-train runs (off, raw, fill-only) printed identical totals of 128.86 / 129.68. Field diff is 3 cases: one sponsor recovered, one name corrupted, one neutral wrong fill. The fill-only guard did not stop the corruption, because noisy candidate lines persist in the page pool and are rebuilt by later reconciliation. Default off |
| Confidence-pin relaxation | Unpin the cases whose fields are correct but whose verdict is pinned to NEEDS_REVIEW | DEAD-END (disqualification trap) | The 74 pinned cases are 65 gold-APPROVED and 9 gold-DENIED, with no signal separating them. Naive unpinning mints 9 catastrophic false approvals, taking the gate from 1 to 10. The pin is calibration-correct, not lossy |
| Fee-value shape classifier | Template shape-matching of fee glyphs against the OCR hypothesis | DEAD-END | Synthetic-validated at 0 wrong accepts, but on the real damaged cases it reads 0: it needs the clean fixed-box receipt geometry that the damage destroys, so it cannot even locate the fee region |
| Sponsor-digit shape recovery | Recover SPN-#### digits on degraded scans by glyph-shape hypothesis matching | DEAD-END | Fixed-format numerics share near-identical glyph geometry; the correct candidate ranks 52/200, which is chance. Recorded so no future loop re-litigates it |
| Closed-menu template matching | Match a damaged field against the closed menu of legal values by locked-geometry template correlation | NO-GO-ON-VALUE | Ranks the correct home world 1 of 13, but at a margin statistically indistinguishable from a control that was wrong, and the same matcher mints confident winners on unrelated rows. Narrows a menu to a shortlist; cannot mint a value. One narrow variant (declared purpose, 13× the margin) is queued, gated on prefix-fit and margin, and never for sponsor digits |
| Fee inference from document absence | Treat "no receipt page in the packet" as evidence of a waived fee | DEAD-END | Tested corpus-wide after the pattern appeared in a directed review: the 381 no-receipt cases are 263 paid / 89 waived / 14 unpaid / 15 unknown, and the runtime-legal subgroup splits the same way. Mode-impute survives a third audit. Do not revisit doc-absence fee conditioning |
| Blanket strip-weld (all fields) | Apply the sponsor weld mechanism to every field | REJECTED-ON-SAFETY | +0.033 candidate-level upside against 8 wrong parses, 3 of them on decision-bearing fee rows |
| Sentinel-collision audit | Check whether the fee "unknown" sentinel can collide with a real read | DEAD-END (protective) | Already layer-split in our representation; no collision exists. Audited and closed |
| Conflict-rule conformance | Verify the field manual's note-over-stamp precedence and case-id binding rules | VERIFIED-SAFE | 0 of 64 note-approved cases ever flip to DENIED; 27/27 applicant bindings correct with decoys rejected. No fix needed |

## Staged and deferred

| Lever | What it was | Verdict | Evidence |
|---|---|---|---|
| Truncation-prefix guard | Abstain when the matched label zone is a truncation prefix of a longer vocabulary label, instead of binding it to the nearest match | STAGED (not in the frozen image) | Fixes the one note misread disclosed in MEMO.md. Full battery green: 435 tests including 40 new, red-team 4/4, template suite 6/6, a 23,066-line corpus sweep with 251/253 mints unchanged and both changes benign, and an in-container A/B whose only adjudication change was the target case. Deliberately held out of the frozen image rather than slipped in after the freeze |
| Label-anchored rotation detection with faint-preserving stretch | Detect page orientation by fuzzy-matching label text at each rotation, before a page-global binarization kills faint ink | DEFERRED | Diagnosed on a case where a human reads the rotated page but no cell of a 4-page × 4-rotation × 5-enhancement × 3-mode grid recovers the value, because light-grey ink dies under global Otsu |
| Note-scoped sparse re-read | Sparse-mode OCR scoped to the note band, feeding the existing parser | DEFERRED | Targets ink-bloat notes that are eye-readable but defeat character segmentation. Needs a full A/B, red-team re-pass, and rebuild before it could ship |
| Multi-segment fragment reassembly (gated variant) | A version of realignment whose detector does not fire on clean pages | DEFERRED | The clearest documented headroom, blocked on a gate that is both harmless and useful. Ships if and when one exists |

## Standing conclusions

- **The binding constraint is reading power on the hard tail**, not orientation plumbing, not adjudication
  logic, and not calibration. Six independent lines of evidence converge on it: escalation net-negative,
  template adjudication 1/19, discharge heads 0/14, realignment 0/41 addressable, the confidence pin as a
  disqualification trap, and the realignment safety battery.
- **Generic mechanisms are judged on harm, validity, and cost — not on train-set gain.** A lever that adds
  nothing on train can still ship as insurance if it is provably harmless, and a lever that gains on train
  is still killed if it harms clean cases.
- **Answer-key and injection channels are anti-signal** and are never read; hidden spans are stripped before
  rasterization.
- **Scoring runs need a quiet machine.** The per-case wall-clock deadline means concurrent load manufactures
  fallback rows and depresses the score, so no two full-train scoring runs ever overlapped.
