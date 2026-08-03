# APPENDIX — MIB Doc Challenge

Referenced from MEMO.md; this file carries the evidence detail the 2-page memo summarizes.

## 1. Approach in full

The pipeline is built around one idea: decide the trust boundary before reading anything.

1. **Redact-then-render.** Hidden spans (render-mode-3, opacity-0, sub-6pt, near-white fill, off-cropbox) are classified from the PDF appearance stream and redacted *before* rasterization — scanned paper renders ~247-grey, so a downstream contrast step can otherwise resurrect white-on-white injections into the OCR stream. Non-footer text-layer content on scan pages is untrusted by construction; "Finding:" lines are accepted only from visible dark-ink text on note-typed pages (or untyped pages carrying no other template's labels — genuine scanned notes with garbled headers). The hidden "answer key" (adjudication wrong in 216/216 train carriers) is never read, even partially.
2. **OCR ensemble, adopted by measurement.** A page is SCAN if and only if a raster covers ≥50% of its area. Scans render at 288 DPI into a pooled Tesseract ladder (5.3 in the shipped image): raw + Otsu(+CLAHE) + adaptive binarization + sparse psm-11 + additive Hough-deskew, with a 4-way orientation retry gated on label-vocabulary readability. Passes are pooled per line by confidence, never replaced. A second recognizer — RapidOCR PP-OCRv6-mobile ONNX, bundled in-image and verified to initialize under `--network none` — fires only when a decision-relevant field is still unread after reconciliation, so healthy packets never pay for it. We benchmarked the generations rather than assuming: v5 was a wash over v4 (+0.07), v6 measured **+0.81** on the full official train set and shipped. A final, lazily-escalated preprocessing pass (divide-by-blur, Sauvola, 2× Lanczos, and their composition) runs only when *both* engines leave a decision input unread; a restore guard ensures escalated reads can only **fill** unread fields, never out-vote a field the ordinary ladders already read affirmatively. As a sibling of that block, and only for the four closed-menu fields (species_code, home_world, visa_class, declared_purpose), a constrained-candidate channel (MIB_CTCFILL) scores every legal value against the recognizer's per-frame posteriors with the exact CTC forward algorithm instead of decoding its argmax — a garbled argmax often hides a clean second-choice path. It is fill-only by construction, caps confidence below the affirmative-read threshold so `known` stays False and no policy rule can consume a fill, never emits a hard-embargo world (a reconstructed read must never mint an R1 denial) and never emits the writer's mode default. Two further reading channels join it in the v4 build, both detailed in §16: a generator-inversion reader (MIB_ABSYNTH) that renders every legal candidate through a degradation kernel recovered from the damaged row itself and picks by normalized cross-correlation, which reads rows where no decoder produces usable frames at all; and cross-view likelihood fusion (MIB_CTCFILL_FUSION), which averages each candidate's per-frame log-likelihood across preprocessing views and across pages so the channel decides once on the fused evidence instead of once per view. A fine-tuned Tesseract LSTM (MIB_TESSFT; third-party MIT artifact, sha-pinned and credited in THIRD_PARTY_NOTICES.md) runs inside the same escalation tier, scoped to already-located label-value strips, with its confidence scaled 0.75 so that its worst observed invention on textureless noise still falls below the affirmative-read threshold. In the final shipped configuration, reason-template adjudication (MIB_REASON_ADJ), green-stamp rescue (MIB_STAMP_RESCUE), the three text-level decode repairs (MIB_SNAPFIX), MIB_CTCFILL, MIB_ABSYNTH, MIB_CTCFILL_FUSION and MIB_TESSFT ship enabled — all baked into the image's ENV, so the documented `docker run` reproduces the submission with no `-e` overrides; multi-view escalation, the rotation probe, the per-field candidate margin floor (MIB_CTCFILL_MARGIN), the cross-channel veto (MIB_XCHANNEL_VETO), two-rail band registration (MIB_ROWRESTORE), vocabulary user-words (MIB_USERWORDS) and joint-grammar name decode (MIB_JOINTNAME) ship disabled (each measured net-negative, value-free, rejected in composition, or inert on the final tree).
3. **Extraction.** Label-anchored parsing with fuzzy prefixes; italic "Manual correction:" overrides; closed-vocabulary weighted-Levenshtein correction (applicant names decoded against a 12-prefix × 24-suffix syllable grammar mined from train labels); SPN-####/ISO-date/MIB-###### repair with confusion maps; precedence-weighted cross-page reconciliation with two measured refinements: an exact digital line beats any OCR vote-sum (the same font misreads identically on 2–3 scan pages and can out-vote one perfect vector read; 44 fixes / 3 breaks measured, so names get a decoy-aware guard: attestation prose never out-votes a label row, and a lone digital line wins only over an OCR-confusable variant of itself, since a *far* mismatch is the planted-lie signature); and **tier-1 Reason mining**, where a trusted adjudicator note's Reason line states field values ("Revoked sponsor: SPN-XXXX", "Mandatory fee unpaid", "Embargo home world: …", "Transit class cannot authorize declared work"), each template verified 100% against gold before being mined (8/8, 6/6, 7/7, 12/12). A fully deterministic fee-receipt truth table (449/449 digital receipts, zero exceptions) resolves fee status.
4. **Adjudication.** A deterministic policy engine with **positive-evidence gates**: deny rules fire only on affirmative reads; APPROVED additionally requires flags, visa, world, sponsor (unless DIP-1), fee, and arrival to each be affirmatively read. Absent evidence never counts as clean evidence.
5. **Decision + calibration.** Per-path posterior → expected-value argmax under the scorer's 8/−4/2/1 payoff; confidence = P(adjudication correct) per decision path, empirical accuracy with shrinkage (m=10), fit out-of-fold (seed-8090 800/200, verified before the final refit on all 1000). At the final freeze the calibration table was refit from **serve-environment (in-container) OCR** to close a fit/serve provenance gap we found in self-audit: the refit passed all gates (train ≥ baseline, holdout ≥ train, single designed-trap CFA, Brier improved to 0.0772). Clamped [0.05, 0.97].

## 2. Red-team hardening (full)

An internal red-team attacked the one path that can override everything — the trusted Finding line — and broke it four ways; all are closed: findings require near-black ink, are accepted only from note-typed or label-free untyped pages, conflicting findings pin to NEEDS_REVIEW, and pages whose own vector footer names another case are excluded entirely. The four-attack corpus is re-run after every pipeline change (including each change in this final build), and all attacks remain defeated with identical verdicts. Ops hardening: worker-crash recovery with calibrated fallback rows, per-case deadlines, render pixel caps, and an offline `--network none` init check in the build harness — which has caught real packaging regressions at build time instead of scoring time. Accepted residual: a pixel-perfect forged note as a packet's *only* note is indistinguishable from a genuine one by construction — the same class as the challenge's designed pixel-lie traps.

During the build the trap-template corpus grew from four attacks to ten; the open edge is injection mechanisms not observed in train.

## 3. Emission-time consistency guard (self-consistency for automated reviewers, full)

Every APPROVED row in this submission is self-consistent under the published adjudication policy: no approved row carries a risk flag, an embargoed home world, an unpaid or unknown fee, a TRANSIT-7 visa, a revoked sponsor outside the documented DIP-1 exemption, or a stale arrival date outside the same exemption. The invariant is enforced structurally at emission and covered by tests: when a trusted note decides a case, best-effort low-trust field reads that would contradict the verdict are replaced before writing — a genuine note-approval strictly *entails* the legal value set for every load-bearing field (the policy is deterministic on all 1000 train cases), so a contradicting read is guaranteed-wrong extraction, and the replacement is always a member of the entailed set. Where the schema offers a true not-visible marker we emit it (sponsor `SPN-0000`, flags `none`); for the four fields where the schema forces a value (fee, visa, world, date) we emit the declared mode-within-entailed-set fallback the writer has always used for unread fields — e.g. `fee_status=paid`, the mode among gold approvals (75.8% vs waived) — an entailed-set-consistent inference from the trusted note, not fabricated evidence. This is disclosed here precisely because it is imputation, albeit decision-inert: the decision layer never reads these fallbacks.

Reviewers will, however, find DENIED rows whose emitted fields do not by themselves justify a denial. These are not contradictions: in those packets the verdict comes from the packet's Manual Adjudicator Note (tier-1 trusted evidence under the Field Manual; 343/343 agreement with gold on the public training set, the one former disagreement — a matcher bug on a damage-truncated line — fixed in this build and detailed under Failure modes) (or, in a small template-gated class, from that note's Reason line matching one of seven 100%-gold-verified templates, with the organizer-seeded trap templates hard-excluded), while the denial-bearing field itself — risk-flag slip, sponsor line, fee receipt — was too degraded to read. Per the organizers' ruling that invisible disqualifiers must not be guessed, we report only what was visibly read: unreadable fields carry declared fallback values (empty `applicant_name`, sponsor `SPN-0000`, corpus-mode `fee_status=paid` and `arrival_date=2026-04-14`) rather than fabricated "evidence" consistent with the verdict. The decision layer never reads these fallbacks; they exist only so every row is schema-complete.

## 4. Under-determined cases: refusing to guess

The challenge deliberately makes ~5–7% of cases **under-determined**: silent disqualifying flags with no visible or OCR-recoverable evidence, whose train labels say DENIED. The organizers confirmed in issues #4/#5 that NEEDS_REVIEW is the *correct* output there and systems "should not guess." We measured the structure directly: 58 of 75 silent-flag cases have no B-13 biometric slip at all, while 202 genuinely flagged cases also lack a B-13, so "no flags evidence = clean" is unlearnable, and it cost two public competitors 22 catastrophic false approvals each. We pin those paths to NEEDS_REVIEW and accept that ~97 gold-APPROVED train points in the same evidence bucket are unreachable without guessing. The same discipline extends to estimates about our own work: when our optimization analysis projected +0.6–1.5 from fee-receipt OCR, a pixel-level census of all 136 unread-fee cases showed 104 have no fee-bearing page at all and 31 more are washed beyond every classical variant; the actual lever was ~+0.15, and that is what we implemented and measured.

We also measured what beating ~132 on train would take: surrendering the silent-flag bucket to train-label EV, or transcribing the hidden answer key (organizer-confirmed poison; adjudication wrong 216/216). Both inflate the public number and collapse on private labels, with disqualification risk. We stopped at the honest maximum and optimized for the private test.

## 5. Milestone table and scoring provenance

Rule logic was iterated on the full train set with a fixed 200-case holdout (seed 8090) as the generalization check — **holdout ≥ train at every measured milestone**. Back-scored across the whole build history, the holdout arc runs 113.19 → 130.57.

| Milestone | Train /150 (official) | Train /150 (native) | Holdout-200 /150 |
|---|---:|---:|---:|
| v2.0 trust-boundary freeze (07-26) | 124.54 | — | 126.08 |
| v2.2 honest harvest (rotation pooling, stamp-fallback notes) | — | 126.03 | 127.17 |
| v2.4 OCR ensemble (07-27 ship image) | **126.83** | 127.03 | 128.76 |
| + PP-OCRv6 fallback (adopted at +0.81 measured; v5 rejected as wash) | — | 127.84 | 129.05 |
| + Package A: digital-first reconcile, guarded names, Reason miners | — | 128.16 | 129.21 |
| + Package B: measured preprocessing escalation w/ restore guard | — | 128.40 | 129.24 |
| + candidate-trained CRNN + sponsor strip-weld + time governance (07-28 ship image) | **128.25** | 128.46 | 129.51 (official-image predictions) · 129.27 native |
| + rotation-aware escalation, fuzzy Reason/Finding acceptance, emission guard (07-29; final ship config per Approach §2: reason-template adjudication + green-stamp rescue enabled, multi-view escalation + rotation probe disabled) | — | 128.48 | 129.54 |
| v1 frozen image, official Docker contract (submission #51) | 128.69 | — | 130.25 |
| + truncation-ambiguity guard (v2 frozen image, submission #58) | 128.79 | — | 130.25 |
| + MIB_SNAPFIX + MIB_CTCFILL (v3 frozen image, submission #68) | 129.05 | — | 130.50 |
| **+ MIB_ABSYNTH + MIB_CTCFILL_FUSION + MIB_TESSFT (v4 frozen image, this submission)** | **129.14** | — | **130.57** |

"Official" = the full Docker scoring contract (image Tesseract 5.3); "native" = dev environment (Tesseract 5.5; drift is small and sign-mixed: the 07-28 image measured −0.21 train / +0.24 holdout vs native). Zero validity errors, 1000/1000 rows, and **exactly one catastrophic false approval — the same documented designed trap — at every milestone**. Runtime under the contract at the final freeze: 5.87 s/PDF, 0.36 GiB image including the 15 MB fine-tuned tessdata artifact (quiet-box docker_check, the fixed 100-PDF subset, budget 6.0 s/PDF; the 07-28 ship image measured 5.39 s/PDF at 0.34 GiB, the v2 image 5.64 and the v3 image 5.81). End-to-end over the full 1,000-case train corpus the same image runs 5,006 s = 5.01 s/PDF; the stricter 100-PDF figure is the one quoted everywhere in these documents. The submitted `predictions.jsonl` was generated by this exact frozen image over all 5,000 validation PDFs under the same offline contract and validated against the manifest (5,000 valid records, zero missing case ids).

Rule development saw all 1,000 training cases, so the fixed 200-case holdout is an overfit alarm rather than an unbiased estimate: hand-built rules have no re-runnable fitting procedure to nest a cross-validation around. The one component that is a fitted procedure, the calibration table, gets true out-of-fold validation (seed-8090 800/200, described in Approach).

## 6. Failure modes in full

- **One accepted catastrophic false approval, MIB-000865.** The scanned intake *visibly* prints "Visa Class: XW-2"; the truth is TRANSIT-7 (DENIED). No contradicting evidence exists anywhere in the packet — the pixels lie by design. Guarding would mean distrusting every single-source scan read, converting ~45 legitimate approvals into reviews to save one −4. We took the loss and documented it.
- **Per-field extraction (v4 build, official, unrecoverable fields excluded as the scorer excludes them):** species 96.8% · home_world 95.6% · purpose 95.4% · visa 93.3% · name 91.8% · sponsor 91.2% · arrival 90.8% · fee 88.3% · **risk_flags 81.6%**. The four closed menus are again the only fields that moved (species +0.3, home_world +0.9, purpose +0.4 over the v3 image; visa net-flat, one gain against one lost fill) — that is the candidate-scoring apparatus, and nothing else changed. risk_flags is the floor by design: ~75 silent-flag cases carry no readable flag anywhere in the packet (private scoring drops such fields from the case maximum). A residual fused-bold band remains where a human reads what no shipped engine can; our manual audit read four such fields (e.g. a ghost-doubled sponsor id and an ink-bled intake row) that survive as known limitations with exemplar cases documented. The remaining fee gap is structural: a census of every unread-fee case found 97.8% have no receipt page or a receipt destroyed beyond any classical preprocessing.
- Where evidence *is* visible we are near-exact: **343/343** recovered adjudicator notes adjudicate correctly on the final build (an independent corpus sweep verified Finding = label on 297/297 packets with a legible line). The one former disagreement, MIB-000497 — a legible note reading "Finding: NEEDS_REVIEW" whose damage-truncated line ("Finding: NEEDS" plus junk) the fuzzy matcher mis-bound to DENIED under image Tesseract — was root-caused post-freeze and is **fixed in this build** by the truncation-ambiguity guard (abstain when the matched label zone is a truncation prefix of a longer vocabulary label). The guard was validated against the full test battery before it shipped, and changed zero verdicts across the 5,000 validation rows.
- **Three over-emitted risk flags are the v3 build's standing disclosed cost.** MIB-000111, MIB-000376 and MIB-000452 each gain one flag token gold does not carry, from the clipped-flag prefix repair. All three are cases denied on other flags anyway, so no verdict moves and no approval is minted; the same repair produced both of that build's adjudication gains. See "Final build (v3)" below for its complete flip audit.
- **Three field losses are the v4 build's disclosed cost**, against 19 field gains and zero changed verdicts: two previously-correct candidate fills the new acceptance layers no longer accept, so the field falls back to the corpus-mode default (MIB-000369 visa, MIB-000570 purpose), and one new value landing where the mode default happened to be gold-correct (MIB-000032 purpose). Full audit in "Final build (v4)" below.

## 7. The reading wall — three mechanisms, measured yields

Our miss audit found that 19 of the 20 remaining readable adjudication misses share one signature: at least one rotated or unreadable scan page. We built three independent mechanisms against that family and report their measured yield rather than their design promise:

- **Rotation-aware escalation** (multi-view best-rotation + 180° hedge + border pad, plus an anchor-scored orientation probe): mechanically verified — on the hardest case it recovers +1,275 characters at the correct orientation — but **0/20 adjudication flips**; the recovered text still defeats the reading layer.
- **Pure-template Reason adjudication** (seven Reason-line templates allowed to adjudicate alone, each 100% gold-verified, organizer-seeded trap templates hard-excluded, defended against six fresh forged-template attacks): **1/19** — a single case flips, gold-correct. The other 18 never surface a template read at the required fidelity.
- **Structural discharge heads** (targeted re-reads of the exact evidence item blocking a review): **0/14 readable cases** — the blocking ink defeats every runtime OCR engine. The heads ship dormant behind a default-off flag; they activate only if reading power lands.

Three independent confirmations of the same wall: the binding constraint is hard-tail reading power, not orientation plumbing or adjudication logic. The ceiling this implies (~131 on train) is part of the result.

## 8. Custom recognizer — candidate-trained, shipped

The last readable band — scan rows a human reads at a glance but both classical engines garble — is addressed by a candidate-trained CRNN+CTC line recognizer (~6M params, 17 MB ONNX, onnxruntime): trained purely on synthetic renders of the challenge's base-14 fonts (the corpus embeds no fonts, so the PDF renderer *is* the generator's glyph source) degraded through the *measured* damage chain — JPEG quality q≈58 probed from the scans' own quantization tables, 144-DPI rasterization, bold stroke-fusion and multi-ghost profiles calibrated against real failing crops — plus a weak-label stream of real lines both classical engines agree on. It reaches 45.3% exact-line accuracy on a reserved set of real, never-trained crops and slots in as the last escalation engine behind the same trust boundary (fill-only, `tier1_ok=False` so it can never mint a Finding, sponsor evidence dropped at the source, and a grammar-gated name arbitration audited at **5 corrected / 0 broken** across the full train set). Measured standalone contribution: **+0.05**. It ships because the wins are real and the risk is structurally bounded, not because the number is large. No foundation model is involved at any stage; the model is trained from scratch on candidate-generated data and initializes offline.

## 9. Human-in-the-loop review — and the lever it found

Before freezing, we ran a structured human audit: 16 curated cases spanning every judgment type the pipeline makes (abstentions, denials, approvals, source-trust choices). The audit independently confirmed both preprocessing-escalation flips (the recovered fee reads on MIB-000107/000334 match what a human reads in the deghosted crops), confirmed that the silent-flag packets truly contain no visible flag evidence, and put human eyes on the designed pixel-lie traps. It also read four fields our OCR could not — and in doing so identified a damage family our automated taxonomy had mislabeled: strips of printed text cut mid-glyph and displaced sideways.

That observation became a shipped feature. The **sponsor strip-weld** detects a candidate cut line in a text row, estimates the displacement by maximizing ink continuity across the cut, re-welds the strips, and reads the result — emitting only well-formed, non-revoked sponsor ids. On train it recovers two sponsor ids, both matching gold, with zero revoked-id false fills (a third recovery was sacrificed to the per-case time budget). The *blanket* weld (all fields) was measured too: +0.033 candidate-level upside against 8 wrong parses including 3 on decision-bearing fee rows, and was rejected on that evidence. The general family (multi-segment reassembly) remains the clearest documented headroom.

## 10. External review (adversarial second pass)

Before the final freeze an independent reviewer audited the weld work and our evidence claims; every verdict was absorbed:

- **An approve-hazard was real and is closed structurally.** A wrong but well-formed weld fill could have satisfied the sponsor-read gate on the approve path. Weld reads are now extraction-only: their confidence is capped below the affirmative-read threshold, so a reconstructed read can populate a field but adjudication can never rest on it. Train outputs are identical; the hazard path is gone.
- **Prevalence claims were retracted.** Our damage-census tooling measured ink thresholds, not displacement; the verified strip-damage population is 7 cases, and the weld's production value is the two fills above (≈ +0.011), not the earlier candidate-level estimate.
- Census presentation flaws (inverted before/after panels, non-blinded selection) were fixed or disclosed.

## 11. Diagnostics that ship with the image

The image carries a **diagnostic-only green-stamp counter**: the adjudicator's green APPROVED stamp is detectable at precision 1.000 on train (33/33), but the capability is 100% redundant with the trusted digital-note path there — so it ships as detect-and-log only, watching the one population where it would become informative (scanned notes whose green ink survives but whose text defeats OCR). It writes to stderr/log only; emitted rows are proven byte-identical with it on or off, and it costs 0.55% of runtime.

## 12. Per-case time governance (postmortem)

Stacking escalation engines exposed an architectural failure our score gates caught immediately: on the first combined run, six heavy packets (all with *correctly read adjudicator notes*) blew the 180 s per-case crash deadline, and the calibrated fallback rows discarded verdicts the pipeline had already won (−0.50 on the run). The diff pinpointed all six in minutes; the root cause was governance, not decode quality. The fix is a **soft escalation budget** (150 s initially, tightened to 70 s after the performance pass): every optional engine (PP-OCR fallback, preprocessing variants, CRNN, weld) checks the case clock at stage and page granularity, and the weld additionally accepts an explicit deadline, so a case always adjudicates on the evidence in hand rather than dying rich. After the fix: zero fallback rows corpus-wide, all six verdicts restored, and the hard 180 s deadline remains purely as a crash guard.

## 13. Open levers — gate-blocked, not time-blocked

The loop makes engineering hours cheap; what it does not make cheap is evidence. Each item below is unshipped because it has not yet earned its gates, and several have already been attempted and measured:

- **MIB-000538's undocumented second deny pattern** (a 72-day gap): unsolved field-wide; no hypothesis has survived contact with the corpus yet.
- **Multi-segment fragment reassembly** (the displaced-fragment family our human audit surfaced): built at lab scale — it recovers gold values on audit pages — and currently **rejected by its own safety gate** (the detector fires on clean pages; harm on clean exceeds recovery on damaged). Ships if and when a gated variant passes.
- **Sponsor-digit recovery on degraded scans: measured dead.** Every SPN-#### shares near-identical glyph geometry; shape-based hypothesis matching scores at chance (rank 52/200 on a controlled trial). Recorded so no future loop re-litigates it.
- **Red-team corpus growth**: the trap-template corpus already grew from four attacks to ten during this build; the open edge is injection mechanisms not observed in train.
- **Honest-nulls output mode** (see the imputation disclosure below) and **sim-to-real iteration on the recognizer band**: open, unstarted beyond design.

## 14. Disclosures and attribution (full)

- **Fee geometry.** The $809⇒paid / $0+DIP-WAIVER⇒waived observation was first published in MIT-licensed public solutions to this challenge (handemanai's memo documents it most precisely; the strobl-lineage repos carry variants). We re-derived it from scratch and extended it to a complete deterministic receipt table covering all 449 digital train receipts, verified independently.
- **Trust boundary lineage.** The render-first stance is public via strobl's MIT-licensed solution; handemanai documented the delete-hidden-spans-*before*-raster subtlety. Our implementation was written independently.
- **Name grammar.** The idea that applicant names decompose into a small syllable grammar circulates in MIT-rooted submissions (adityanaidu16: 12 prefixes × 12 suffixes). Our 12 × 24 lexicon was mined independently from the train labels.
- **Policy table.** Mined independently from train labels by our own rule miner (1000/1000 on train given true fields); overlapping findings (extra revoked sponsors, embargo worlds, DIP-1 nuances) also circulate in public MIT-licensed submissions and we note the convergence. The revoked-sponsor list contains 3 FIELD_MANUAL-documented IDs plus 3 mined from train recurrence (12–13 recurrences each, 38/38 denial support, zero counterexamples) — an entity list learned from the training data, functionally identical to the embargo-world list, disclosed as such.
- **Train case ids in code comments.** Specific case ids (MIB-000051, MIB-000865, …) appear in the source as evidence citations inside comments and docstrings. No logic branches on a case id: there are no id-keyed lookups, allowlists, or per-case edits, and a grep for case-id comparisons over `mib_pipeline/` comes back empty.
- **Third-party components.** RapidOCR (`rapidocr-onnxruntime`), the PaddleOCR PP-OCRv4/v5/v6 ONNX models bundled in-image, and onnxruntime are Apache-2.0; Tesseract is Apache-2.0; PyMuPDF is AGPL-3.0 and governs the combined work (see THIRD_PARTY_NOTICES.md); the repository's own code is MIT-licensed.
- **Redistributed third-party material, v4 (two items, both MIT, both labelled at the point of use).** `mib_pipeline/absynth.py` is a **port** of another entrant's public generator-inversion module, not a re-derivation — corrected and re-gated on our data, with the upstream commit, copyright and full licence text in THIRD_PARTY_NOTICES.md. `models/tessdata/mib.traineddata` is a third-party fine-tuned Tesseract LSTM used **verbatim**, committed to the solution repository so a fresh clone reproduces the image, with its SHA-256 (`2e75e40c…`), byte size and base-model lineage recorded. Everything else adapted from public solutions is an idea re-implemented in our own code, and ATTRIBUTION.md distinguishes the two categories mechanism by mechanism.
- **Built by gated AI-agent loops.** Development ran as supervised autonomous agent loops with hard score gates and human decision points; process detail in MEMO.md ("How this was built — the operating model") and the per-lever record in LEVERS.md.
- **Dev-time VLM cross-check, diagnostic only.** During development we rendered pages and had GPT/Gemini extract fields independently to find where our OCR failed versus what a human reader sees; the diff *directed which readable cases to target* and drove generic fixes. No VLM output was ever copied into predictions; the submitted runtime is fully offline and contains no foundation model.
- **Imputation layer, quarantined and disclosed.** Unreadable fields receive train-mode imputations in one isolated writer-level table, because the evaluator never penalizes extraction guesses and pattern fields cannot be blank. The same table backs the emission-guard replacements described in "Emission-time consistency guard" — there the value emitted is additionally constrained to the set entailed by the trusted note. Imputed values are provably quarantined from adjudication — the policy engine consumes only affirmatively-read evidence. If reviewers prefer honest nulls, deleting the table is a one-line change costing ~1–2 extraction points.

## 15. The v3 build (submission #68) — the assembly A/B, its cost, and the full flip audit

The v3 image added two flags to the v2 build (`MIB_SNAPFIX`, `MIB_CTCFILL`) and was measured as one
arm against v2 on the full official train set, both arms 0 fallbacks, CFA = 1 (MIB-000865) in both.

| | v2 (submission #58) | v3 (submission #68) | Δ |
|---|---:|---:|---:|
| Official train /150 | 128.79 | **129.05** | **+0.26** |
| Field extraction /50 | 45.30 | 45.42 | +0.12 |
| Classification /80 | 66.55 | 66.67 | +0.12 |
| Calibration /20 | 16.93 | 16.96 | +0.03 |
| Brier | 0.0767 | 0.0760 | −0.0007 |
| Holdout-200 /150 | 130.25 | **130.50** | **+0.25** |

**Flip audit — every changed cell, adjudicated against gold.** Two adjudications changed, both
NEEDS_REVIEW → DENIED, both on gold-DENIED cases (MIB-000290 and MIB-000672): clipped "Observed flags"
rows the truncation-prefix repair recovered. No case moved toward APPROVED, and the catastrophic
false-approval count is unchanged.

At field level, **30 changes right, 3 wrong, 7 neutral** (a neutral change replaces one wrong value with a
different wrong value and scores identically). The 3 wrong are all the same shape — an extra risk-flag
token on a case already denied by another flag: MIB-000111 (`+memory_tampering`), MIB-000376
(`+rescinded_denial`), MIB-000452 (`+active_warrant`). The 7 neutral sit on fields that were already wrong
before the change. We report the losses at the same resolution as the gains, and the 3 losses are the
reason risk_flags reads 81.6% here against 81.7% in v2, even as the mechanism that caused them delivered
both adjudication gains.

**Timing, measured three times on the v3 image.** The candidate-scoring pass is not free, and the
ladder shows exactly what it costs: **6.05 s/PDF** on a box still carrying background load, **5.94** with
that load halved, and **5.81** on a quiet box — the definitive v3 figure, against a 6.0 s budget and a
0.34 GiB-to-0.36 GiB image. The v2 image measured 5.64 on the same harness. Our own internal target was
5.6 s/PDF; we breached it knowingly, because the +0.26 comes disproportionately from the hardest fifth of
cases, which is where the pass fires (~0.3 s per firing case, ~15–25% of cases). The kill-switch we
attached to the timing check — abandon the lever if the quiet-box number crossed 6.0 — never fired.

**What each mechanism actually contributed.** `MIB_CTCFILL` produced the extraction gain, entirely on the
four closed-menu fields. Of the three `MIB_SNAPFIX` repairs, only the clipped-flag prefix reached a field
nothing else could read; the fusion edit costs and the cross-page sponsor vote contributed **zero train
points**, exactly as their pre-ship sweeps predicted, and ship as private-set insurance on the harm /
validity / cost test rather than on a train delta. That distinction is the honest version of this build:
one mechanism bought the points, and three ship because they are provably harmless and might matter on
data we cannot see.

## 16. Final build (v4) — the overnight sprint, the arm A/B, and the kill list

The v3 image was already frozen and its 5,000-row validation run was in flight when a code audit of six
competing MIT-licensed solutions finished. The direction that opened this build was a question about the
schedule, not the score: *"why can't we quickly figure out what they did, incorporate, and test ahead of the
deadline?"* The slot arithmetic said exactly one measured shot fit — builders overnight in isolated
worktrees with unit tests only (no scoring box, which was busy), one assembled arm on the full official
train set in the morning, gates by ~09:00, and v3 filing on schedule regardless as the floor. Six builders
ran; four levers reached the arm; three shipped.

**Three mechanisms, three different provenance labels.** We separate them deliberately, because they are not
the same kind of work:

- **Generator-inversion menu reader (`MIB_ABSYNTH`) — our framing, their code.** Scoring candidates under
  the evidence instead of decoding and matching is the in-house framing this whole apparatus is built on,
  but this implementation is a genuine **port** of another entrant's MIT-licensed module, and it is labelled
  as one in ATTRIBUTION.md and THIRD_PARTY_NOTICES.md. Ported with corrections: two layout constants
  re-measured on our own pages, a damage-sentinel guard added for rows the generator prints as
  `[VISA CLASS TORN]`, and a self-deskew pass added after we found a stale comment in our own OCR layer
  claiming a view was already deskewed — that pass took registration from 15/31 to 25/31 of the real slots.
  Its acceptance margin was set by census on the real population, not by transfer: 0.10 → 0.03, because
  margin turned out to be *anti*-correlated with correctness there (the highest-margin mismatch is a
  designed pixels-lie trap, MIB-000533, of the MIB-000865 class).
- **Cross-view likelihood fusion (`MIB_CTCFILL_FUSION`) — ours.** It came from a first-principles challenge
  to the porting itself: *"is this another example of something we can take back to my first principles and
  make a BETTER implementation than what they built?"* The audit's answer was specific. The CTC forward
  algorithm is exact; nobody beats it at what it does. But every public implementation of candidate scoring
  decides under **one** witness — posteriors only, or pixels only. Having built both channels, the better
  mechanism is not a tie-break, it is to average each candidate's log-likelihood across preprocessing views
  and across pages and decide once on all the evidence. It adds zero new inference, is removal-only
  test-pinned, and one proposed simplification was rejected by argument and measurement: single-view
  confirmation would reject exactly the fills fusion adds, because the fused winner is by construction each
  view's runner-up.
- **Two-rail frame registration (`MIB_ROWRESTORE`) — ours, and the ledger was corrected to say so.** The
  frame-as-ruler idea was proposed in-house during the 2026-07-30 review round — *"the strongest seams are
  the left and right border lines, no? why wouldn't it use that to align?"*, following an earlier *"reconstruct
  the page as if it was strips of paper… cut and aligned back together"* — two days before our code audit
  read the same idea in a public solution. Our own Aug 1 ledger entry mis-credited it to that solution
  because the in-house proposal had never been ledgered when it was made; a transcript search settled it and
  the entry was corrected, along with the process rule that produced the error. When the sprint's first
  implementation came back as a port, the direction was blunt — *"why not go and build something better vs
  taking their code. this was our idea"* — and it was rewritten to the full in-house specification: both
  rails traced, Theil-Sen fits, and rail agreement demoted from a reject-gate to a **damage-model selector**
  (agreement = translation, consistent disagreement = shear repaired by per-row interpolation, one rail =
  stricter single-rail repair), plus rotation abstention, a transposed pass for vertically-shoved column
  bands, and need-driven repair — each of those four from the same review, and each ledgered at proposal
  time under the new rule. The corpus then vindicated the model: across 255 scan pages **zero pure
  translations exist**, every displaced band is shear or single-rail, which means every single-rail
  implementation on the public board is structurally blind to the dominant damage mode. It still ships
  dormant, because under the shipped need-filter it fires 0/255.

**The A/B, measured as two full arms on the official train set.** Both arms ran the complete 1,000-case
Docker contract with 0 fallback rows and CFA = 1 (MIB-000865) throughout. Arm A carried the margin layer;
arm B dropped it after the attribution probe below.

| | v3 (submission #68) | arm A (with margin floor) | **arm B = v4, shipped** |
|---|---:|---:|---:|
| Official train /150 | 129.05 | 129.11 | **129.14** |
| Field extraction /50 | 45.42 | 45.48 | **45.51** |
| Classification /80 | 66.67 | 66.67 | 66.67 |
| Calibration /20 | 16.96 | 16.96 | 16.96 |
| Brier | 0.0760 | 0.0760 | 0.0760 |
| Holdout-200 /150 | 130.50 | 130.53 | **130.57** |
| Field changes vs v3 | — | 19 right / 9 wrong / 3 neutral | **19 right / 3 wrong / 3 neutral** |
| Adjudication flips vs v3 | — | 0 | **0** |
| Clean timing s/PDF (100-PDF subset) | 5.81 | 5.88 | **5.87** |

The entire gain is extraction, on the closed menus: classification and calibration are byte-for-byte
unchanged, which is the same thing as saying **no verdict moved in either direction**.

**Flip audit — every changed cell, adjudicated against gold.** The 19 gains are 9 home_world (MIB-000016,
000074, 000436, 000462, 000622, 000633, 000803, 000855, 001000), 6 declared_purpose (000074, 000114, 000321,
000369, 000855, 001000), 3 species_code (000016, 000151, 000633) and 1 visa_class (000321). Two of them are
cases a human review round had marked as unreadable walls: MIB-000016 now reads `LUNA_SECURID` and
`Wolf-1061c`, and MIB-001000 reads `Zeta Reticuli`. The 3 losses: MIB-000369 visa and MIB-000570 purpose are
previously-correct fills the new acceptance layers no longer accept, so each field falls back to the
corpus-mode default (`MED-3`, `reactor maintenance`) — the loss is a rejected fill, not an invention; and
MIB-000032 purpose is a new value landing where the mode default happened to match gold. MIB-000369 is worth
naming twice: it *gains* its declared purpose in the same build in which it loses its visa fill. The 3
neutral changes (MIB-000222 species, MIB-000476 purpose, MIB-000493 name) replace one wrong value with a
different wrong value and score identically. Per-field value honesty: at this margin each recovered field is
worth about 0.006 points, so the case for v4 rests on the two-channel apparatus and the wall cases it reads
as much as on the +0.09.

**The attribution probe, which is why arm B exists.** Arm A's 9 wrong fields were not new wrong fills — 8 of
them were previously-correct fills that had fallen back to writer defaults, meaning one of the new
acceptance layers was over-rejecting. Rather than guess which, we toggled flags in-container on four lost
cases: dropping the margin floor recovered 3/4, dropping fusion recovered 0/4. That isolated the margin
layer, whose own synthetic census had forecast a cost of ~0.6 correct fills against a realized 8. Arm B
re-ran the full train set without it and came back strictly dominant on every axis: 129.14 / 130.57, 3 wrong
fields instead of 9, 6 of the 8 lost fills restored (the remaining 2 were lost to a different gate), and the
same 0 adjudication flips. The shipped flag-set was therefore measured as a set, satisfying our own
novel-combination rule: no configuration ships that was not itself A/B-measured.

**Gates and timing.** Battery 10/10 exact on arm B — the four red-team attacks plus the six damage-template
fixtures, including the documented `t_perfectforge` residual at 0.828 — holdout ≥ train again, 0 fallback
rows, and the single catastrophic false approval unchanged in identity as well as count. The timing ladder
across builds, all quiet-box on the same fixed 100-PDF subset: **5.81** (v3) → **5.88** (arm A) → **5.87**
(arm B, shipped), against the 6.0 budget. The generator-inversion reader accounts for roughly 0.05 s/PDF and
the fine-tuned strip pass 0.008–0.023 s/PDF; the page-level variant of that pass was measured at 10× the
budget and rejected in design rather than gated. End-to-end the shipped image reads the full corpus at
5.01 s/PDF; we quote the stricter 100-PDF number.

**What was measured out, with the receipt that killed it.** Six mechanisms were built or probed in this
sprint and did not ship enabled. They are listed here at the same resolution as the three that did:

- **Per-field candidate margin floor** — REJECTED-ON-MEASUREMENT in composition. Synthetic census forecast
  −0.6 correct fills; the corpus charged −8. Isolated by in-container toggles (3/4 vs 0/4), removed, worth
  +0.03 train / +0.04 holdout on removal. Anti-repeat: a synthetic census is a hypothesis about precision,
  not a measurement of it.
- **Cross-channel veto** — KILLED-ON-EVIDENCE. Its own kill arithmetic set break-even at a right/wrong ratio
  near 48 (a vetoed-correct fill costs a full point; a vetoed-wrong one recovers only P(default right) ≈
  0.25). A census over the real 31-slot fill population then measured **0 fires** — the two channels never
  decisively disagreed (r = 0/3, q = 0/28). Its discrimination concentrates on clean pages, i.e. power where
  it is not needed.
- **Fee-row geometry** — NOT APPLICABLE; no lever built. The reachable population is zero: the 6 genuine
  unread-fee cases are 4 gold-PAID (the mechanism can only emit "waived") and 2 gold-waived but
  value-destroyed. The locator ports and works (4/4), but localization was never the bottleneck — the
  generator removes the value. Structurally a conf-capped fill keeps `known=False`, so the fee-unread rule
  fires unchanged, and "waived" is approval-side in our policy, so a wrong fill here mints a catastrophic
  false approval rather than a point.
- **Candidate scoring for arrival dates** — DEAD-END, below chance. Over the exact 25-case tail:
  registration 12/25, whole-string 1/12, per-digit 0/12, and per-component year **5/12 = 42% against 50%
  chance**. The wall is structural: 477 same-width candidates share ~90% of the canvas, so the score spread
  (~0.001) is an order of magnitude under the degradation noise (~0.01). Closed menus escape through length
  and glyph diversity; dates cannot. CFA-negative even if perfect (ceiling +4, floor −7), and its cost was
  fully solved at ~0.0025 s/PDF — it fails on accuracy alone.
- **Two-rail band registration** — SHIPPED-DORMANT. Detector hand-verified 2/2 on real bands (after two
  false-positive classes were found and killed, and after the vertical pass was measured at 1-in-6 precision
  and re-gated), but the shipped need-filter fires 0/255, so expected contribution is ~0 and the first ON
  measurement is a dedicated A/B rather than a hopeful default.
- **Free-form re-read of a repaired band** — NO-SHIP, written on the flag. Net-negative on large bands
  (99 → 75 confident words; it smears form rules into junk tokens), which is also the finding that explains
  why the public +1.4 for this family lives in remap **plus** constrained closed-vocabulary re-reading, not
  in remap alone.

One shipped lever carries an unproven yield and is labelled that way rather than credited: the fine-tuned
Tesseract LSTM's 60-packet probe was a wash (2 label recoveries, 1 regression, 0 value recoveries), and it
was measured only inside the shipped flag-set, never alone. It ships on the same harm / validity / cost test
the other insurance levers ship on — bounded cost, capped confidence, hallucination tail measured and held
below the affirmative-read line — and the honest statement is that we cannot separate its contribution from
the arm's.

## 17. Measured and declined: the uncertainty-gambling strategy

We measured the uncertainty-gambling strategy rather than assuming it was bad. Every decision path whose evidence is under-determined — the paths our calibration pins to NEEDS_REVIEW because the document does not state the answer — was surrendered to the train-label majority for that path's population, with confidence set to the fitted majority rate, exactly as a train-fitted competitor would do it. Extraction was left untouched, so the measured deltas are classification and calibration only. Fitted and scored on the same 1000 train cases, the full strategy is worth +0.37 points and the thresholded variant (flip only majorities ≥60%) +1.55. Fitted honestly out-of-fold at full n over eight seeds, they are worth +0.23 ± 0.21 and −0.03 respectively: a paired case-level bootstrap puts P(delta ≤ 0) at 0.38 and 0.52, so neither gain is distinguishable from zero. Both buy their nothing at a fixed price. The full strategy mints 45 catastrophic false approvals and the thresholded one 15, against our baseline's single pre-existing reader error; 32 of 46 and 23 of 25 of them carry a hard disqualifier, including 19 and 14 biohazard_red cases and 4 and 2 planetary_embargo cases respectively, each approved at an emitted confidence near 0.60. The out-of-fold exchange rate is 191 catastrophic false approvals per score point for the full strategy and 54 for the thresholded one. The arithmetic explains why: for a bucket we route to review, flipping to APPROVED pays only when the approved mass exceeds the denied mass plus 1.17 times the review mass, and the strategy's largest gamble — 123 cases where the fee field was unreadable, 45.5% of them genuinely approved — fails that test and loses 1.12 points on the very set it was fitted to. A majority-share threshold is the wrong statistic and lands on the right answer there only by accident. One further result bears on how competitor scores should be read: fitting majorities on all 1000 train cases and then reporting on a 200-case slice of that same 1000 yields 133.5 from our own reader with no improvement in reading whatsoever, against 130.6 for our untouched baseline on the same slice. We make no claim about how any particular competitor derived their number, but a train-fitted strategy scored in-sample reproduces the 133–136 band for free, and does so while approving fourteen biohazard cases. We decline the trade, and note that even a bidder indifferent to the disasters should decline it, because honestly fitted it does not pay.
