# Attribution

Ideas adapted from MIT-licensed public MIB Doc Challenge solutions. All
code here is our own implementation; thresholds and guards were re-derived
and verified against our corpus evidence. The one exception is
`mib_pipeline/absynth.py`, which is a genuine PORT rather than a
re-implementation — its upstream MIT notice is reproduced in
`THIRD_PARTY_NOTICES.md`.

- **Fuzzy whole-phrase Finding recovery with runner-up margin**
  (`mib_pipeline/fields.py: _fuzzy_phrase_finding`): idea from
  tylergibbs1 (`tylergibbs1/mib-doc-challenge`, commits `a4a4310`,
  `7b647df`; MIT) and kirtandesai's solution memo (MIT). Our variant
  matches the whole phrase "FINDING: <LABEL>" at partial-ratio >= 78 with
  a >= 3-point runner-up margin, implemented on stdlib difflib.
- **Reason-template line identification**
  (`mib_pipeline/fields.py: _reason_template_line`,
  `mib_pipeline/vocab.py: REASON_TEMPLATES`): idea from kirtandesai (MIT;
  reported 22/22 Reason-line decisions on train). Our variant is
  deliberately narrower: the matched template only identifies a Reason
  line inside the existing stamp+Reason two-signal path and never supplies
  the Finding label, because 2 of the 18 concrete templates map to
  multiple labels in the corpus (organizer-seeded traps; see
  research/12-package-b-spec.md).
- **Fusion (2-gram <-> 1-gram) OCR edit costs with stricter re-accept**
  (`mib_pipeline/vocab.py: fusion_distance`, `_fusion_rematch`, gated
  behind `MIB_SNAPFIX=1`): multi-character confusion costs (rn<->m,
  cl<->d, ii<->n, vv<->w, li<->h, nn<->m, ri<->n at 0.4) from the
  mib-intake public solution (`mib/lexicon.py` `_MULTI_CONFUSIONS`; MIT);
  the discipline that a fusion-bridged match must clear a TIGHTER
  acceptance threshold than a raw match, plus the bridged-only single
  confusions c<->o / e<->a, from balawal's public solution
  (`mib/fuzzy.py` `snap_ligatures` / `snap_name`; MIT). Our variant is a
  full-matrix DP layered under the existing `match_vocab` as a
  second-chance pass that fires only on distance rejections.
- **Flag truncation-prefix acceptance licensed by label context**
  (`mib_pipeline/vocab.py: _truncated_flag`, gated behind
  `MIB_SNAPFIX=1`): rule from the mib-intake public solution
  (`mib/lexicon.py` `snap_flag`; MIT) — a >= 6-char exact unique prefix,
  or a >= 3-char fuzzy prefix at weighted-distance <= 1.0 with near-tie
  rejection, accepted ONLY for tokens from the value side of an
  Observed-flags label; free text keeps the strict rule.
- **Cross-page per-digit majority vote for sponsor_id**
  (`mib_pipeline/fields.py: sponsor_digit_vote`, gated behind
  `MIB_SNAPFIX=1`): idea from the handemanai public solution
  (`mib/pipeline.py` `_sponsor_digit_vote`; MIT) — per-position majority
  over a Hamming-<=2 cluster of SPN reads with the revoked-sponsor
  abstention guard. Our variant is fill-only (fires only while
  sponsor_id is unread), requires reads on >= 2 distinct pages, and
  abstains on any read outside a single cluster because no selected
  winner exists to anchor on. The tolerant SPN prefix (S/5/$, P/F,
  N/M/H/R/U/W) follows balawal's `snap_sponsor` (MIT).
- **Constrained-candidate CTC scoring against the bundled recognizer**
  (`mib_pipeline/ctcfill.py`, gated behind `MIB_CTCFILL=1`): mechanism
  from the MIT-licensed "moonshots" public solution (`mib/ctcscore.py`):
  instead of decoding the recognizer's argmax string and snapping it to
  the vocabulary, score every legal closed-menu candidate directly
  against the rec model's per-frame posteriors with the exact CTC
  forward algorithm — a garbled argmax often hides a clean second-choice
  path. Our implementation targets the PP-OCRv6 rec ONNX we already
  bundle (charset rebuilt from rec_keys.txt — v6 ships no `character`
  metadata; head layout verified at load), keeps the model's dynamic
  input width instead of the reference's fixed 320-px pad, scores
  label-inclusive candidate strings so the shared label prefix
  double-checks the locator, and anchors value strips on the OCR
  ladder's own word-box geometry instead of their template matching.
  Gates (log-prob floor, runner-up margin, null-hypothesis margin,
  locator confidence) were re-derived on our dev corpus starting from
  their flagread calibration (floor -3.5 / margin 2.0 nats).
- **Margin-as-confidence and dual-resolution consensus on the CTC fill**
  (`mib_pipeline/ctcfill.py`, gated behind `MIB_CTCFILL_MARGIN=1` on top
  of `MIB_CTCFILL=1`): idea from the MIT-licensed ShreyShingala public
  solution (`ocr-document-pipeline-challenge`,
  https://github.com/ShreyShingala/ocr-document-pipeline-challenge,
  `mib/row_restore.py`): `choose_equal_length` returns the
  best-minus-runner-up score margin alongside the winner so the margin
  itself is the confidence signal, and `consensus` accepts a
  reconstructed value only when two independently rendered views agree
  on it and both clear per-field floors (their defaults: date margin
  0.25, sponsor margin 0.40). Our variant re-derives the floors on our
  own length-normalized nats-per-character scale rather than importing
  theirs — they are per field and equalize TOTAL evidence across menus
  that differ 1.7x in candidate length — and takes the two views as two
  rec-model input resizes of one 288-DPI crop instead of two page
  renders, because re-rendering the page costs orders of magnitude more
  than a second rec pass. Layered purely restrictively on the shipped
  gate, so it can only remove fills, never add them.
- **Joint-grammar name decode**
  (`mib_pipeline/vocab.py: correct_name_joint`, gated behind
  `MIB_JOINTNAME=1`): idea from the MIT-licensed "moonshots" public
  solution (`mib/pipeline.py: _snap_name`): decode a garbled two-token
  applicant name over the JOINT first x last name grammar instead of
  snapping each token independently, so a clean token carries its badly
  garbled partner (their measurement: 54% vs 30% recovery on 1,010 real
  garbled reads, 0/2,121 clean reads broken). Our variant replaces their
  rapidfuzz joint scan with per-position `weighted_distance` shortlists
  over the attested 144-token lexicon (the 12x24 cartesian list doubles
  every real token with a phantom edit-distance-1 neighbour; the true
  grammar was re-derived structurally and verified against all 1,000
  train gold names) with the `partial_ratio_bound` prefilter in front of
  every full distance computation, and only fires after the shipped
  matcher failed to produce a grammar-legal name.
- **Two-rail frame registration** (`mib_pipeline/row_restore.py`, routed
  into `mib_pipeline/ctcfill.py: fill_restored`, gated behind
  `MIB_ROWRESTORE=1`): OURS, idea and implementation. Recorded here as a
  courtesy note rather than an attribution, because the boundary matters.
  The strip-realignment concept is Arthur's (2026-07-28, LEDGER.md:
  "reconstruct the page as if it was strips of paper ... cut and aligned
  back together"), built and lab-validated as lever W1 and shipped
  narrowly as the sponsor cut-strip weld. The frame-as-ruler gate that
  unblocked it is Arthur's too (2026-07-30: "the strongest seams are the
  left and right border lines, no? why wouldn't it use that to align?"),
  proposed in-house before we read any public code. The approach was
  independently confirmed in muhammadbalawal's public MIT solution on
  2026-08-01, and ShreyShingala's public MIT solution measures the
  approach paying within budget in its own pipeline; that external
  evidence is what moved the lever off our deferred queue, and both are
  credited for it. No third-party expression is used: the module was
  written to our own two-rail specification (both rules traced, Theil-Sen
  baselines, a rail-agreement MODEL SELECTOR over
  translation/shear/single-rail, need-driven repair scoping, and
  post-repair straightening validation), and a line-level diff against
  ShreyShingala's `mib/row_restore.py` shares nothing but `import cv2`,
  `import numpy as np`, `return None`, `return out`, `continue` and
  `else:`. Accordingly it carries no entry in THIRD_PARTY_NOTICES.md.
  Two properties of ours have no counterpart in the public solutions: the
  shear model (a band whose two ends moved by different amounts, which a
  single-rail constant-shift design cannot express OR detect — and which
  is the ONLY mode present in our corpus, zero pure translations across
  255 probed train pages), and the transposed pass for vertically shoved
  column bands (`MIB_ROWRESTORE_VERTICAL`, default off pending
  verification).

- **Anchor-scored rotation probe for baked-in rotations**
  (`mib_pipeline/pipeline.py: _probe_rotation` and friends, gated behind
  `MIB_ROT_PROBE=1`): idea from tylergibbs1
  (`tylergibbs1/mib-doc-challenge`, commit `028ba78`, who credits naidx0;
  MIT) and kirtandesai's solution memo (MIT). On pages whose
  footer-stripped primary OCR is near-empty (<100 chars and <2 form
  anchors), probe np.rot90 turns in order (0, 1, 3, 2) at half resolution
  and score each candidate by distinct form anchors + capped content
  length + mean OCR confidence, with the always-upright vector footer
  regex-stripped before scoring; early-exit at >=2 anchors and >=80
  chars. kirtandesai's variant (rank by recognizable document words
  instead of characters) was benchmarked against character ranking on our
  rotation-family fixture pages and won (10/11 correct selections vs
  5/11 under human-verified page orientations), so the shipped scorer
  ranks by document words drawn from the closed field vocabularies.
- **Generator-inversion menu reader (analysis by synthesis)**
  (`mib_pipeline/absynth.py`, gated behind `MIB_ABSYNTH=1`): ported from
  luke-harriman's public solution
  (`luke-harriman/mib-doc-challenge-solution`, commit `820b0cd`; MIT,
  Copyright (c) 2026 Luke Harriman — full notice in
  `THIRD_PARTY_NOTICES.md`). Two of their files are combined, because each
  fixes the other's weakness. From `lib/absynth2.py`: the insight that the
  generator's raster layout is known exactly (Helvetica, 13.2 px em, left
  margin x = 106, row pitch 31.15 px on the 1224x1584 page), so a row can
  be located on the grid rather than searched for; and the
  self-calibrating per-row degradation fit, which recovers the damage
  kernel from that row's OWN known label bitmap instead of from an offline
  calibration. From `lib/mfr_reader.py`: the two-anchor registration
  cross-check (title + a known-content "Case ID: <cid>" fiducial, required
  to agree geometrically before the registration is trusted), the
  ellipse-erode + Gaussian degradation model, the `flatten` background
  division, and the FIXED COMMON CANVAS for candidate NCC — without which
  scores are incomparable across candidates of different length, which is
  how a short value loses to a long one on a smeared row. Their gate
  (`margin >= 0.10`, measured at 100% precision on their train split) is
  taken unchanged, since it is computed on the same NCC statistic.
  Our variant: it is wired as a LAST RESORT behind `MIB_CTCFILL` on the
  four closed menus only, so it sees exactly the slots `writer.py` would
  otherwise impute with a mode default; it reuses the pipeline's own
  deskewed 288-DPI render at 0.5x instead of re-extracting and
  re-deskewing the embedded JPEG; the degradation grid is collapsed from
  their 56 anisotropic combinations to 24 isotropic ones (their
  anisotropic pairs only won on streak damage our `flatten` already
  removes); the search is cropped to the 520x720 block the layout can
  reach; and the per-field damage sentinels the generator actually prints
  ("[REGISTRY LOST]", "[SPECIES WHITEOUT]", "[VISA CLASS TORN]",
  "[PURPOSE ILLEGIBLE]" — counted in `research/scan_ocr.jsonl`) are scored
  as null candidates, generalising their single "[RISK PANEL MISSING]"
  entry, so an affirmatively destroyed row abstains instead of resolving
  to the least-bad menu value. Guards mirror `ctcfill.py` exactly:
  fill-only, confidence capped below the affirmative-read threshold,
  never a hard-embargo world, never the writer's mode default.
  The cross-channel veto built on this machinery
  (`mib_pipeline/absynth.py: veto`, gated behind `MIB_XCHANNEL_VETO=1`) is
  our own: no public solution scores one channel's candidate under
  another's evidence. It reuses the reader's registration and kernel
  recovery unchanged and only ever WITHDRAWS a ctcfill fill, never
  replaces one.
- **Fine-tuned Tesseract LSTM for the generator's font, as a targeted
  escalation pass** (`mib_pipeline/ocr.py: tessft_engine`, `tessft_strips`,
  `tessft_lines`; `models/tessdata/mib.traineddata`; gated behind
  `MIB_TESSFT=1`): the model artifact AND the idea of running it as a
  second recognizer both come from Shrey Shingala's public solution
  (`ShreyShingala/ocr-document-pipeline-challenge`, MIT, Copyright (c)
  2026 Shrey Shingala; `Dockerfile` lines 26-28 and the fine-tuning
  sections of `REPORT.md`). The `mib.traineddata` file is used verbatim,
  unmodified and NOT retrained: their own paired negative shows why —
  fine-tuning further on a guessed noise model made their results worse
  (580 vs 591 correct fields). Their measured facts: held-out CER 9.59%
  -> 0.19% on the generator's font, and, run as a full second engine over
  every scan, +1.26 s/PDF for +0.21 score and -1 catastrophic false
  approval in their image.
  Our integration deliberately differs on the cost axis, because a
  corpus-wide second engine is exactly what we cannot afford: our
  escalation tier is reached by >=35% of the corpus and one fine-tuned
  pass over a full 288-DPI render measures 0.371 s (against stock's
  0.185 s), i.e. ~0.30 s/PDF. Instead the pass runs only inside the
  already-lazy escalation tail, and there it reads only label-anchored
  VALUE STRIPS for the fields still unread, located from the word-box
  geometry the ladder already stashed (locator mechanism borrowed from
  our own `ctcfill.locate_strips`, itself credited above). Measured on 20
  packets of the real escalation population: 1.35 strips/case at 0.0167
  s/strip = 0.0226 s per escalated case. Reads are additive, pooled by
  confidence, and confidence-scaled by 0.75 — not a calibration fix (on
  409 words both engines read identically the means were 92.68 stock vs
  91.96 fine-tuned) but a hallucination guard: on textureless noise the
  fine-tuned model invents words up to conf 73.2 where stock peaks at
  50.6, and 0.75 keeps that ceiling below the affirmative-read threshold
  of 0.55.
  Not a repeat of `MIB_USERWORDS` (ledgered NO-SHIP at -0.22, whose
  anti-repeat forbids revisiting "without a different integration
  design"): that lever biased a general recognizer's DECODER toward the
  closed vocabulary and its post-mortem named the mechanism as
  "vocab-biased decoding corrupts reads", where this one swaps the MODEL
  and imposes no vocabulary prior. A vocabulary prior can only pull a read
  toward a legal string, so its errors are precisely the ones the matchers
  cannot reject; a misreading recognizer produces strings they do reject.
  The shared risk — displacing a correct read on a clean row — is what the
  pooling, the restore guard, the 0.75 cap and
  `test_clean_row_read_is_not_displaced` exist for.
  Note also that the artifact is a fine-tune of `tessdata_best` eng (float
  `Lfx512`) while our stock engine is the int-quantized `Lfx192` build of
  the same lineage, so this work does not separate the value of the font
  fine-tuning from the value of the larger base model.
