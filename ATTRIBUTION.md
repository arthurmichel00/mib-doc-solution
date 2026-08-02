# Attribution

Ideas adapted from MIT-licensed public MIB Doc Challenge solutions. All
code here is our own implementation; thresholds and guards were re-derived
and verified against our corpus evidence.

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
