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
