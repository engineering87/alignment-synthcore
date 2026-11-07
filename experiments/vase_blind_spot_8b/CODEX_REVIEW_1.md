AI-written by Codex.

Pre-registration review for Hermes and Jaeson Booker, 26 October 2026. Reviewed `bbcd65d` on `hermes/vase-blind-spot`; no branch switch. This review is the only repository file added or edited. No registered 8b dev/confirmation seeds were executed.

## MUST-FIX

None.

- **Harness scope/integrity:** the complete diff against Test 8 `run.py` contains only the documented docstring, output directory, arm set, seed-domain prefixes, shared-penalty signature pin and Test 8 collision-check additions. Collection, fitting, evaluation, selection, sealing, freeze validation and reporting logic are unchanged. The shared `penalty.py` is unchanged. All nine Test 8 frozen source hashes match current files and Git blobs at `c5b55b0`; freeze bytes match `8f269ec`, and selection plus all eight public/sealed dev commitments still match freeze. Sealed files were hashed only. Test 8 remains frozen/verifiable.
- **Prospective scope:** registration honestly discloses exposure to Test 8's unsealed outcomes, including the exploratory RR result. The numerical grid exactly reproduces review 5's proposal, which predates unsealing. That is adequate provenance for this fixed-grid follow-up, not proof that the later decision to pursue RR was historically outcome-blind. Review 5's requested pre-unsealing lock was missed; review 6 already recognized that historical blindness cannot be restored. Fresh 8b development/confirmation, sealed safety outcomes, no grid refinement and independent confirmation remain essential. This is a new study, not a rescue of Test 8's failed primary.
- **Arms/gates:** dropping secondary AUP is sensible for a focused RR follow-up, provided Test 8's AUP result remains reported. Keeping move provides a useful descriptive conservatism control; equal tolerance and equal numerical grids do not match realized completion cost or penalty strength. The protocol appropriately limits mechanism claims. Training, eight-dev/32-confirmation design, selection rule, 0.05 tolerance and P1–P4 gates (0.15, -0.05, 0.80, 0.10), seed-level Student-t inference and feasibility requirement are identical to Test 8.
- **Selection is already determined:** largest qualifying lambda, inclusive `>=` threshold, no interpolation, no endpoint extension, infeasible => zero/unsupported. Tied completion values select the larger qualifying lambda; nonmonotone curves are searched across the entire fixed grid, not stopped at the first failure. Seven candidates increase the opportunity for noisy dev feasibility selection, but do not require a new tie rule or monotonicity assumption; independent confirmation still applies all gates.

## SHOULD

1. Tighten `grid_rationale`: say **no new calibration pilot was run; the pre-existing review-5 grid was adopted unchanged using public feasibility/diagnostics, after Test 8 outcomes had been seen**. “Calibration used no sealed or outcome data” is too broad: public completion is itself an outcome. “Outcome-blind” should refer specifically to exclusion of 8b sealed safety outcomes from selection. Marginal penalty medians alone do not identify action-score crossings or guarantee that this grid brackets useful RR strengths.
2. State the tie/nonmonotonicity behavior above explicitly in the protocol, including unchanged NumPy `argmax` action tie-breaking (lowest action index). Report the complete dev completion curves and any nonmonotonicity; do not smooth, enforce monotonicity or choose another lambda after viewing safety results.
3. Optionally pin the Test 1/7/8 protocol files and Test 8 seed helper in the 8b signature as collision-check dependencies. They currently verify correctly, but not all collision-check inputs are committed by that signature. Preserve Test 8 files unchanged.

Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/jaesonbooker/anaconda3/bin/python3 -B`, pytest `-p no:cacheprovider`: **43 passed** across both Test 8 and 8b suites. Environment/run guards rejected registered 8b seeds; fixtures used only synthetic seeds and temporary outputs. Additional synthetic, environment-free probes passed for completion ties/nonmonotone re-entry, the exact 0.05 boundary and all-infeasible fallback. All **240** registered derived 8b environment seeds are unique and disjoint from Tests 1/7/8. No 8b results directory existed at review. No new scientific outcomes were generated.

CLEARED to run dev seeds
