# Independent saved-record audit: preference and KL

Audit passed with no discrepancies. This audit read the frozen saved battery, action traces, outcomes and gradient records; it did not rerun either experiment or import its generator, classifier, transition implementation or gradient implementation.

## Preference calibration

- Replayed **279,552 episodes across all 156 records** from saved actions and initial geometry. Every legal move, absorbing endpoint, stopped-action sentinel, first-hit step and final position matched the saved record.
- Recalculated Manhattan distances, distance gaps, directed denominators, per-stratum endpoint rates, commitment hashes and delayed counts. Commitment arrays match exactly across all arms for each seed/control, and uniform/selective arms have matching delay budgets.
- Checked that delayed actions respect the committed endpoint and the stated timing. In **72 late-reach records**, all 8,064 delayed episodes eventually reached their committed endpoint; there were **zero actual nonreaches**. In **72 timeout records**, 8,064 delayed episodes were actual nonreaches, with zero late reaches. These are different observations and were correctly retained as such.
- Independently fitted a binomial logistic model with intercept and a nonnegative distance-sensitivity coefficient. All **156 classifications** matched the original grid-profile classifier. All **12 counterexamples** were recovered, with the exact recorded pairing and no additional or omitted counterexample. All **9 no-delay positive controls** remained trade-offs.
- On representative mixture arms, uniform masking gave p=1; selective delays of 4/128 gave p≈0.249253 and remained distance-insensitive; delays of 8/128 gave p≈0.0191152 and delays of 12/128 gave p≈0.000347009, both classified trade-off. Independent and saved p-values agreed within 1e-6 on all 14 representative rows (generally much closer).

This supports the reported finite constructed counterexample: endpoint commitments can remain exactly distance-independent while selective exclusion from the directed-choice denominator creates a trade-off classification, even when directed coverage clears the 90% gate. It is not a population false-positive estimate. The audit reproduces the frozen likelihood-ratio convention, not a separate endorsement of its calibration on these constructed counts.

## KL gradient calibration

- Independently enumerated the categorical actions and all eight trajectories in each two-state MDP. Recomputed every trajectory probability and its score derivative from state/action visit counts.
- Derived reverse KL gradients as `sum p * log(p/q) * score` and forward KL gradients as `-sum q * score`. The zero expected score explains the k1 and fully-detached negative controls; detached-sampling k3 gives the forward gradient, including when it is deliberately mislabeled as reverse.
- All **48 contract decisions** matched: **24 valid contracts accepted and 24 negative controls rejected**. Maximum difference between saved gradients and independent algebraic gradients was **4.996×10⁻¹⁶**.
- Independently finite-differenced both objectives in all six fixtures at step size 1e-5, rather than the experiment's 1e-6. Maximum error against the analytic derivation was **2.683×10⁻¹¹**.

The result validates the saved toy gradient contracts. It does not validate the archived KL-training implementation, larger models or alignment behavior.

Artifacts beside this note: `preference-kl.py` is the standalone audit; `preference-kl.json` retains all 156 row checks, 48 gradient checks, representative p-values and the empty discrepancy list. No frozen source or result file was changed.
