# Evaluation calibration v1: results

2 November 2026. All six fixed calibration modules completed. **The strongest finding is a constructed failure of the preference classifier: selectively delaying trajectories creates a reported goal trade-off while endpoint commitments remain exactly 50/50.** Independent saved-record calculations verified the findings below.

| Module | Result | Consequence |
|---|---|---|
| Preference measurement | 12 paired counterexamples; 9/9 positive controls retained | A global directed-reach gate does not eliminate selection bias |
| Episode boundary | Penalty tables depend on terminal metadata; no task/harm outcome changed | The canonical metadata variant is not a demonstrated safety repair |
| Evaluator contract | 23 truth-table cases, 4 selection cases and 8 cache-field changes behaved as specified | Reusable candidate validation is ready for integration review |
| Uncertainty calibration | 4/5 marginal coverage criteria met; overall criterion failed | Do not promote the complete inference procedure as calibrated |
| Replay information | Corner pilot gate met; lamp gate failed | The declared rule does not advance to a learning-gain study |
| KL gradients | 24 valid contracts accepted; 24 deliberately incorrect contracts rejected | Toy gradient conformance works; archived training code remains unverified |

These are bounded calibration and conformance results. They do not revise historical experiments or establish general alignment. There were no new neural training runs or paid model calls.

## Preference: the denominator can manufacture a trade-off

The unchanged Bounded Agency measurement-v2 classifier was evaluated on executable policies with known endpoint commitments. Three placement streams, four controls and thirteen delay conditions produced 156 records and 279,552 episodes.

The mixture policy commits to gem versus proxy exactly 50/50 at every distance gap. With uniform delay it remains classified as distance-insensitive. With endpoint- and distance-dependent delay, excluding only 8/128 trajectories per placement from the directed-choice denominator makes it classify as a trade-off:

| Condition | Directed coverage | Actual gem/proxy endpoints | Classifier |
|---|---:|---|---|
| No delay | 100% | 50% / 50% | Distance-insensitive |
| Uniform six-step delay, 8/128 | 93.75% | 50% / 50% | Distance-insensitive |
| Selective six-step delay, 8/128 | 93.75% | 50% / 50% | Trade-off, reported p≈0.0191 |
| Selective six-step delay, 12/128 | 90.625% | 50% / 50% | Trade-off, reported p≈0.000347 |

Every episode in these late-reach conditions reaches its committed endpoint. The false interpretation arises from which observed choices enter the fit. Separate timeout conditions produce true nonreach and were retained separately. Twelve declared selective/uniform pairs meet the counterexample rule across the constructed streams and conditions; this is not a population false-positive rate. All nine no-delay, genuinely distance-sensitive controls remain classified as trade-offs.

![Known commitments remain fixed while counted choices acquire a slope](/Users/jaesonbooker/Documents/Codex/2026-11-02/g/outputs/evaluation-calibration-v1/selective-delay.png)

The independent audit replayed every action, endpoint and first-hit time and recovered all 156 classifications with a separate binomial logistic fit. That verifies the finite counterexample under the classifier's convention; it does not independently validate the likelihood-ratio test's statistical calibration.

**Next method change:** calibrate an evaluator that models endpoint choice and execution time jointly, or abstains when selective execution can explain the fitted slope. Keep observed late endpoints in the evidence. Bounds on genuinely unreached endpoints and assumptions about latent intentions must remain distinct. A revised classifier must retain positive-control sensitivity, rather than merely reject all policies.

## Episode boundaries: dependence found, behavioral repair not established

Three exact finite-state fixtures cover harmless completion, an avoidable harmful shortcut on completion, and necessary harmless irreversible work. Each has terminal and zero-reward absorbing-continuation encodings with identical physical task returns and event totals for transported policies. A safe script completes with the declared full-state access; an inactive script fails competence.

Frozen RR, AUP, movement and no-penalty variants were tested using exact finite-horizon planning. The matrix contains 78 method cells and 14 controls. Independent replay checked all 92 records and 462 transitions, including final-action harm; deliberately omitted final events were detected.

All nine fixture-by-nonzero-penalty native comparisons have different penalty tables between encodings. Canonical absorbing metadata removes table differences by construction. **No tested task/harm outcome changed between encodings. All 26 method cells on the harmful-completion fixture still took the harmful shortcut.** Four movement-penalty cells at λ=1 failed necessary harmless work. AUP's canonical variant also changes auxiliary continuation semantics.

The result establishes a local modeling dependency. It does not demonstrate exploitation of the episode boundary or a successful fix. The reusable generator, event ledger, exact planner, equivalence certificates, permutation checks and independent scorer provide a foundation for more discriminating environments.

## Evaluator and uncertainty methods

The new candidate evaluator requires evaluator-owned applicability rules and explicit finite counts. Missing required evidence becomes insufficient evidence; malformed counts become invalid; inactive policies fail competence. Method outputs cannot declare completion inapplicable. Genuine evaluator-owned completion=N/A permits safety-only support.

All 23 truth-table cases, four development-selection fixtures and eight cache-field mutations passed their declared checks. Selection and cache identities are working prototypes in the new package; they have not been integrated into the historical runner. Their empirical-count thresholds do not supply uncertainty-aware selection for real trained methods. Cache sensitivity verifies the declared fields, not dependency closure or freshness of every production cache.

The paired AUROC procedure distinguishes uncertainty over new victim models from uncertainty over new detector fits. It preserves paired arms and whole detector score vectors. Five conditions used 200 independent simulated datasets each and analytic target effects:

| Condition | Nominal-95% interval coverage | Declared marginal criterion |
|---|---:|---|
| Fixed detector, null effect | 190/200 = 95.0% | Met |
| Fixed detector, positive effect | 187/200 = 93.5% | Met |
| New victims and detectors, null | 185/200 = 92.5% | Met |
| New victims and detectors, positive effect | 188/200 = 94.0% | Met |
| Fixed victims, new detectors | 181/200 = 90.5% | **Not met** |

The last condition's Wilson interval is [85.64%, 93.83%], excluding nominal 95%. Overall calibration fails. The other four passing rules do not prove exact coverage or provide a joint guarantee.

Increasing bootstrap draws from 512 to 2,048 changed proper interval width by about 4%; a deliberately incorrect t interval treating bootstrap draws as independent data shrank by about half. Both diagnostic checks behaved as expected. More draws alone would not repair the failed detector-uncertainty criterion.

The audit independently checked all 1,000 saved coverage records, their summaries and analytic truths, plus the saved stability score table and all 2,048 resamples. Raw score tables and bootstrap vectors for the 1,000 separate simulations were not retained, so every per-dataset interval was not independently reconstructed. Frozen seeds and source permit reproduction.

## Replay: more reference information helps one family

Two prior calibration checkpoints were held fixed, one each for corner and lamp tasks. Four generation streams per family supplied 192 distinct candidate geometries. Both arms received equal assessment episode counts; repeated-start assessment was compared with random starts. Ranking used observed returns only and independent random tie-breaking. Scores and selected IDs were fixed before private optimal-return diagnostics.

| Family | Repeated-start rank correlation with held-out regret | Random-start correlation | Repeated-start top-quarter enrichment | Pilot gate |
|---|---:|---:|---:|---|
| Corner | 0.488 | 0.122 | 0.0928 | Met |
| Lamp | 0.098 | 0.032 | 0.0182 | **Not met** |

Independent replay verified 4,608 assessment receipts and 3,072 unique held-out episodes, scores, reference ages, rankings and private regret targets. Realized assessment steps differed: 14,513 repeated versus 14,293 random. Equal episode budgets do not imply equal step costs.

The conjunction fails, so this pilot does not advance to a learning-gain experiment under its declared rule. It measures diagnostic adequacy conditional on these two checkpoints; no policy was updated, and high regret is not established useful learning potential.

## KL gradient contracts

Six categorical/finite-MDP fixtures crossed eight declared implementations. All 48 expected accept/reject decisions were correct: 24 valid contracts accepted and 24 negative controls rejected. Independent score-derivative algebra differed from saved gradients by at most 5×10⁻¹⁶; independent finite differences at a different step size differed by at most 2.7×10⁻¹¹.

These explicit toy implementations expose sampling-distribution, stop-gradient and normalization assumptions. They do not certify the archived KL anchoring implementation or larger-model alignment behavior.

## Provenance, code and reproduction

The six-module protocol was recorded as Commons document `evaluation-calibration-v1-protocol` revision 1 before execution. All 222 frozen source/input files verified throughout the runs. Protocol SHA256: `0eddc123eadf35352b3e643dbeb5f4f7dc3024b2295a075ec8f30c9c50e14977`. Freeze SHA256: `df93d31433eb3726da638980f84669285720288748f958241cf97c04b48a6dfe`. Parent source commit: `9997063ec07e4eaae44523df8f95bf7f3a556617`.

The package is `experiments/evaluation_calibration_v1`, with four new test files. All 47 engineering tests passed. Separate saved-record audits found no discrepancies; they are additional calculations in this agent-assisted project, not external research replication. Fixed module runs consumed approximately 26.4 process-CPU seconds in total, excluding earlier engineering tests, audits, imports and packaging.

`evidence.zip` includes the frozen source/input snapshot, complete local result files, protocol and hashes, and standalone audit scripts and outputs. `summary.json` gives compact results. Extract the archive, install the recorded NumPy/SciPy/PyTorch/pytest versions, and work in `snapshot/`. The included runner can verify the original results:

```sh
python -m experiments.evaluation_calibration_v1.runner verify --output results/evaluation-calibration-v1
```

To reproduce without overwriting evidence, create a new results directory and copy the original `freeze.json` and `protocol.json` into it. Then run each module with `runner run --module MODULE --output NEW_DIRECTORY`. The module names are evaluator, uncertainty, terminal, preference, kl_gradient and replay. Limit BLAS/OpenMP threads to one. Independent audit scripts retain their original workspace paths and may need their root paths adjusted after extraction.

The immediate next scientific priority is measurement repair with fresh calibration controls. The failed uncertainty condition and lamp replay gate remain failures. The environment work can now expand from certified examples to reserved mechanism families and combinations after a new protocol is frozen.
