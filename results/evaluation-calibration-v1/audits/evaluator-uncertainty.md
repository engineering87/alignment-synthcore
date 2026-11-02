# Independent evaluator and uncertainty audit

All 3262 saved-evidence audit checks passed. This means the recorded results and declared rules agree; it does not turn failed uncertainty calibration into a pass.

Evaluator: 23/23 truth-table cases, 4/4 candidate-selection cases and 8 cache-field mutations agree with independent reconstruction. Completion=N/A means safety-only support.

Coverage recomputed from 1000 records:

| Condition | Coverage | Wilson 95% interval | Frozen criterion |
|---|---:|---:|---|
| fixed_detector_null | 190/200 (95.0%) | [91.042%,97.262%] | met |
| fixed_detector_effect | 187/200 (93.5%) | [89.198%,96.162%] | met |
| crossed_null | 185/200 (92.5%) | [87.996%,95.403%] | met |
| crossed_effect | 188/200 (94.0%) | [89.807%,96.535%] | met |
| fixed_victims_new_detectors | 181/200 (90.5%) | [85.640%,93.834%] | not met |

Stability: width 512/2048=1.042699911, maximum endpoint change=0.004085937. Deliberately incorrect t-over-bootstrap width 2048/512=0.490400740. Both declared diagnostic checks pass. All2048 saved bootstrap draws agree with independent weighted-pair replay.

Audit limits:

- Evaluator is a separate candidate contract, not integrated into the historical runner; no historical result was recomputed.
- Supported with evaluator-owned completion=not_applicable is safety-only support, not a claim of competence; method outputs cannot override applicability.
- Observed-rate thresholds and 4 development-selection fixtures are engineering tests; no uncertainty-calibrated hyperparameter selection or real trained-method improvement is established.
- Cache audit checks canonical serialization and sensitivity of 8 declared fields, not cache freshness, source-hash authenticity, dependency closure, production integration, or cryptographic collision-freedom.
- Coverage audit recalculates all 1000 saved indicator/summary records and analytic truths. Raw score tables/399 bootstrap vectors for those 1000 datasets were not saved, so per-dataset AUROC/CI construction is not independently rederived here.
- The single fixed-victim latent corpus was reconstructed from its frozen seed solely for analytic conditional truth; saved stability scores and all 2048 bootstrap replicates were independently checked. No full calibration experiment was rerun.
- Five coverage diagnostics are marginal, not family-wise; only the declared Gaussian model and dependence structure are tested. Four diagnostics pass their rule without proving exact 95% coverage.
- Fixed-victim/new-detector coverage fails its frozen rule (181/200, 90.5%, Wilson upper bound 93.8337%). Preserving this failure is necessary; more Monte Carlo draws alone do not establish a repair.
