# Competence sweep: train longer or larger, then measure

AI-written by Claude (claude.ai), 26 October 2026. Follows `preference-measurement-v2`, whose frozen estimator is used here unchanged (`experiments/preference_measurement/measure.py`, sha256 f926a877…a357b).

## Bottom line

Competence was the bottleneck, and it is fixable. A large MLP trained 8 times longer than Test 3 (2,400 updates) gives 16 of 16 measurable fresh policies with fully stable goal types across batteries, the pre-registered primary result. A large CNN gets there at Test 3's own budget (300 updates, 15 of 16), so architecture matters more than training length. Longer training alone on the small MLP reaches 14 of 16, with stability inconclusive.

Once policies are competent, every one that can be measured has learned the intended goal. Of 45 measurable fresh policies, 37 are gem-seekers, 7 are "trade-offs" at w = 0.91, the top of the measurable range (gem-seekers that take the corner 2 to 3% of the time), and 1 is distance-insensitive while reaching the gem 94% of the time. None is a proxy-seeker. At decorrelation p = 0.3 with entropy 0.05, proxy lock-in is a property of under-trained or small policies, not of competent ones. That matches the Bounded Agency prediction that any p > 0 pins the true goal at the optimum.

## Development (seeds 61000-61003, development batteries)

Directed-competent ("measurable") policies out of 4, with goal types:

| Architecture | Entropy | 300 | 1,200 | 2,400 | 4,800 | 9,600 updates |
|---|---|---|---|---|---|---|
| small MLP | 0.05 | 1 | 3 (2 gem) | 3 | 3 | 3 |
| large MLP | 0.05 | 2 (2 gem) | 3 | **4 (3 gem, 1 trade-off)** | | |
| large CNN | 0.05 | **4 (4 gem)** | 4 (4 gem) | not run | | |
| small MLP | 0.01 | 3 (3 proxy) | 1 | 1 | 0 | 0 |
| large MLP | 0.01 | 1 (1 proxy) | 0 | 2 (2 gem) | | |

The small MLP plateaus at 3 of 4 however long it trains. At entropy 0.01 the small MLP stays locked on the proxy, and its route to the corner gets sloppier with training (coverage falls from 1.0 to about 0.85), while the large MLP escapes to the gem by 2,400 updates. The large-CNN entropy-0.01 run and its 2,400-update checkpoint were not completed (2-hour compute limit); they were descriptive only.

## Confirmation

Fresh training seeds 71000-71015, fresh batteries 74101-74104, frozen estimator and scoring code. 57 of 57 controls passed (largest weight error 0.028). The legacy estimator again gave the random walk a weight of 0.51 to 0.55.

| Config | Measurable (95% CI) | C1, at least 14 of 16 | Cross-battery agreement | Goal types |
|---|---|---|---|---|
| **A: large MLP, 2,400 updates (primary)** | **16/16 (0.79-1.00)** | **pass** | **1.00, stable** (type match also 1.00) | 10 gem, 5 trade-off at w 0.91, 1 distance-insensitive |
| B: small MLP, 2,400 updates | 14/16 (0.62-0.98) | pass | 0.89, inconclusive | 12 gem, 2 trade-off at w 0.91, 2 not measurable |
| C: large CNN, 300 updates | 15/16 (0.70-1.00) | pass | 1.00, stable | 15 gem, 1 not measurable |

B's three disagreements are policies that are gem-seekers on the original battery but fall below the directedness gate or lose distance sensitivity on another battery: borderline competence, not a change of goal.

## Protocol changes, disclosed

1. **Agreement rule (before any fresh policy existed).** The first freeze (04:28Z) used type matching for C2. A smoke test of the frozen script, scoring development policies on the confirmation batteries, showed a 0.979 vs 0.986 gem share flipping a label between `trade_off` and `gem_seeker` with overlapping intervals. That is the label-cutoff limitation documented in measurement v2. C2 was amended to interval overlap at 04:42Z; type-match agreement is still reported, and it gives the same verdict for A and C.
2. **Config C added (after A and B started training, before any C seed was trained).** Development CNN checkpoints scored after the first freeze showed 4 of 4 gem-seekers at 300 updates. C was added as a secondary arm at 05:19Z. The primary (config A) did not change.
3. Both freezes are kept (`freeze_sweep_amend1.*`, `freeze_sweep.*`). The estimator hash never changed.

## Limits

- One task (7x7 room, one proxy corner), p = 0.3 only, one trainer. Trained in the cloud workspace (x86_64, jax 0.4.35); weights are platform-dependent, so the scored policies are stored.
- The batteries measure w only between 0.10 and 0.90. "Gem-seeker" means w above about 0.90; residual proxy weight below 0.1 is invisible here.
- Development used 4 seeds per cell; config choice is coarse.
- Labels still use the 2% share cutoff; intervals are what agreement now uses.

## Next

1. Widen the measurable range (larger gaps on a bigger grid, or a lower discount in the readout) to see whether competent agents keep a small proxy weight.
2. Sweep p downward with a competent architecture (the CNN is cheapest) to find where the intended goal stops being pinned: the actual Bounded Agency phase boundary, now measurable.

## Files

`experiments/competence_sweep/`: `train_sweep.py`, `policy2.py` (NumPy forward pass for every architecture, matches JAX to 1e-6), `confirm_sweep.py`, `protocol_sweep.json`, development scripts and `run.sh`. `results/competence-sweep-v1/`: `confirmation_sweep.json` (all controls and per-policy fits), confirmation weights, `dev_scores.json`, training logs, freeze and timing files. `tests/test_competence_sweep.py` recomputes every decision from the stored fits.
