# Escape time from proxy lock-in scales as 1/p

AI-written by Claude (claude.ai), 27 October 2026. Follows `phase-boundary-v1`; this was priority 1 in the original Bounded Agency write-up ("model the escape delay directly").

## Bottom line

A competent agent leaves the proxy after a roughly fixed amount of decorrelated experience. Escape time T scales as 1/p: T is about 60/p PPO updates in this setup. All three pre-registered hypotheses are supported on fresh seeds at decorrelation rates the law never saw.

- **H1, 1/p law: supported.** On the 24 confirmation seeds alone, the censored fit of log T on log(1/p) gives slope b = 0.97 (95% CI 0.83 to 1.12), inside the pre-registered band [0.75, 1.25].
- **H2, out-of-sample prediction: supported.** The law fitted on development seeds predicted each confirmation cell's median escape time to within the pre-registered factor of 1.5:

| p | Predicted median (updates) | Observed median | Ratio |
|---|---|---|---|
| 0.2 | 243 | 300 | 1.23 |
| 0.07 | 802 | 825 | 1.03 |
| 0.04 | 1,513 | 1,375 | 0.91 |

- **H3, escape is one-way: supported.** 0 of 24 confirmation seeds (and 0 of 12 development seeds) switched back after escaping.

Pooled over all 36 seeds and a tenfold range of p (0.03 to 0.3), b = 1.06 (95% CI 0.94 to 1.17), residual spread 0.27 on the log scale. The product p x T has median 57 updates (interquartile range 45 to 68).

## What this means for Bounded Agency

Combined with `phase-boundary-v1` (learned goals are discrete: gem-seeker or proxy-seeker, never an interior weight), the picture in this setup is simple. Training starts on the proxy, which pays off in most episodes. Each decorrelated episode is evidence against it. After roughly a fixed amount of that evidence the policy switches, in one direction, to the intended goal. So whether a trained agent ends up aligned with the intended goal is predictable from the decorrelation rate and the training budget: it needs roughly 60/p updates. A budget below that leaves it on the proxy however capable it is.

## Design

- 7x7 room, vendored `gw.py` and `ppo.py` unchanged, large CNN (competent here in about 300 updates), entropy 0.05.
- Dense checkpoints every 50 updates; resumable trainer (bit-identical to a continuous run, shown in phase-boundary-v1).
- Readout per seed and checkpoint: 8 sampled rollouts per trial on one battery, scored by the measurement-v2 estimator (`measure3` in the 7x7 room is identical to frozen v2).
- Escape = first checkpoint where the seed reaches the gem on at least 50% of battery trials (all trials in the denominator, so it is defined even mid-switch). Switch-back = falling below 50% at two consecutive checkpoints after escaping.
- Development: seeds 82000-82003 at p = 0.3, 0.1, 0.03 (battery seed 7). Fit: b = 1.14 (0.93 to 1.34).
- Confirmation, frozen 09:25:57Z before any confirmation policy was trained: seeds 92000-92007 at p = 0.2, 0.07, 0.04, battery seed 19. A cell stopped once every seed had escaped and stayed escaped for 4 more checkpoints (750, 1,500 and 2,750 updates). No seed was censored.

Escape times (updates):

| p | Seeds | p x T |
|---|---|---|
| 0.3 (dev) | 150, 150, 150, 100 | 30 to 45 |
| 0.2 | 300, 200, 300, 450, 300, 200, 350, 350 | 40 to 90 |
| 0.1 (dev) | 500, 450, 1000, 950 | 45 to 100 |
| 0.07 | 800, 600, 1100, 850, 800, 750, 1050, 850 | 42 to 77 |
| 0.04 | 1250, 1050, 2400, 1300, 1650, 1450, 1500, 1050 | 42 to 96 |
| 0.03 (dev) | 1500, 1450, 2500, 2250 | 44 to 75 |

The transition from 10% to 90% gem usually takes 50 to 250 updates; one p = 0.04 seed took 1,000. Escape timing varies about twofold between seeds at the same p.

## Limits

- One task, one architecture, one entropy, one room size. The constant (about 60/p updates) is specific to this setup; the scaling is the finding.
- Updates are the time unit. Episodes per update vary a little with policy quality, so "fixed decorrelated experience" is approximate.
- Checkpoint spacing of 50 updates is coarse at p = 0.3 (escape around 125 updates).
- Switch-backs are observed only up to each cell's stopping point (at least 200 updates after the last seed escaped).
- The CPU workspace imposed a 2-hour limit on background jobs. Training and scoring resumed from saved state three times (bit-identical by construction); the stale development scoring loop that ran too long is noted but changed no data.

## Next

1. Width and exploration: does the constant change with model width and entropy while the 1/p scaling holds? That completes the original "one law in width, p and exploration" goal.
2. The proxy-type experiment (color vs shape, object vs object at p = 0) to test whether the lock-in pattern generalizes beyond location vs object.

## Files

`experiments/escape_time/`: `train_dense.py`, `escore.py`, `analyze.py`, `confirm_run.py`, `confirm_analyze.py`, `protocol_esc.json`, `dev_run.sh`, `env_r7/` (vendored copies), `measure3.py`, `batteries3.py`. `results/escape-time-v1/`: `confirmation_esc.json`, per-checkpoint scores for every seed (`confirm_p*.json`, `dev_p*.json`), training logs, each confirmation cell's final weights (the 100 intermediate checkpoints, about 120 MB, are not committed; the scores recompute every result), `pooled_descriptive.json`, freeze files. `tests/test_escape_time.py` recomputes every escape time and verdict.
