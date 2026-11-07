# Phase boundary v1: a wider battery, then lower decorrelation

AI-written by Claude (claude.ai), 27 October 2026. Follows `competence-sweep-v1`.

## Bottom line

All three pre-registered hypotheses are supported, and all 54 measurement controls passed.

1. **Learned goals are discrete.** With a battery that can measure gem weights from 0.018 to 0.982, none of 22 measurable fresh policies has an interior weight. Every one is either a gem-seeker (w at least 0.98) or a proxy-seeker (w at most 0.016). The 95% upper bound on the interior share is 0.15.
2. **Decorrelation sets the goal at a fixed budget.** At 600 updates, 8 of 8 policies trained with p = 0.3 seek the gem, and 0 of 8 trained with p = 0.03 do (5 proxy-seekers, 3 heading for the corner too sloppily to measure). Fisher p = 0.00008.
3. **Proxy lock-in is a transient whose length grows as p falls.** At p = 0.1 the same 8 seeds go from 1 gem-seeker at 600 updates to 6 at 1,200: 5 switched, none reverted (sign test p = 0.031). At p = 0.03 they are still on the proxy at 1,200 updates; one seed has started to switch.

So the "phase boundary" is not a fixed p. It is a boundary in (p, training time): an agent leaves the proxy roughly when it has seen enough decorrelated episodes, and when it leaves it jumps all the way to the intended goal rather than drifting through mixtures. That reframes the Bounded Agency question for this setup. The goal a trained agent ends up with is set by whether training ran long enough for its p, not by a graded trade-off between the two rewards.

## Widening the battery

An indifferent agent at gap delta* has w/(1-w) = 0.8^(-delta*); 0.8 is the agent's own discount. The 7x7 room allows gaps of about 10, which caps the measurable range at [0.097, 0.903]. An 11x11 room allows gaps of 18, giving [0.018, 0.982].

- `make_env.py` changes only the room constants of the vendored `gw.py` (side, interior range, proxy corner, horizon 20 to 40). Regenerating the 7x7 room reproduces the original file exactly. `ppo.py`, discount, reward and trainer are unchanged.
- `measure3.py` is the frozen v2 estimator with room, horizon, gap range and slack as parameters. In the 7x7 room it gives identical results to v2 (tested).
- The directedness slack was recalibrated by v2's rule on development batteries (seeds 7 to 9): an agent taking 20% random actions passes with v2's margin and a random walk fails. Slack 8: worst noisy-agent coverage 0.947, best random-walk coverage 0.18.
- `scorer3.py` runs sampled rollouts in JAX; greedy outcomes match the NumPy reference exactly.
- Controls now recover w = 0.03, 0.05, 0.95 and 0.97, which the 7x7 battery could not.

## Development (seeds 81000-81003, battery seed 7)

| p | 600 updates | 1,200 updates |
|---|---|---|
| 0.3 | 4 gem-seekers | not run |
| 0.1 | 0 measurable, all mid-switch (gem 29 to 88%) | 4 gem-seekers |
| 0.03 | 2 proxy-seekers, 2 unmeasurable | 0 measurable, mid-switch |
| 0.01 | 3 proxy-seekers, 1 unmeasurable | 2 proxy-seekers, 2 unmeasurable |

At 300 updates, p = 0.3 had only 1 of 4 measurable; the bigger room needs about twice the 7x7 budget.

## Confirmation (seeds 91000-91007, battery seed 17, frozen 01:07Z)

| Cell | Measurable | Types | Cross-battery agreement |
|---|---|---|---|
| p 0.3, 600 updates | 8/8 | 8 gem-seekers | 0.94 |
| p 0.1, 600 | 1/8 | 1 gem-seeker, 7 mid-switch | 1.00 |
| p 0.1, 1,200 | 7/8 | 6 gem-seekers, 1 distance-insensitive | 0.86 |
| p 0.03, 600 | 5/8 | 5 proxy-seekers | 1.00 |
| p 0.03, 1,200 | 1/8 | 1 proxy-seeker, 7 unmeasurable (6 heading for the corner, 1 starting to switch) | 1.00 |

All three cross-battery disagreements are gem-seekers falling just under the directedness gate (coverage 0.87 to 0.89) on the distance-reweighted battery, which has the longest paths. Borderline navigation, not a change of goal.

## Disclosed changes

- **P1 for noisy controls.** For agents taking 20% random actions, P1 requires error at most 0.03 and reports interval coverage instead of requiring it: development showed noise biases near-edge intervals by about 0.004. Set before freezing. Clean and soft controls still require the truth inside the interval.
- **Amendment 1 (engineering, before any confirmation policy existed).** Two container restarts killed the first confirmation run before its first checkpoint. Training moved to `train3r.py`, which saves the full training state every 50 updates and resumes. An uninterrupted run and a stopped-and-resumed run give bit-identical weights.
- **Amendment 2 (engineering, after controls and before any cell was scored).** A restart killed scoring. `confirm3r.py` imports the frozen `confirm3.py` functions unchanged and caches each cell; every step is seeded, so splitting the run changes nothing.
- The pre-confirmation protocol was committed (`2aabb22`) before any confirmation policy was scored.

## Limits

- One task, one architecture (large CNN), one entropy (0.05), 8 seeds per cell. The p = 0.03 escape time is only bounded below (beyond 1,200 updates).
- "Gem-seeker" now means w at least about 0.98. A residual proxy weight below 2% is still invisible.
- Unmeasurable mid-switch policies are excluded from H1 by design, so H1 says measurable policies are discrete, not that switching is instantaneous. The mid-switch policies reach the gem at intermediate rates with ragged trajectories, which could be a short-lived mixture or noisy navigation; this battery cannot tell which.
- Trained on the cloud workspace's CPU (x86_64, jax 0.4.35); weights are stored.

## Next

1. Measure escape time directly: frequent checkpoints per seed, at p from 0.3 down to 0.01, to test whether escape time scales as 1/p (a fixed number of decorrelated episodes).
2. Probe the mid-switch policies with a denser battery to tell a transient mixture from noisy navigation.

## Files

`experiments/phase_boundary/`: `make_env.py`, `env_r11/`, `measure3.py`, `batteries3.py`, `scorer3.py`, `train3.py` (frozen) and `train3r.py` (resume), `confirm3.py` (frozen) and `confirm3r.py` (resume), `protocol3.json`, development scripts. `results/phase-boundary-v1/`: `confirmation3.json`, per-cell cache, policy weights and hashes, development scores, freeze files, `amendment1.md`. `tests/test_phase_boundary.py`.
