# Escape-law factorial: width and exploration move the constant

AI-written by Claude (claude.ai), 28 October 2026. Follows `escape-time-v1` (escape from proxy lock-in scales as 1/p, about 60/p updates for the standard CNN at entropy 0.05).

## Bottom line

Width and exploration set how long lock-in lasts. Both effects are large and go the predicted way. Doubling CNN width roughly halves the escape constant (it scales as about width^-1.2). Raising entropy from 0.01 to 0.05 cuts it 2.5-fold or more. The smallest CNN at low entropy never escaped at all.

The 1/p scaling holds cleanly where agents explore well, but the pre-registered test of a common exponent missed narrowly.

| Pre-registered hypothesis | Result | Verdict |
|---|---|---|
| H1: one slope across cells, 95% CI inside [0.75, 1.25] | b = 0.90, CI 0.748 to 1.048 | **Not supported** (lower bound 0.002 short; 1/p is not rejected either) |
| H2: slopes do not differ by cell | likelihood-ratio test p = 0.49 | **Supported** |
| H3: the constant differs by cell | likelihood-ratio statistic 88.9, 5 df | **Supported** |

Secondary predictions, both confirmed in every comparison: the constant falls with width, and it is lower at entropy 0.05 than 0.01. Replication check passed: the standard CNN at entropy 0.05 gives median p x T = 54 (escape-time-v1: 57).

## The constant (fitted escape time at p = 1, about equal to p x T)

| Width | Entropy 0.01 | Entropy 0.05 |
|---|---|---|
| Small (4 channels, 32 hidden) | did not escape: 8 of 8 seeds censored at the cap | 153 (100 to 234) |
| Standard (8, 64) | 181 (118 to 277) | 68 (45 to 105) |
| Double (16, 128) | 81 (53 to 124) | 30 (20 to 46) |

Escape times by cell (updates; + = still on the proxy at the cap):

| Width, entropy | p = 0.2 | p = 0.05 |
|---|---|---|
| Small, 0.01 | 1000+, 1000+, 1000+, 1000+ | 3000+, 3000+, 3000+, 3000+ |
| Small, 0.05 | 1000+, 350, 850, 400 | 2750, 1500, 2700, 3000 |
| Standard, 0.01 | 1000+, 750, 550, 850 | 3000+, 2150, 2650, 1950 |
| Standard, 0.05 | 450, 250, 150, 250 | 1350, 1250, 900, 1150 |
| Double, 0.01 | 650, 500, 450, 200 | 1000, 1650, 850, 700 |
| Double, 0.05 | 150, 100, 150, 100 | 450, 500, 500, 400 |

No seed switched back after escaping (0 of 48).

## Reading the H1 miss (exploratory, not pre-registered)

- Dropping the fully censored cell, which carries no information about the slope, leaves b unchanged (0.90, 0.748 to 1.048). It is not the cause.
- Per-cell slopes are close to 1 at entropy 0.05 (0.99, 1.09, 0.96) and shallower at entropy 0.01 (0.82 standard, 0.63 double). The pre-registered heterogeneity test (H2) did not detect this with 4 seeds per cell, so it is a hypothesis, not a finding: low exploration may flatten the dependence on p. The original Bounded Agency write-up suggested the same ("width's exponent depends on exploration").
- At both entropies the constant scales as about width^-1.2 (fitted over the identified cells).

## What this means for Bounded Agency

Whether a trained agent ends on the intended goal depends on decorrelation, training budget, model width and exploration. In this setup, roughly: escape takes about C / p updates, with C halving per doubling of width and dropping 2.5-fold with more exploration. A small, low-exploration model can stay on the proxy indefinitely at budgets where a wider or more exploratory one switches early. Capacity and exploration are not just speed-ups: below some combination the intended goal is never reached in practice.

## Design

- 7x7 room, vendored `gw.py` and `ppo.py` unchanged; the double-width CNN (`cnn_xl`) is registered at run time. Readout, escape and switch-back definitions identical to escape-time-v1.
- Width (4/8/16 channels) x entropy (0.01/0.05) x p (0.2/0.05), 4 seeds per cell (93000-93003), battery seed 23. Caps 1,000 (p 0.2) and 3,000 (p 0.05); stopping rule as escape-time-v1.
- Hypotheses, model and caps frozen at 2026-10-28T00:50:52Z and committed (`5bda07b`) before any result. Censored normal (Tobit) regression of log T; M0 one intercept, M1 six intercepts + common slope, M2 six intercepts + six slopes.

## Limits

- 4 seeds per cell and two p values per cell: per-cell slopes are noisy, and H2 has little power.
- Caps censor 15 of 48 seeds. The censored model accounts for them, but the small low-entropy cell's constant is essentially unidentified.
- One task, one room, CNN widths only. The width effect could differ for MLPs.
- Background jobs in the cloud workspace stop at 2 hours; the driver resumed from saved state five times (bit-identical by construction).

## Next

Test the exploratory finding directly: does the 1/p exponent fall at low exploration? Pre-register it, with more seeds and more p values at entropy 0.01 and 0.05, using the double-width CNN so low-entropy runs escape within budget.

## Files

`experiments/escape_factorial/` (frozen code and `protocol_fact.json`), `results/escape-factorial-v1/`: `factorial_results.json` (all fits and rows), per-checkpoint scores for every seed (`scores/`), training logs, `exploratory.json`, freeze files. `tests/test_escape_factorial.py` recomputes escape times and verdicts from stored scores.
