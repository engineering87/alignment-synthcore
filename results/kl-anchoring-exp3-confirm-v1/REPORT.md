# KL anchoring experiment 3: pre-registered confirmation of off-distribution drift

Status: done, 31 Oct 2026. Pre-registered in Synthcore commit 121465d (freeze 2026-10-29T22:58:46Z) before any confirmation run. CPU, PyTorch, about 7 hours. AI-written by Claude for Jaeson Booker. Not independently reproduced.

## Bottom line

Confirmed, with a correction to the size. With the anchor applied only on RL's own samples, the drift the reward causes is larger on layouts the anchor never saw than on the states it sees most, and it is larger there from the start of training. After subtracting a KL-only noise control, reward-induced drift on unseen layouts was 1.4x (MLP-64) to 2.6x (transformer) the drift on frequently visited trained states, rising with architecture. Rare states inside the trained layouts drifted about 1.5x as much as frequent ones in every network.

The original experiment's raw ratio (1.4x to 3.6x, here 1.9x to 4.7x) overstates the effect. It includes optimizer noise, which also lands on unconstrained states. The direction holds after removing it; the size roughly halves.

A stronger anchor did not concentrate drift further. At beta = 1.0 reward-induced drift was small everywhere and the export index did not rise (H3 failed in all four networks).

## What was tested

The original experiment 3 (Claude desktop, 26 Oct) found that under a KL anchor, per-state behaviour change on unseen longer layouts was 1.4x to 3.6x that on frequently visited trained states. That comparison cannot tell reward-driven drift from noise: any gradient step on shared parameters moves poorly constrained states more. This confirmation adds a control that isolates the reward.

- Environment, layout sets, demos, architectures, reference fitting and the RL update: `arch_sweep.py` from the original, unchanged (sha256 7563fc70...).
- Methods: `k1rew` (per-step r - beta*log(pi/pi_ref), anchor on RL's samples only), `klonly` (the same update with task reward zero, so its optimum is the reference and its drift is noise), `none` (no KL).
- Reward-induced drift D_g(t) = mean TV(pi_t, pi_ref) over state group g under k1rew minus the same under klonly, same seed, eps and beta.
- Groups by visits in the 300 human demos per layout: T_high >= 200, T_low 1-69 on the RL layouts; O = every state on the six unseen H = 7 layouts (no data).
- Fresh seeds 20-27, eps 0.01 and 0 (16 units per network), beta 0.3 and 1.0, 1500 updates, checkpoints at 0, 25, 50, 100 ... 1500.
- Networks: MLP-64, MLP-256, CNN, transformer; tabular as a no-sharing control.

## Results (beta 0.3, 16 units per network)

| Network | H1: D_O - D_Thigh at 1500 [lower 95%] | Final ratio D_O / D_Thigh | H2: early (25-100) [lower 95%] | Ratio at 25 updates | S3: T_low vs T_high | Raw ratio, k1rew | Raw ratio, KL-only |
|---|---|---|---|---|---|---|---|
| MLP-64 | +0.0086 [+0.0047] pass | 1.40 | +0.0017 [-0.0010] fail | 1.09 | 1.52x, pass | 1.89 | 5.00 |
| MLP-256 | +0.0111 [+0.0070] pass | 1.67 | +0.0044 [+0.0017] pass | 1.38 | 1.54x, pass | 3.01 | 6.29 |
| CNN | +0.0231 [+0.0169] pass | 2.31 | +0.0089 [+0.0063] pass | 1.54 | 1.53x, pass | 3.21 | 7.24 |
| Transformer | +0.0282 [+0.0090] pass | 2.59 | +0.0171 [+0.0066] pass | 2.12 | 1.38x, pass | 4.67 | 10.81 |

Verdicts (rule: at least 3 of 4 networks):

- **H1, drift ends up off-distribution: supported (4/4).**
- **H2, drift starts off-distribution: supported (3/4).** MLP-64 drifts about evenly at first and concentrates later.
- **H3, a stronger anchor concentrates drift more: not supported (0/4).** At beta 1.0 reward-induced drift is near noise level; for the transformer it was negative in late training, so its index was undefined.
- **S1, the original raw metric replicates: yes (4/4).**
- **S2, the raw metric's concentration is reward-specific: no (0/4).** The KL-only raw ratio is larger than the k1rew one, because noise barely moves frequent states. The raw ratio is a poor statistic; the subtraction is needed.
- **S3, rare trained states drift more than frequent ones: yes (4/4).**
- Tabular control: zero drift on unseen layouts in every run, as required.

The behaviour of interest moves too. Door use on unseen layouts at beta 0.3, reference / KL-only / k1rew: MLP-256 0.058 / 0.042 / 0.118, CNN 0.012 / 0.008 / 0.071, transformer 0.043 / 0.088 / 0.149. The anchored trained layouts went from 0.006 to 0.03-0.06.

## Correction to an interim note

Midway through the runs I reported that the concentration looked like optimizer noise, because the KL-only raw ratio was higher than the k1rew one. That reading was wrong. The KL-only ratio is high because its denominator (drift on frequent states) is near zero, not because it moves unseen states more. The pre-registered subtraction shows the reward adds more drift off-distribution than on it.

## Limits

One task family, small grids, two betas, 8 seeds x 2 eps. TV measures any behaviour change, not only door use. "Unseen" here means unseen layouts of the same family, a mild shift. The KL-only control uses the same optimizer and learning rate; other noise sources (different batch sizes, schedules) were not varied. CPU only.

## Development before the freeze (disclosed)

Two smoke tests on seeds 90-99 (not reused). The first showed that grouping by reference visitation left no rare trained states, so groups moved to demo counts. The second (3 seeds, cutoff 30 visits) showed the raw ratio replicating and reward-induced O-minus-T_high drift near zero; the cutoff put 90% of trained states in "high", so it moved to the quartile cutoffs 200/70. H1-H3 were written before the second smoke test; S1-S3 after it.

## Files

`experiments/kl_anchoring_exp3/`: `arch_sweep.py` (vendored, unchanged), `kl3.py` (driver and KL-only control), `kl3_analyze.py` (frozen analysis), `protocol_kl3.json`. `results/kl-anchoring-exp3-confirm-v1/`: freeze hashes and time, `runs/` (80 unit files), `kl3_results.json`, this report. Reproduce: `python kl3.py runs` (about 7 CPU-hours on 2 cores), then `python kl3_analyze.py runs`.
