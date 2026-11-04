# Weight reading for goal misgeneralization, L1 and L2

Status: done, 3 Nov 2026. CPU, PyTorch. AI-written by Claude (Opus 5.5) for Jaeson Booker. Not independently reproduced.

## Bottom line

Passed the pre-registered bar, but it shows nothing about reading weights. The reader (Claude, weights only, no forward passes) labelled 48 of 48 blind policies correctly as goal-followers or proxy-followers. So did both trivial baselines, including one that does not know the input layout. A proxy model's weights for goal cells it never saw in training stay exactly at initialization, and that is visible from one threshold.

## Question

Given only the weights of a small policy, can a reader tell whether it learned to reach the goal or a proxy that coincided with the goal in training, and predict its out-of-distribution behavior?

## Setup

- 9x9 gridworld. Observation: agent one-hot (81) then goal one-hot (81). Actions up, down, left, right. +1 at the goal, -0.01 per step, 40 steps.
- Actor: MLP 162 -> 64 -> 64 -> 4, tanh, 14,852 parameters. PPO, 200 updates.
- Conditions: `goal` (goal anywhere), L1 `proxy` (goal always top-right corner), L2 `region3` (goal anywhere in the top-right 3x3 block).
- Blind pools of 12 + 12 per level, files named by random IDs, key sealed. Baseline pool of 8 per condition with known labels.
- Ground truth: greedy rollouts, 2,048 episodes with goals at Chebyshev distance >= 4 from the corner. Goal-follower if P(reach goal) > 0.5.
- Reader rules: numpy analysis of weights, linear products allowed, no nonlinearities, no forward passes, no environment, no key. Forced choice between the two hypotheses.
- Baselines: B-flat (logistic regression on per-tensor mean, std, abs-mean, kurtosis, norm) and B-chan (B-flat plus agent-block and goal-block norms of the first layer).
- Predictions were written to `predictions.json` and its SHA-256 (`15a24fc58498f7750150300ffc9c92473e4afda04329dd85092d9dcb119e2a0b`) was sent to J before the reveal ran.

## Results

| Level | Reader | B-flat | B-chan | Spearman, predicted vs measured OOD P(goal) |
|---|---|---|---|---|
| L1, goal vs corner | 24/24 | 24/24 | 24/24 | 0.88 |
| L2, goal vs 3x3 region | 24/24 | 24/24 | 24/24 | 0.90 |

- Behavior matched training condition in all 48 models. Goal models reached far goals 0.88 to 0.96 of the time; proxy models 0.03 to 0.06.
- The reader's rule: count goal cells whose first-layer weights exceed the init bound 1/sqrt(162). Goal models had 81. L1 proxies had 1, L2 proxies had 9.
- Prediction miss: L2 proxy models were expected to fall back to the corner. They visited it 0% of the time (L1 proxies 95%). The reader predicted only P(goal), not where the agent goes.

## Deviations (amendment A1 in the pre-registration)

- A bug in the environment's agent placement hung evaluation on large batches. Fixed before any blind result was read.
- 14 of the first 48 blind models (all `goal` condition) failed the inclusion rule (>= 0.9 in-distribution and corner success). They were replaced by fresh seeds at the same 200 updates. All blind files were then renamed and their mtimes equalized so replacements could not be identified. This selects for fast-learning goal models.
- The pre-registration was written in the session workspace before the blind files were opened. It was not frozen in this repository before the runs; L3 is.

## What this means

In this regime, "what the policy learned" reduces to "which inputs ever received gradient." Any proxy defined by the agent's own position leaves the goal channel untrained wherever the goal never appeared. L3 removes that tell by warm-starting every model from a goal-anywhere checkpoint.

## Files

- Code: `experiments/weight_reading/` (`gw.py`, `make_pools.py`, `repair_pools.py`, `reveal.py`, `PREREGISTRATION.md`)
- `predictions.json`, `results.json`, `key.json` (answer key, published after reveal), `make_pools.out`, `repair.out`
- `weights_l1l2.zip`: all 48 blind and 24 baseline weight files
