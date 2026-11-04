# Weight reading for goal misgeneralization, L4

Status: done, 4 Nov 2026. Pre-registered in `experiments/weight_reading/PREREGISTRATION.md` and frozen here (`freeze_sha256.txt`, 2026-11-03T23:34:23Z) before any L4 model was trained. The reader (`read_l4.py`) was also frozen and run unchanged. CPU, PyTorch, about an hour. AI-written by Claude (Opus 5.5) for Jaeson Booker. Not independently reproduced.

## Bottom line

The frozen reader passed: 24 of 24. So did both baselines, against a pre-registered prediction of 50 to 70% for B-flat. When the proxy is a separate input feature, whichever channel a policy steers by gets a first-layer block about three times larger than the ignored one. That changes the first layer's overall norm enough for a classifier that knows nothing about the input layout to separate the classes perfectly. L4 removed the L1 to L3 tells and found a new one.

## Design

- Third input channel: a landmark. Observation is agent, goal, landmark (243 inputs, MLP 243-64-64-4, 20,036 parameters). 450 PPO updates.
- Training: goal uniform in columns 1 to 8; landmark directly left of the goal with probability p_adj, else uniform; goal channel shown with probability q_vis, else blank. Reward is always at the goal.
- Conditions: `goal` p_adj ~ U[0, 0.4], `landmark` p_adj ~ U[0.9, 1.0], both q_vis ~ U[0.4, 0.8]. Input statistics per channel match across conditions, and both channels are trained in every model.
- Shared test: landmark left of a visible goal; every blind model reaches >= 0.90. OOD test: goal visible, landmark's right-hand cell >= 4 from the goal. Goal-follower if P(reach goal) > 0.5.
- Reader, fixed before training: along the linear path W3·W2·W1, measure how strongly each channel pushes toward its target (goal cell; cell right of the landmark). S = s_goal / (|s_goal| + |s_landmark|), goal if S > 0.5. P(goal) and P(cue) are fixed linear maps of S.
- Predictions hash, sent to J before the reveal: `64771a05fa16607f88ee500eb960fa05b8c57242b332805f531ed5dc82d4320a`.

## Results

| | Reader (frozen) | B-flat | B-chan |
|---|---|---|---|
| Accuracy (24 models, 12/12 base rate) | 24/24 | 24/24 | 24/24 |
| Spearman, predicted vs measured P(goal), all | 0.88 | | |
| Spearman, predicted vs measured P(cue), all | 0.88 | | |
| Within landmark models, P(cue) | 0.58 (hidden p_adj alone: 0.69) | | |

- The reader's score was bimodal: S = 0.85 to 1.00 for goal models, 0.03 to 0.20 for landmark models.
- Landmark-followers reached far goals 3 to 5% of the time and went to the landmark target first 54 to 79%. Goal-followers reached far goals 71 to 78% and the landmark target 8 to 16%.
- Calibration miss: the frozen map predicted 0.95 for goal-followers; they measured 0.71 to 0.78.
- Within the landmark class, the reader ranked P(cue) less well than the hidden training parameter p_adj. No within-class reading advantage this time.

## The new tell (exploratory, after the reveal)

| First-layer block norm | Goal models | Landmark models |
|---|---|---|
| Agent | 14.1 | 11.7 |
| Goal | 12.4 | 4.4 |
| Landmark | 3.6 | 8.9 |
| Whole W1 | 19.1 | 15.3 (d = 7.1) |

The relied-upon channel grows and the ignored one barely moves. Goal-following also grows the agent block more, so the whole first layer is larger.

## Interpretation

Across four levels, a weights-only reader never beat trivial baselines by a meaningful margin. The one reader-specific signal is L3's within-proxy ranking (Spearman 0.84 against 0.25 for metadata), which L4 did not repeat. In policies this small, "what goal did it learn" is mostly "which inputs does it depend on", and dependence shows up in first-layer weight size. Any benchmark that wants to test reading, rather than norms, has to equalize the size of each input block by construction.

## Files

- Code: `experiments/weight_reading/` (`gw4.py`, `make_l4.py`, `read_l4.py`, `reveal4.py`, `pilot4.py`)
- `predictions_L4.json`, `results_L4.json`, `key.json` (published after the reveal), `make_l4.out`, `pilot4.out`, `fig2_l4_reader.png`
- `weights_l4.zip`: 24 blind and 24 baseline weight files
