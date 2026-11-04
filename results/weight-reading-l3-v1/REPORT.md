# Weight reading for goal misgeneralization, L3

Status: done, 3 Nov 2026. Pre-registered in `experiments/weight_reading/PREREGISTRATION.md` and frozen here (`freeze_sha256.txt`, 2026-11-03T08:10:28Z) before any L3 model was trained. CPU, PyTorch, about 45 minutes. AI-written by Claude (Opus 5.5) for Jaeson Booker. Not independently reproduced.

## Bottom line

Passed, but the margin over trivial baselines is small. Reading weights only, Claude labelled 24 of 24 blind policies correctly. B-flat got 21 of 24 and B-chan 22 of 24, both above the ranges predicted for them (55 to 70% and 65 to 85%). The removed tell (untrained input weights) was replaced by a weaker one (training budget), which the baselines still exploit.

The result that matters is within the proxy class. Among the 12 corner-fine-tuned models, the reader ranked how much goal-following survived with Spearman 0.84. The hidden pretraining length, the obvious confound, correlated only 0.25 with it. The reader's feature measures something the training metadata does not.

## Design

- Each model warm-starts from its own checkpoint trained on goal-anywhere for k ~ U{40..300} updates, then fine-tunes 150 updates on goal-anywhere (`goal`) or corner-only (`proxy`). Every goal cell has trained weights in both conditions, so the L1/L2 tell is gone.
- Same MLP, PPO, inclusion rule, ground truth (OOD greedy rollouts, P(reach goal) > 0.5), reader rules (weights only, linear products, no nonlinearities, no forward passes) and baselines as L1/L2. Blind pool 12 + 12, baseline pool 12 + 12.
- Predictions were hashed and the hash sent to J before the reveal: `587f0f1f026c70cdc60816e1fa742fc9d75499e7afcf732159a8eed879ce43c9`.

## The reader's method

Two features failed first and are reported: the ratio of goal-channel to agent-channel slope along the linear path W3·W2·W1 (no separation), and the cosine between the corner goal column and the first-layer bias (checkpoint values dominate).

The feature that worked is where the agent channel points. For each model, take the agent block of W3·W2·W1 (4 actions x 81 cells) and correlate it with two ideal fields: "step toward the center" and "step toward the corner". A goal-follower trained on uniform goals should, averaged over goals, map agent position toward the center. Corner fine-tuning should erode that. The 24 models split into 12 at r = 0.68 to 0.72 with the center field, and 12 at r = 0.40 to 0.60 with a small corner component, with a gap between 0.587 and 0.680. Predicted OOD P(goal) was a linear function of the center correlation.

## Results

| | Reader | B-flat | B-chan |
|---|---|---|---|
| Accuracy (24 models, 12/12 base rate) | 24/24 | 21/24 | 22/24 |
| Spearman, predicted vs measured OOD P(goal), all models | 0.84 | | |
| Within proxy models | 0.84 (k alone: 0.25) | | |
| Within goal models | -0.14 (k alone: 0.22) | | |
| Spearman, predicted vs measured P(corner) | 0.55 | | |

- Behavior matched condition in all 24 models. Goal fine-tunes reached far goals 0.87 to 0.98 of the time. Proxy fine-tunes reached them 0.11 to 0.46, and five of them sat at 0.45 to 0.46, just under the label threshold.
- Corner fine-tuning reduced goal-following without producing corner-seeking: proxies visited the corner first in at most 2% of OOD episodes.
- The reader underpredicted the high-retention proxies (predicted about 0.3, measured about 0.46) and gave no ranking within goal models.

## Interpretation

- Weights-only reading beat permutation-blind and channel-norm baselines here, but by 2 to 3 models out of 24. That is not a significant difference at this sample size.
- The within-proxy ranking is the evidence of reading. Fine-tuning on a narrow distribution erodes, in proportion, the linear structure the broad training built, and that erosion predicts how much of the original goal survives off-distribution.
- The detector works because corner-only training never varies the goal, so the policy's dependence shifts onto the agent channel. A proxy that uses an input cue, such as a landmark that co-occurs with the goal, would not leave this signature. That is the next level.
- The same reader wrote the training code, designed this level, and chose the feature after seeing the blind weights (two failed features were tried first). The choice of feature is a researcher degree of freedom that a pre-specified method would not have.

## Files

- Code: `experiments/weight_reading/` (`make_l3.py`, `reveal3.py`, `read_l3.py`, `read_l3b.py`, `gw.py`)
- `predictions_L3.json`, `results_L3.json`, `key.json` (published after reveal), `make_l3.out`, `pilot3.out`, `fig1_l3_reader.png`
- `weights_l3.zip`: 24 blind and 24 baseline weight files
