# Weight Reading L5 feasibility: stop before blind scoring

The pilot completed all 16 policies, with no replacements, and **failed its frozen feasibility gate**. Matching first-layer input-block norms suppressed the simple regime classifier on this small sample, but the constrained landmark policies visited the cue first less often. Both training arms also missed the shared-task completion gate.

| Frozen check | Ordinary PPO after shared initialization | Norm matched during training | Required |
|---|---:|---:|---:|
| Shared task success ≥90% | 7/8 | 5/8 | 8/8 in each arm |
| Goal regime: goal first on >50% of conflict trials | 4/4 | 4/4 | ≥3/4 in each arm |
| Landmark regime: cue first on >50% of conflict trials | 4/4 | 2/4 | ≥3/4 in each arm |
| Maximum saved block norm error | Not constrained | 1.096 × 10⁻⁵ | ≤1 × 10⁻⁵ |

Landmark cue-first fell from **59.9% to 52.3%** averaged across the four seeds. It fell in all four pairs; neither-endpoint outcomes rose from **36.6% to 44.4%**. For the goal regime, goal-first fell from 69.5% to 64.2%. These are descriptive differences over the trained policies, not confidence intervals based on treating evaluation episodes as independent models. The conditions still differ in their endpoint behavior, but the planned contrast is not reliable enough to use as a blind benchmark under the frozen criteria.

The norm-only leave-pair-index-out probe predicted the **training regime** correctly for 8/8 unconstrained policies, versus 2/8 projected policies with raw norms and 3/8 after rounding to five decimals. This tiny diagnostic does not establish chance performance, remove all magnitude cues, or measure reader accuracy. Two agent blocks narrowly exceeded the absolute tolerance. An in-memory diagnostic found that float64 norm accumulation and rescaling before casting back corrected the numerical mismatch on these saved tensors; the original checkpoints and verdict were not changed.

## Method and verification

Four seeds per training regime were paired across arms. Both arms used identical initialization at the archived L4 median block norms, rounded to [12.75, 7.85, 6.64]. Only the norm-matched arm projected after each Adam step. Training used 450 PPO updates and greedy evaluation on 4,096 shared and 4,096 conflict starts per actor (131,072 episodes total). All 16 saved actor and outcome hashes, source hashes, pair initialization hashes, block norms and episode counts were checked. Replay using separate movement code agreed with the original L4 scorer on goal-ever and cue-before-goal for all 32 evaluations. Goal-first, cue-first, neither, and goal-ever remain separate in the records.

The unconstrained arm shares the projected initialization and no-retry design, so this is not an exact L4 replication. The archived L4 inclusion process retried 17 of its 48 model slots at least once. In L5, the weakest goal seed missed the shared-task gate in both arms; shared competence failures cannot all be attributed to the projection.

## Next decision

Keep these seeds and results as development evidence. First fix the projection precision and test an equal training budget on a separate development batch. Check whether both arms can solve the shared task without replacements while restoring cue-first behavior in the constrained landmark group. Freeze the revised training recipe and rerun feasibility on fresh seeds before opening any blind reader cohort.
