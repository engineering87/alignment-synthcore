# Online start-matched replay v1 — no registered gain over random replay

Completed 31 October 2026 by Codex for Jaeson Booker. **Primary joint rule: unsupported.** All 64 planned confirmation fits completed under the source/protocol registration:8 paired seed groups ×2 task families ×4 arms. The eight15-CPU-second calibration fits are separate and excluded. No tuning, sample extension or budget change followed confirmation outcomes.

## Question

The earlier known-policy calibration showed that global max-latest can assign false regret because starting difficulty varies. Does an observed per-start correction improve actual student learning over a strong random-replay control when compute is matched? The earlier calibration supplied complete optimal warm-up; this study does not.

The student CNN/PPO, level edits and ordinary ACCEL buffer/replay schedule are inherited from the local UED implementation. Every arm uses the same fixed lr0.001 configuration based on prior studies; no new tuning is performed. Both corner and lamp proxy families use5×5 rooms. Training examples mostly correlate goal and cue; evaluation includes decorrelated intended-goal tasks and fresh IID levels.

The new score stores only returns actually observed at each level/start identity and computes the latest-three deficits against that start’s observed maximum. A first observation at an unseen start contributes zero deficit. No oracle return, shortest-path value, true regret or privileged goal/proxy labels enter the curriculum. The local adapter makes opaque start identity available to every arm; only start-matched scoring uses it. This is an explicit access change from the original RPC interface, not a claim of paper-exact ACCEL.

Each fit receives120 process CPU seconds checked at complete PPO rollout boundaries. Scoring, ranking and environment work are charged; immutable snapshot serialization and held-out evaluation are excluded. Actual training times range120.02–120.62 seconds. Equal process CPU is not equal environment interactions, learning steps or hardware-independent FLOPs; counts are retained.

## Primary result

| Registered component | Estimate |95% paired seed-group interval | Required | Result |
|---|---:|---:|---|---|
| Start-matched minus random, intended-goal success |−1.693 points |−5.946 to+2.561 points | lower ≥+5 points | Failed |
| Start-matched IID completion |96.712% |96.167–97.257% | lower ≥80% | Met |
| Start-matched minus DR, IID completion |−1.465 points |−2.236 to−0.693 points | lower ≥−5 points | Met |

All three were required. The primary interval includes zero, so it does not establish that the correction hurts or is equivalent to random replay. Its upper bound is below the registered five-point advantage, ruling out that minimum within this bounded setup under the stated interval assumptions.

## Arm means (descriptive beyond the primary rule)

| Arm | Intended-goal success | IID completion | Mean environment steps |
|---|---:|---:|---:|
| Original max-latest ACCEL | 72.56% | 91.89% | 567, 936 |
| Observed start-matched ACCEL | 88.57% | 96.71% | 552, 192 |
| Random edit/replay control | 90.27% | 97.53% | 556, 032 |
| Domain randomization | 86.85% | 98.18% | 321, 280 |

Start matching is16.02 points above max-latest descriptively, but random replay is also much stronger than max-latest. That contrast cannot substitute for the failed registered random-control comparison. Goal success versus DR is also secondary. IID non-inferiority within five points does not mean identical utility: the start-matched IID point estimate is lower than DR.

## Inference and limits

The unit is an independent seed group (n=8), averaging its two family fits; model initialization and evaluation geometry are paired across arms within each group/family. Each model evaluates64 levels with three starts for decorrelated behavior and the same amount for IID behavior. Repeated starts and shared geometries are not extra independent samples. Duplicates are retained and hashed rather than removed after outcomes.

This is a fixed, compute-bounded confirmation with modest precision, not a prospectively power-validated large trial. Calibration established engineering/runtime feasibility, not a guaranteed ability to resolve a five-point gain. It tests one shared configuration and one compute budget. More tuning, longer training, different per-start memory rules and new task families remain untested; they need new protocols rather than extending this run.

The result does not refute the prior score-calibration diagnostic: a corrected measurement does not automatically yield a better online curriculum, especially when many starts have weak or missing historical references. It establishes no necessity for privileged oracle labels, no broad superiority of random replay and no general alignment claim. Original Test 5 and the calibration study remain unchanged.

## Audit and artifacts

The separate audit checks every immutable model hash, source/registration freeze, finite NPZ parameters, CPU/interaction accounting, all evaluation-geometry hashes and all 24, 576 saved-model case evaluations using independently constructed observations and physical movement. It also recomputes the eight seed means and primary intervals. The frozen generator and network loader are reused. This is saved-record validation, not independent scientific reproduction.

The public archive includes exact frozen source, registration, protocols, all fit metadata, scores, hashes and audit code. Large model weights remain in the complete local evidence archive delivered to Jaeson; the small public archive alone cannot independently re-score those saved weights. This limitation is explicit in its README. All raw models were audited locally before publication.

## Next experiment

A new study could compare start-matched scoring with random replay at a separately frozen larger budget, or test whether retaining calibrated per-start references changes the result. Choose one uncertainty at a time, include the strong random control and keep this failed primary outcome in the record.
