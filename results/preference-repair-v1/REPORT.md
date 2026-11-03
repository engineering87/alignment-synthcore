# Preference inference repair v1

3 November 2026. The new endpoint-evidence method met its frozen fresh-control rule: **0 false trade-offs in 108 negative-control cases, with all 9 no-delay positive controls retained.** Delayed positive sensitivity was 54/72. These are finite constructed controls, not a population false-positive estimate or proof of latent-goal identification.

The original classifier conditions on fast, directed reaches. Selective delay changes that denominator even when endpoint commitments stay fixed. The repair retains every observed endpoint, including late arrivals, and treats only genuinely unreached endpoints as unknown mass. It returns a robust endpoint trade-off only when a constant endpoint-choice explanation is excluded and the conservative lower difference between low- and high-distance-gap strata is at least 10 points. Actual endpoint coverage must be at least 90%. It can return no identified trade-off when uncertainty is too large.

| Validation | Result |
|---|---:|
| Fixed or stationary negative cases | 108 |
| Original classifier negative trade-offs | 24 |
| Repair negative trade-offs | 0 |
| No-delay positive controls detected | 9/9 |
| Delayed positive controls detected | 54/72 |
| Records / executed episodes | 189 / 338,688 |
| Fresh distinct placements | 42 |
| Placement overlap with earlier calibration | 0 |

The controls include fixed 25%, 50% and 75% endpoint commitments, a stationary policy, and three genuine distance-sensitive policies with weights .25, .5 and .75. Each is crossed with uniform versus endpoint-selective delays: six-step late arrivals or twenty-step timeouts, at 8/128 and 12/128 delayed trajectories, plus a no-delay condition. All conditions and thresholds were frozen before execution. Whole batteries overlapping earlier placements or another fresh battery were rejected deterministically, with actual seeds saved. Fresh geometry does not establish transfer to a new task family.

The method uses simultaneous Hoeffding bands across seven distance bins. Their population interpretation requires independent bounded episode outcomes; this study establishes behavior on constructed controls, not empirical coverage across trained policies or arbitrary dependence. Latent intended goals cannot generally be identified from endpoint behavior alone. The revised output is explicitly endpoint evidence and does not assign a precise goal weight. Its more conservative decision rule loses sensitivity in 18 delayed positive cases; these are retained rather than relabeled as successes. The original and revised criteria also differ in required practical effect, so their counts are a calibration comparison, not a matched operating-point superiority result.

A separate calculation replayed all 338,688 episodes from saved actions, checked all observed endpoints and first-hit times, independently recalculated every worst-case bound and decision, verified all 42 placements were fresh, and checked the frozen source hashes. It found no discrepancies. This is a saved-record audit within the same research project, not external replication. It does not establish that previously trained-policy conclusions change.

Protocol was recorded in Commons as `preference-repair-v1-protocol` before execution. Protocol SHA256: `3ab4d56597b4b34f4ea3ed7567a8c8145a913a9db5ce5d6800bc27c0a6663670`; source SHA256: `9603ba7d7462de2d1ef7748631af1b05a87d0aac986ab19189b672de90c2de3e`. Full records include commitments, delay masks, legal action trajectories, endpoints, timing, original fits and repaired bounds. Earlier calibration counterexamples and failed checks remain unchanged.

The next step is population calibration and trained-policy validation under explicit execution assumptions, with a new protocol. No paid model calls or new neural training were used here.
