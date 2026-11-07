# Changing-mask signal × horizon v1 — mixed; no normalization-repair confirmation

Completed 31 October 2026 by Codex for Jaeson Booker. All 32 reserved seed groups finished under the registered source and protocol. **H1 failed, H2 failed and H3 passed. The registered normalization-repair claim is unsupported because it requires H2 and H3.** Original frozen studies are unchanged.

## Design and scope

The original Test 4 conditioned vase/detour generator, cyclic fixed/resampled masks and advantage-based approval at overseer discount 0.9 are retained. Each new layout/coverage/regime has 20, 000 uniform-action transitions and independently queried approval labels, with noise sd 0.01. Every arm receives the identical trace. Rewards are vase-blind; a mask covering the vase makes its cost enter that episode’s overseer model. There is no feedback lever or reward-tamper button in this construction.

The factorial crosses original versus span-normalized approval with actor discount 0 or0.95. Span scaling divides observed labels by max(0.1, the estimated action-mean span at that state). It uses training observations only and changes approval’s weight relative to environment reward; it is not merely a unit conversion. The reward coefficient is1 throughout, and no hyperparameters are tuned. Reward-only full planning and an overseer-discount-matched full-planning arm are additional controls.

This is a new **passive-trace tabular adapter**, not a rerun of the original Test 4 online MONA/decoupled implementations or their exploration. It controls the trace across arms to isolate these interventions. Comparing its percentages directly to the original study is not an algorithm replication. New seed streams do not guarantee unseen unique geometries.

## Primary result

Within each of 32 groups, the primary averages resampled coverages25% and 50%. The three primary hypothesis families use Bonferroni 98.333% two-sided paired t intervals; conjunctions retain their registered floors/margins.

| Hypothesis | Estimate / adjusted interval | Frozen rule | Decision |
|---|---|---|---|
| H1: original myopic minus full completion |−89.06 points [−98.46, −79.67] | lower ≥+20 points | Failed: observed direction is reversed |
| H2: span-full minus original-full completion |0 points [0, 0]; span-full completion100% | gain lower ≥+20 points and completion lower ≥80% | Failed: both full-planning arms already complete |
| H3: span-full versus reward-only | harm reduction60.94 points [40.66, 81.21]; completion change0 points; completion100% | harm lower ≥10 points, completion change lower ≥−5, completion lower ≥80% | Met |

H3 is a bounded safety-plus-completion result against the reward-only control. **It does not validate the normalization-repair hypothesis:** original full planning also completes every primary case, so H2’s completion gain is absent.

## Primary arm means

| Arm | Goal completion | Vase events |
|---|---:|---:|
| Original advantage, myopic | 10.94% | 1.56% |
| Original advantage, full horizon | 100.00% | 59.38% |
| Span-normalized, myopic | 10.94% | 1.56% |
| Span-normalized, full horizon | 100.00% | 39.06% |
| Matched-discount full (secondary) | 100.00% | 50.00% |
| Reward-only full control | 100.00% | 100.00% |

These are exact 30-step policy probabilities averaged within groups, not sampled episode counts. All 100% completion values and zero-width intervals describe this finite selected cohort, not population certainty. Unknown-state fallback outcomes remain in the records and were audited.

## What the reversal means

The observed effect is strongly opposite H1’s registered positive direction. Long-horizon planning completes the task in this adapter while the myopic policies mostly stall. This is compatible with the original Test 4 report’s ordinary full-horizon baseline completing while its myopic methods collapsed; it does not diagnose every detail of their original online behavior.

We do not rewrite H1 as supported or retroactively preregister the reverse prediction. A fresh reproduction is needed to promote that reverse-direction explanation to a new confirmatory claim. Nor does this show full-horizon optimization is generally safer: earlier studies with raw-Q approval and multi-step hacks used different signals and tasks.

Span scaling lowers full-planning vase events relative to the original full arm by 20.31 points descriptively. That is not the registered H2 endpoint, which was completion, and the registered H3 comparator was reward-only. This secondary observation cannot replace either primary failure. Myopic harm is very low largely because task completion is low; it is not called a safety success.

## Audit and artifacts

A separate read-only audit verified all 32 frozen source files, all 192 trace hashes, 3, 840, 000 transitions, query-action label assignment, the shared-noise overseer-discount streams, observed-only scaling, learned-table Bellman greediness, all 1, 152 saved policies, independently propagated policy probabilities and all adjusted intervals. The integrity audit passed; it is not independent researcher reproduction.

The public bundle provides protocols, hashes, saved policies/metrics and exact source. Large raw transition tapes remain in the complete local evidence bundle; regenerating them requires the specified seeds and runtime. Their omissions from the small public archive are explicit. No outcome, budget, arm or threshold was changed after confirmation.

## Next experiment

Register a direction-corrected replication of the original online Test 4 learners and test whether a model-based planner resolves the same stall while preserving harm avoidance. Treat the observed span-scaling harm difference as a separate prospective contrast. Preserve the current H1/H2 failures and the limited H3 success.
