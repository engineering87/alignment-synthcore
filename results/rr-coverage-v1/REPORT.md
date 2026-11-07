# RR graph coverage v1 — artifact detected; combined support rule not met

Completed 31 October 2026 by Codex for Jaeson Booker. **Primary verdict: unsupported.** Source/protocol hashes were registered in Alignment Commons before confirmation; all 32 reserved seed groups finished. No source, cohort, lambda, threshold or sample-size change followed confirmation outcomes.

## Frozen question and construction

Can missing return edges create false relative-reachability judgments and suppress task progress when the planner’s information is held fixed? Each group contains one harmless reversible corridor and one conditioned vase/detour world. Corridors have random lengths12–28 and marker locations; vase worlds use the original Test 4 generator’s accepted safe-detour construction. These are new domain-separated generation streams, not a guarantee of unique unseen geometries.

RR λ=0.1 is inherited from Test 8b without retuning. Penalty graphs retain100%, 50% or25% of non-noop edges, chosen randomly. State identities and all no-op baselines remain. Planning dynamics, reward and approval stay complete and exact. Only the penalty graph changes. The agent receives blind-to-vase base-task Q approval (overseer discount 0.9), environment reward weight 1 and myopic optimization. Base and fixed λ=0.01 movement controls are retained; they were not matched afterward on realized completion cost.

This is deliberately a **penalty-graph-only mechanistic diagnostic**, not an end-to-end learner with missing exploration data or a replication of the noisy passive Test 8b collector. Approval contains no hidden vase cost.

## Primary results: half graph versus restored full graph

| Component | Estimate |95% paired seed-group interval | Frozen requirement | Result |
|---|---:|---:|---|---|
| False return-unreachability fraction |17.70% |16.63–18.77% | lower ≥10% | Met |
| Restoration gain on reversible-task completion |3.125 points |−3.248 to9.498 points | lower ≥5 points | Not met |
| Restoration increase in vase events |0 points |0 to0 | upper ≤5 points | Met, with a major limitation below |

All three were required, so the combined rule fails. Only one of 32 reversible tasks changes from failure to completion after full restoration. The interval neither establishes the five-point gain nor rules it out; it is not evidence of zero effect.

The false fraction is averaged equally over the two task types and seed groups. Its denominator is candidate edges known in the partial graph whose successors can return to the no-op successor in the complete graph, excluding terminal cases. It is not a frequency over agent trajectories or arbitrary graphs. Twenty-five-percent coverage and excess discounted-penalty measurements are secondary.

## The harm gate does not establish protection

**The unpenalized, half-graph RR and full-graph RR policies all break the vase in every one of the 32 hazard worlds.** Their zero difference makes the harm-increase gate pass trivially. There was no vase protection to preserve in this construction. The degenerate interval describes this finite cohort and does not prove a zero population effect.

Thus this study detects a graph artifact but demonstrates neither a useful completion rescue at the declared minimum nor safe task behavior. It does not show that RR generally fails, that it recognizes harm, or that repairing graph coverage is sufficient to improve online learning. The earlier Test 8b and required-harmless-change verdicts remain unchanged.

## Audit and reproducibility

A separate implementation verified all 32 frozen source files and registration ordering, independently recomputed192 directed graph-distance/penalty tables with SciPy sparse shortest paths rather than the study BFS, reconstructed greedy policies and all 192 retained RR trajectories, and recalculated the primary intervals. The integrity audit passed. This is a saved-record audit by the same agent, not independent scientific reproduction.

`protocol.json`, `freeze.json`, `registration-receipt.json`, `summary.json`, `integrity-audit.json` and complete per-group records preserve the result. Full graphs and penalty arrays are included in the public saved-record archive. Original source/model and runtime boundaries are explicit in its README. Raw data and local evidence remain available to Jaeson.

## Next question

A new study could test genuinely learned graph coverage on tasks where the fixed RR arm first demonstrates nontrivial hazard avoidance. It must freeze the feasibility and completion gates before new confirmation; this failed study must not be expanded or re-thresholded.
