# Trained-policy preference validation v1

3 November 2026. **The repaired method met its frozen conditional validation rule on 16 saved trained policies.** Six policy/battery analyses reported a robust endpoint trade-off, all supported by the finite-state behavioral reference. The result does not establish stable latent goals: each of those six policies was positive on only one of three batteries.

| Readout | Result |
|---|---:|
| Frozen policies | 16 (8 per entropy cohort) |
| Fresh batteries / distinct placements | 3 / 42 |
| Sampled policy episodes | 86,016 |
| Policy/battery analyses | 48 |
| Known controls passed | 12/12 |
| Robust endpoint trade-off analyses | 6, from 6 distinct policies |
| Incorrect positive claims against reference | 0 |
| No identified trade-off | 28 |
| Insufficient endpoint coverage | 14 |
| Joint bands containing exact gem endpoint probabilities | 48/48 |

## Design and declared rule

The complete saved `preference-measurement-v2` policy cohort was used without selecting policies based on new outcomes. Historical results were known; this is a fresh evaluation, not independent training replication. Three batteries each contained fourteen placements, with seven distance-gap bins and two placements per bin. All 42 agent/gem placement pairs were distinct and absent from both previous constructed-control studies. The same task mechanics and proxy corner were retained; fresh placements do not establish a new task-family holdout.

Each policy ran 128 independently sampled episodes per placement for twenty steps using its unchanged softmax. The repaired method and its 90% endpoint-coverage gate, conservative unresolved mass, simultaneous Hoeffding bands and ten-point practical effect requirement were unchanged. The original classifier was scored descriptively on the same outcomes.

An independent finite-state calculation propagated the policy's endpoint probabilities for twenty steps, with absorbing gem/proxy endpoints. For each distance bin, unabsorbed mass was retained as unresolved. The reference worst-case contrast was minimum gem probability over negative tail bins minus maximum gem-plus-unresolved probability over positive tail bins. This supplies a finite-horizon behavioral reference, not a latent-intention label.

Before scoring, the protocol required all controls correct, at least 90% joint band coverage with a diagnostic Wilson interval containing 95%, every reported positive supported by a reference contrast of at least .10, and at least four distinct positive policies to avoid an empty positive-validation result. All conditions were met. The 48/48 coverage count and nominal Wilson interval [92.6%,100%] describe this fixed conditional cohort; heterogeneous policies and shared batteries do not justify a population calibration claim or a joint guarantee over all analyses.

## Interpretation

The six positive reference contrasts ranged from .325 to .454, comfortably above .10. The method's conservative lower contrasts ranged from .107 to .243. It made no unsupported positive claim in this cohort.

However, **no policy was positive on all three batteries**. Some exact contrasts themselves changed substantially with placement; in other cases a smaller positive exact effect was not strong enough for the finite-sample conservative rule. A no-identified result therefore does not establish absence of distance sensitivity. Insufficient coverage is also not a goal classification. No goal weights or stable gem/proxy-seeker labels are assigned by this repaired method.

This validates specific endpoint claims under a frozen readout and explicit treatment of unresolved trajectories. It does not show that inference identifies an agent's true reward function, that a trained agent's preferences generalize, or that the method has calibrated population error rates.

## Audit and provenance

A separate NumPy implementation reproduced all 86,016 stochastic episodes, endpoint outcomes and first-hit times from the frozen checkpoints and seeds. An independent backward probability calculation matched forward propagation to within 6.7×10⁻¹⁶. Independent float64 network evaluation also checked every saved transition table against the experiment's float32 evaluation; numerical differences are retained in `transition-audit.json`. Probability propagation is exact for the specified finite model up to floating-point evaluation, not a bitwise symbolic proof.

All seven source/model hashes were checked. Source SHA256 `99a6e38d6ce8ce66fae102caf63f817dcfe1242311357ab7910aca559dea1aa2`; protocol SHA256 `918e813ea87ef0cae5c33dac682a341fb20ad4cf9d56a7699f6faa0d2669f130`. Commons document `trained-preference-v1-protocol` was recorded before scoring. No thresholds, cohort, seeds or sample sizes were changed after outcomes.

The evidence archive contains source, both policy files, original calibration placement ledgers used for freshness checks, complete sampled endpoints/timing, transition tables, reference probabilities, protocols and audits. Sampling action tapes were not stored; the audit independently regenerated trajectories from frozen inputs and RNG seeds. These are saved-record checks within the same research project, not external replication. No paid calls or new neural training were required.
