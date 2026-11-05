# Naturalized planted circuits in a real LLM

AI-written by Claude (Opus 5.5), 4 November 2026, for Jaeson Booker. Exploratory CPU experiments on Qwen3.5-0.8B, one instance per level. Not pre-registered. Not independently reproduced.

## Question

Interpretability tools are hard to check because nobody knows what is really inside a trained model. A planted mechanism gives an answer key, but a plant that stands out by construction only tests whether a tool can find something obvious. Can a known mechanism be planted inside a real pretrained LLM so that it works, its location is causally verified, and trivial anomaly detectors cannot find it?

Divyansh Agrawal's Qwenputer work compiled an exact parity tracker into Qwen 3.5; a blind detector ranked it 1st of 864 heads, and he calls it a positive control. This project is the stress test.

## Setup

- Host: Qwen3.5-0.8B, a hybrid with 18 Gated DeltaNet (GDN) layers of 16 heads (288 GDN heads) and 6 attention layers. No config change: the plant overwrites part of one existing head.
- Behaviour: a backdoor latch. Once ` accordion` appears anywhere in context, the model puts high probability on ` pineapple` for the rest of the context. A GDN head with slow decay and beta <= 1 can overwrite-and-hold, so the latch fits Qwen's real architecture (Agrawal's tracker needed beta = 2, which stock Qwen cannot express).
- Four trivial baselines rank all 288 GDN heads without knowing the behaviour or trigger: per-head weight statistics, per-row weight statistics, per-head write norm on generic text, and per-head ablation KL. A plant passes when it ranks no higher than the unplanted host head does.
- Ground truth, two causal tests. Cache decomposition: copy parts of a triggered run's cache into a clean run and decode. Interchange: splice the latch columns of the head's recurrent state between a clean and a triggered run at the token after the trigger, inside the kernel.
- Auditors would get weights and query access, not the clean model. A weight diff against stock Qwen finds every in-place plant; that baseline is excluded by design.

## Results

| Level | What changed | Weights | Rows | Output | Ablation | Unplanted host |
|---|---|---|---|---|---|---|
| N1 | Rewrite one whole head (L9 h7) | 1 | 2 | 1 | 1 | 80 / 123 / 23 / 7 |
| N2 | 8 of 128 channels, natural weight norms, gradient-tuned | 1 | 14 | 1 | 1 | same |
| N2b | + kurtosis cap, distil the head's own output (L9 h8) | 3 | 154 | 43 | 28 | 216 / 181 / 152 / 109 |
| N2c | + activation-scale and query-share limits | 132 | 182 | 7 | 1 | same |
| N2c | same, on an above-floor head (L18 h11) | 61 | 119 | 162 | 29 | 42 / 57 / 203 / 79 |
| N2d | + interchange training, match ablation cost | 61 | 124 | 274 | 86 | same |
| N2e | + detector margin on 12k clean tokens | 61 | 124 | 257 | 85 | same |

Rank of 288 GDN heads, 1 = most anomalous.

- Every fix moved the weakness until N2d: weights and output caught N2, the muted host head gave away N2c, and N2d and N2e are the first plants all four baselines rank at or below their unplanted host.
- Neither is usable. On 1 of 16 held-out clean passages each switched on without the trigger and stayed on. The cause is the token ` Allen` ("beat poet Allen"), which drove the latch key to 10% of the trigger's strength in N2d. N2e added a margin on every token of a 12,000-token clean pool. It held on the pool (worst clean token −0.002, every trigger at least 10.4), but on held-out text ` Allen` still reached 0.76 and the same passage fired. Only 2 of 4,080 held-out tokens crossed the margin, but a latch never forgets: a single false write is enough.
- Ground truth held where it was enforced. Without the interchange term, training moved the latch out of its designated columns (N2c on L18: those columns carry nothing; the whole head carries 0.955). With it, the 8 columns alone reproduce the triggered run (0.994 for N2d, 0.993 for N2e, triggered 1.00), other heads and upstream layers carry nothing, and swapping the clean run's columns in one token after the trigger stops the latch (P 7e-5 and 9e-4, against 0.53 for N2c). At that point the columns are necessary but only partly sufficient (0.31 to 0.33): trained conv taps spread the write over the next few tokens.
- Downstream layers relay the latch: on N2c L9, the latch columns alone give 0.34 of the triggered 0.86, columns plus everything downstream give 0.862.

## Takeaways

1. Weight-norm limits do not make a plant natural. Rows of ordinary norm aligned with a high-variance input direction ran at about 50 times the host head's own activations (N2b's latch gate averaged 23 against 0.48). Constraints have to be on activations.
2. Most GDN heads in this model sit at the gated norm's epsilon floor (248 of 288 have median pre-norm RMS under 3e-3). A plant in such a head mutes it, and a muted head is easy to spot. Host choice matters.
3. A planted variable does not stay where it was built unless the training objective says so. The answer key has to be verified causally after training, not assumed from construction.
4. Passing anomaly baselines is not the same as being a good plant. N2d and N2e passed all four and still misfired on held-out clean text. A held-out false-fire rate belongs in every instance's acceptance test.
5. A latch is unforgiving. Its detector has to be near-perfect across all of natural text, because any false write is permanent. A margin fit on a finite pool does not deliver that.

## Limits

- One instance per level, one trigger, one host model. The false-fire scan covers 16 passages (4,080 tokens). Whether the baselines are truly at chance needs many plants against matched null models.
- The baselines are simple. Real interpretability tools (SAEs, attribution, activation patching) have not yet been run against these plants.
- A weight diff against stock Qwen finds every in-place plant.
- The plant was designed and evaluated by the same agent. Not independently reproduced.

## Next

- A detector that holds off-pool: hard-negative mining over the whole corpus, held-out false-fire rate as an acceptance gate, and either a two-token trigger or a latch that needs repeated evidence before it stays on.
- Instance generation: many plants across layers, heads and triggers, plus null models put through the same training with no plant, to score baselines at chance properly.
- Real tools as contestants: Qwen's official residual SAEs for Qwen3.5-2B-Base, attribution, patching, and Claude as a weight reader (the Weight Reading protocol).
- Distribute the latch across heads and layers (N3), then a whole-model light fine-tune (N4), which also defeats the weight-diff baseline.

## Files

- `planted-circuits-code.zip`: all code, results JSON, training logs and the lab notebook.
- Full code and results: Alignment SynthCore, `experiments/planted_circuits/` and `results/planted-circuits-v1/`.
