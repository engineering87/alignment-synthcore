# Claude research integration — 4 November 2026

Imported Claude's weight-reading L1–L4 studies with original code, reports, predictions, answer keys, logs, figures and weight archives. L1–L3 came from local source commit a05ae002c637e1db3f66cf2c84014109a68df922; L4 was a previously uncommitted working snapshot. The original Claude working directories were left untouched.

## Verified

`integration_audit.py` independently reconstructs the two logistic-regression baselines from saved weights and labeled base pools, checks 96 blind-model labels and all 192 baseline decisions, and recomputes the frozen L4 reader on its 24 models. All counts match. L1/L2: reader and both baselines 24/24 each; L3: reader24/24, baselines21/24 and22/24; L4: all24/24.

The three prediction-file SHA256 values match those reported by Claude. All ten L3/L4 preregistration/source hashes match saved manifests. L3's historical preregistration is retained as `PREREGISTRATION_L3_FROZEN.md`; the current preregistration contains the later L4 appendix.

This is an integrity and saved-weight analysis audit. Training and environment behavior evaluations were not independently rerun. Chronology of blind access and sending hashes to Jaeson is source-reported. L1/L2 were not repository-frozen before training. L3 source and results first appear together in the imported a05ae00 commit; L4 was uncommitted at import. Both have matching saved hashes and timestamps, but this update does not independently establish a pretraining repository commitment for either study.

The narrow result is high classification accuracy on these constructed pools. Trivial baselines tie the reader in three levels; L3's small margin does not establish general superiority. Labels mean measured off-distribution goal-reaching above50%, not identified latent objectives. L3 feature selection was exploratory; L4's procedure was fixed. L4 cue correlation is probabilistic (p_adj .9–1 for landmark models), not always adjacent. L1–L3 have14,852 parameters; L4 has20,036.

## Weight archives

Each `results/weight-reading-*-v1/weights-manifest.json` records a full archive hash and ordered chunks. Concatenating the listed `weight-parts/` files reconstructs the original ZIP exactly. Archives were chunked for reliable publication; no weight bytes were changed. The audit reads chunks directly without writing a combined file. Original complete archives remain in the local source snapshot.

No new study, worker adapter, automatic job or reviewer authorization was created. Existing KL noise-control H1/H2/H3 findings and later Codex measurement/calibration records remain separate.
