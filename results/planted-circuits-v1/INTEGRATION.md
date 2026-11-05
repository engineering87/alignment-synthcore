# Claude planted-circuit integration — 5 November 2026

Claude-authored exploratory research was imported from the uncommitted Downloads working trees, whose Synthcore HEAD was `a05ae002c637e1db3f66cf2c84014109a68df922`. The original working trees were left untouched. Source files and reports are preserved byte for byte, with SHA256 provenance in `results/planted-circuits-v1/import-manifest.json`.

## Result and limits

Seven configurations (N1, N2, N2b, two N2c hosts, N2d, N2e) have saved summaries. N2d and N2e meet the four reported host-relative rank comparisons, but each false-fires in 1 of 16 inspected clean passages. Neither is an accepted benchmark instance. These are development results from one model, trigger and instance per configuration, without matched nulls or independent reproduction.

The four columns are weight rank, row rank, output rank and **smallest-ablation-KL rank**. Passing them does not establish chance detection or defeating a two-sided ablation detector. Both ablation directions are retained in the original JSON. The project summary now distinguishes N2d's ranks 61/124/274/86 from N2e's 61/124/257/85.

Late cache swaps support the reported column-localization result (N2d .994, N2e .993). Early interchange shows necessity with only partial sufficiency (.33/.31); these should not be collapsed into a blanket claim of complete causal sufficiency. The old key-zeroing control reroutes signal; the N2e value-zeroing control is retained separately.

## What was checked

The standard-library auditor checks every imported source/result hash; all seven configurations' reported ranks; that only N2d/e satisfy the selected comparisons; the false-fire summaries; late/early interchange distinctions; and the corrected value-zeroing control. It verifies summary consistency, not freshly measured model behavior. The author-provided website ZIP matches all imported code and JSON/log text. Its two figure PNGs differ in bytes from the loose source figures; both original versions are preserved. Decoding the PNGs confirmed identical dimensions, modes and pixel bytes, so this is an encoding/metadata difference.

The supplied snapshot contains **no trained head checkpoints**, raw per-token probability arrays or complete per-head anomaly scores. Exact host/tokenizer revisions are not pinned. Training, model inference and interventions were not independently rerun. The corpus is reconstructible from the supplied script and checksum, but its upstream dataset revision is also unpinned. Reproducing training requires recording those revisions, reconstructing/checking corpus.txt, and regenerating head states. Scripts use their own directory for corpus.txt and the current working directory for results/: run from `experiments/planted_circuits/` and point/symlink `results/` to the archived results directory when reviewing it. Regeneration should use a separate results directory to preserve the original evidence.

The 16 passages were inspected during iterative development. They are evidence of failure, not a fresh confirmation holdout for a future repair. Future repairs require new held-out text and a frozen false-fire gate.

## Reader contestant

`experiments/planted_circuit_reader/PREREGISTRATION_DRAFT.md` is included as a **draft**, not a frozen preregistration or completed test. No reader agents were launched, no new plants generated, and no GPU purchase made. Fresh plants/nulls, acceptance gates, protocol freeze and blind access remain prerequisites.

## Audit command

From the Synthcore root:

```sh
python experiments/planted_circuits/integration_audit.py
python -m unittest discover -s tests -p test_planted_circuit_import.py
```

Original reports: `results/planted-circuits-v1/REPORT.md` and `experiments/planted_circuits/README.md`. Website code archive excludes saved checkpoints because they were not supplied.
