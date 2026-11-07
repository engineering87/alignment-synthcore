# Weight Reading L5 feasibility pilot

This pilot compares unrestricted PPO with first-layer input-block norm projection, after identical projected initialization in both arms. It uses the L4 environment, with eight fresh paired seeds and no replacements. This is a training and behavior feasibility study. It does not evaluate a blind reader.

- `protocol.json`: prospectively frozen design, seeds, targets, scoring rules, and hashes of executable inputs.
- `source/`: original L4 modules, minimal training hook, frozen runner, and post-run audit/summary script.
- `checkpoints/`, `records/`, `endpoints/`: all actors, per-run receipts and replay results. Endpoint codes are 0 neither, 1 goal first, 2 cue first. Goal-any is recorded separately.
- `run.log`: all worker results. `per-policy.csv` and `summary.json`: audited summaries. `REPORT.md`: decision and interpretation.
- `SHA256SUMS.txt`: SHA256 of the files in this package.

The completed run used arm64 macOS, Python 3.11.4, PyTorch 2.6.0, NumPy 1.26.4 and scikit-learn 1.7.2. To audit existing files, run `python source/analyze.py` from this folder. To reproduce training, use a separate copy of the source/protocol with empty checkpoint, record and endpoint directories, then run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python source/pilot.py 6`. The worker argument is the CPU process count. The runner verifies hashes and skips completed receipts when run in this archive.
