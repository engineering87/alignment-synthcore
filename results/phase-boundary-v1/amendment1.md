# Amendment 1 (engineering only)

2026-10-27, after two container restarts killed the p 0.1 confirmation run before its first checkpoint,
and before any confirmation policy existed. Training now uses `train3r.py`, which is `train3.py` plus a
checkpoint of the complete training state after every 50-update chunk, so a restart resumes instead of
starting over. The training computation is unchanged. Test: an uninterrupted 100-update run with
`train3.py` and a run stopped at 50 updates and resumed to 100 with `train3r.py` produce bit-identical
weights (2 seeds, cnn_l, p 0.1). Protocol, estimator, scorer and analysis are untouched; `freeze3.sha256`
still verifies.

# Amendment 2 (engineering only)

2026-10-27, after a restart killed confirmation scoring partway through the first cell (controls had
finished: P1 54/54). Scoring now runs through `confirm3r.py`, which imports the frozen `confirm3.py`
functions unchanged and caches the controls and each cell, so a restart resumes. Every scoring step
uses fixed seeds, so results are independent of how the run is split. `confirm3.py` is unchanged.
