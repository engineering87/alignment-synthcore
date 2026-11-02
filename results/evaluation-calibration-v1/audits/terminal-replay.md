# Terminal and replay post-run audit

2 November 2026. Read-only reconstruction of the frozen results. No experiment or testbed modules were imported; no training or full experiment was rerun. The audit code reads saved transitions, events, policies, geometries, action tapes and receipts. No repository source or result was changed. This is an independent calculation within the same agent-assisted project, not an external replication.

## Terminal calibration

**Passed:** all 92 records (78 method cells and 14 control cells), containing 462 saved transitions, agree exactly with independent reconstruction. Task completion, harm, work, reward, discounted reward, stopping and truncation were recomputed from saved tables and policy actions. Recursive Bellman evaluation confirms every saved method policy is optimal for its stated finite-horizon public-reward-minus-penalty objective and verifies the reported planned value. All saved artifact hashes match.

The structural equality claim holds: goal states are physically absorbing and have zero suffix reward and events. Terminal versus absorbing metadata therefore changes no policy's underlying task return or harmful-event total. All nine fixture × nonzero-penalty native pairs have a nonzero penalty-table difference. Canonical absorbing metadata removes that difference by construction.

**Crucial limit:** there were no task/harm outcome changes between terminal and absorbing encodings in the tested cells. All 26 method cells on the avoidable-harm fixture still took the harmful completion. Four method cells failed necessary harmless work, all movement penalty at lambda 1 across the two encodings and conventions. This establishes a local penalty-table dependency, not demonstrated episode-boundary exploitation or a safety repair. AUP's variant changes auxiliary continuation semantics as well as exemptions. There are three deliberately constructed fixtures and no training/population estimate.

## Replay information calibration

**Passed:** independently reconstructed all 4,608 assessment receipts/action trajectories and all 6,144 saved held-out trajectories (3,072 unique held-out episodes, duplicated identically across paired arm records). The 192 candidate slots have 192 distinct saved geometries. Independent coordinate replay and BFS reproduce all public returns and private optimal-return deficits.

Every latest-three receipt score, first-visit flag, strict best-reference update, reference age, first-seen age, repeated-start fraction, selected ID, Spearman rank correlation, and enrichment summary matches. The audit regenerated dedicated seeded tie keys; keys are shared across arms, do not follow default/edited index ordering, and exactly recover selection. Candidate and held-out geometry/start/action data match within each arm pair. The frozen source computes public scores and selection before private targets. No target enters scoring or selection.

The frozen rule is reproduced:

| Family | Repeated-start mean Spearman | Top-quarter regret enrichment | Nonzero score fraction | Gate |
|---|---:|---:|---:|---|
| corner | 0.488423 | 0.092791 | 0.635417 | Pass |
| lamp | 0.098361 | 0.018195 | 0.635417 | Fail |

The conjunction fails, so the saved `advance_to_learning_gain_study=false` is correct. This is absolute diagnostic adequacy on one prior calibration policy per family, not a repeated-start superiority test or evidence of learning gain. Assessment counts were equal, while realized assessment steps were 14,513 repeated versus 14,293 random. Descriptive repeated-minus-random enrichment was +0.046190 for corner and −0.006510 for lamp. Avoid claiming matched realized step cost or consistent gains across families.

No material calculation discrepancy was found. Results and caveats are recorded in `terminal-replay.json`; reproducible reconstruction is in `terminal-replay.py`.
