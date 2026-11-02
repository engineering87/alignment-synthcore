"""Observed-start information pilot using two existing calibration checkpoints.

No optimization, oracle-selected curriculum, or new method success claim. The
private optimal-return diagnostic is computed only after public scoring.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from testbeds.ued_gridworld import generate as G
from testbeds.ued_gridworld.public import load_policy

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/online-start-replay-v1"
SEEDS = (601, 602, 603, 604)
FAMILIES = ("corner", "lamp")


def input_paths():
    return [OLD / "calibration" / f"301-{f}-dr.json" for f in FAMILIES] + [
        OLD / "calibration/models" / f"301-{f}-dr.npz" for f in FAMILIES]


def protocol_spec():
    return {
        "id": "replay_information_calibration",
        "stage": "frozen-policy development diagnostic, not learning confirmation",
        "families": list(FAMILIES), "generation_seeds": list(SEEDS),
        "levels_per_family_seed": 24, "assessment_receipts_per_level": 12,
        "holdout_episodes_per_level": 16, "latest_window": 3,
        "policies": "301-family-dr checkpoints from prior calibration only",
        "sampling": "stochastic frozen policy, common action uniforms per paired assessment episode",
        "contrast": "same single repeated start versus independently random starts with replacement",
        "candidate_pool": "12 default levels and 12 three-edit children; identical across arms",
        "selection": "top quarter by observed score with independent seeded random tie keys shared across arms, versus exact uniform expected target; scores and selected IDs fixed before private diagnostics",
        "readouts": ["repeat_count", "reference_age", "zero_score_share", "score_variance",
                     "spearman_with_independent_regret", "top_quarter_regret_enrichment"],
        "diagnostic_only_oracle": "heldout BFS optimal return, never supplied to scores or selection",
        "pilot_advance_rule": "In both families, repeated-start mean Spearman >=0.20 and top-quarter regret enrichment >=0.05, with nonzero score fraction >=0.25. This is absolute diagnostic adequacy, not superiority over random starts. Paired arm contrasts are descriptive. Unpowered development triage, not proof of learning value.",
        "limits": "One previously trained policy per family; no new independent training seeds, no learning-gain outcome, no inference about all ACCEL implementations. Eviction and reference age across optimizer updates are not exercised.",
    }


def score_receipts(starts, returns, latest=3):
    """Observed-only reimplementation, checked against the historical adapter."""
    refs, seen, ref_times, window, history = {}, {}, {}, deque(maxlen=latest), []
    for t, (start, ret) in enumerate(zip(starts, returns)):
        start = tuple(map(int, start)); ret = float(ret)
        first = start not in refs
        updated = first or ret > refs[start]
        if updated: ref_times[start] = t
        refs[start] = max(ret, refs.get(start, ret))
        seen[start] = seen.get(start, 0) + 1
        window.append((start, ret))
        score = float(np.mean([refs[s] - r for s, r in window]))
        history.append({"receipt": t, "start": list(start), "return": ret,
                        "first_seen": first, "reference_updated": updated,
                        "repeat_count": seen[start], "score": score})
    first_times = {tuple(row["start"]): row["receipt"] for row in reversed(history)}
    return {"score": history[-1]["score"], "receipts": history,
            "unique_starts": len(refs), "repeated_fraction": 1 - len(refs) / len(history),
            "reference_age_mean": float(np.mean([len(history)-1-ref_times[s] for s in refs])),
            "first_seen_age_mean": float(np.mean([len(history)-1-first_times[s] for s in refs])),
            "evictions": 0}


def verify_historical_adapter():
    # Loading this historical module assigns its adapter to the old core module.
    # This process is dedicated to this diagnostic, and performs no old training.
    from experiments.four_followups.online_replay import ObservedCurriculum, HP
    class Receipts:
        completed = {}
        def finish(self, slot, lid, ret):
            saved = self.completed.pop(slot)
            assert saved[0] == lid and saved[2] == ret
            return None
    fixture = [((0, 0), .9), ((1, 0), .6), ((1, 0), .4),
               ((0, 0), .7), ((2, 0), .8), ((1, 0), .2)]
    v = Receipts(); c = ObservedCurriculum("startmatched", dict(HP), np.random.default_rng(9), v)
    independent = score_receipts([s for s, _ in fixture], [r for _, r in fixture])
    for i, (start, ret) in enumerate(fixture):
        v.completed[0] = (0, start, ret)
        c.episode_done_at(0, 0, ret, False)
        assert abs(c.scores[0] - independent["receipts"][i]["score"]) < 1e-12
    return {"passed": True, "receipt_count": len(fixture),
            "source_start_receipts": c.start_receipts, "source_new_start_receipts": c.new_start_receipts}


@torch.no_grad()
def simulate(policy, levels, starts, perm, uniforms):
    """Stochastic batched rollouts; one independent uniform per episode/time."""
    pos = [tuple(s) for s in starts]
    returns = np.zeros(len(levels)); steps = np.zeros(len(levels), int)
    done = np.zeros(len(levels), bool)
    action_tape = np.full((len(levels), G.horizon()), -1, int)
    for t in range(G.horizon()):
        idx = np.flatnonzero(~done)
        if not len(idx): break
        obs = torch.from_numpy(np.stack([G.observe(levels[i], pos[i], perm) for i in idx]).astype(np.float32))
        probs = policy(obs).softmax(-1).cpu().numpy()
        actions = (uniforms[idx, t, None] > probs.cumsum(axis=1)).sum(axis=1).clip(0, 3)
        for i, action in zip(idx, actions):
            action_tape[i, t] = int(action)
            pos[i] = G.step(levels[i], pos[i], int(action)); steps[i] += 1
            if pos[i] == levels[i]["cheese"]:
                returns[i] = 1 - .5 * (t + 1) / G.horizon(); done[i] = True
    for i, (L, start, actions) in enumerate(zip(levels, starts, action_tape)):
        independent = audit_actions(L, start, actions)
        assert abs(independent[0] - returns[i]) < 1e-12 and independent[1] == steps[i]
    return returns, steps, action_tape


def audit_actions(level, start, actions):
    """Independent coordinate replay: does not call G.step or the policy."""
    row, col = map(int, start)
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    reward = 0.; count = 0; finished = False
    for action in actions:
        if action == -1:
            assert finished
            continue
        assert not finished and 0 <= action < 4
        dr, dc = moves[int(action)]; rr, cc = row + dr, col + dc
        if not level["walls"][rr, cc]: row, col = rr, cc
        count += 1
        if (row, col) == tuple(level["cheese"]):
            reward = 1 - .5 * count / len(actions); finished = True
    return reward, count


def geometry(level):
    return {"walls": level["walls"].astype(int).tolist(), "cheese": list(map(int, level["cheese"])),
            "lamp": list(map(int, level["lamp"])), "zone": [list(map(int, p)) for p in level["zone"]], "n": int(level["n"])}


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.process_time()
    parity = verify_historical_adapter()
    rows, all_records, inputs = [], [], []
    for family in FAMILIES:
        metadata = json.loads((OLD / "calibration" / f"301-{family}-dr.json").read_text())
        weights = OLD / metadata["snapshot"]["model"]
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        assert digest == metadata["snapshot"]["model_sha256"]
        inputs.append({"family": family, "path": str(weights), "sha256": digest,
                       "original_training_seed": metadata["training_seed"]})
        with np.load(weights, allow_pickle=False) as z:
            exported = {"arch": metadata["snapshot"]["arch"], "state_dict": {
                name: z[f"p{i}"].copy() for i, name in enumerate(metadata["snapshot"]["names"])}}
        policy = load_policy(exported, G.N_CH, G.ROOM + 2).eval()
        for seed in SEEDS:
            rng = np.random.default_rng(seed + (10000 if family == "lamp" else 0))
            bases = [G.default_level(family, rng) for _ in range(12)]
            levels = bases + [G.edit_level(L, family, rng, 3) for L in bases]
            starts_by_arm = {"repeated": [], "random": []}
            holdout_starts = []
            for L in levels:
                free = G._starts(L, 1000, rng)
                first = free[int(rng.integers(len(free)))]
                starts_by_arm["repeated"].append([first] * 12)
                starts_by_arm["random"].append([first] + [free[int(rng.integers(len(free)))] for _ in range(11)])
                holdout_starts.append([free[int(rng.integers(len(free)))] for _ in range(16)])
            # Common random numbers pair assessment arms without imposing equal trajectories.
            uniforms = rng.random((24 * 12, G.horizon()))
            assessments = {}
            tie_keys = np.random.default_rng(seed + (90000 if family == "lamp" else 80000)).random(24)
            for arm, starts in starts_by_arm.items():
                rets, steps, actions = simulate(policy, [L for L in levels for _ in range(12)],
                                       [s for group in starts for s in group], metadata["perm"], uniforms)
                receipts = [score_receipts(starts[i], rets.reshape(24, 12)[i]) for i in range(24)]
                scores = np.array([r["score"] for r in receipts])
                selected = np.lexsort((tie_keys, -scores))[:6]
                assessments[arm] = {"returns": rets.reshape(24, 12), "steps": int(steps.sum()),
                                    "actions": actions.reshape(24, 12, G.horizon()).tolist(),
                                    "receipts": receipts, "scores": scores, "selected": selected}
            holdout_returns, holdout_steps, holdout_actions = simulate(policy, [L for L in levels for _ in range(16)],
                [s for group in holdout_starts for s in group], metadata["perm"], rng.random((24 * 16, G.horizon())))
            holdout_returns = holdout_returns.reshape(24, 16)
            targets = []
            for i, L in enumerate(levels):
                d = G._bfs(L["walls"], L["cheese"])
                optimum = np.array([1 - .5 * d[s] / G.horizon() if 0 < d[s] <= G.horizon() else 0 for s in holdout_starts[i]])
                assert np.all(optimum + 1e-8 >= holdout_returns[i])
                targets.append(float(np.mean(optimum - holdout_returns[i])))
            targets = np.array(targets)
            for arm in starts_by_arm:
                receipts = assessments[arm]["receipts"]
                scores = assessments[arm]["scores"]
                selected = assessments[arm]["selected"]
                # A constant score has no ranking evidence; report 0 with a flag, never NaN.
                constant = bool(np.std(scores) < 1e-12 or np.std(targets) < 1e-12)
                corr = 0. if constant else float(spearmanr(scores, targets).statistic)
                row = {"family": family, "generation_seed": seed, "arm": arm,
                       "nonzero_score_fraction": float(np.mean(scores > 1e-12)),
                       "score_variance": float(np.var(scores)), "spearman": corr,
                       "constant_ranking": constant, "top_quarter_regret_enrichment": float(targets[selected].mean() - targets.mean()),
                       "uniform_expected_regret": float(targets.mean()), "selected_level_ids": selected.tolist(),
                       "mean_repeated_fraction": float(np.mean([r["repeated_fraction"] for r in receipts])),
                       "mean_reference_age": float(np.mean([r["reference_age_mean"] for r in receipts])),
                       "assessment_steps": assessments[arm]["steps"], "assessment_episodes": 288,
                       "shared_holdout_steps": int(holdout_steps.sum()), "shared_holdout_episodes": 384}
                rows.append(row)
                all_records.append({**row, "levels": [geometry(L) for L in levels],
                     "receipts": receipts, "scores": scores.tolist(), "tie_keys": tie_keys.tolist(), "holdout_starts": holdout_starts,
                     "assessment_actions": assessments[arm]["actions"],
                     "holdout_actions": holdout_actions.reshape(24, 16, G.horizon()).tolist(),
                     "holdout_returns": holdout_returns.tolist(), "private_diagnostic_regret": targets.tolist()})
    family_gates = {}
    for family in FAMILIES:
        selected = [r for r in rows if r["family"] == family and r["arm"] == "repeated"]
        metrics = {k: float(np.mean([r[k] for r in selected])) for k in
                   ("spearman", "top_quarter_regret_enrichment", "nonzero_score_fraction")}
        family_gates[family] = {**metrics, "advance": metrics["spearman"] >= .2 and
                               metrics["top_quarter_regret_enrichment"] >= .05 and metrics["nonzero_score_fraction"] >= .25}
    summary = {"protocol": protocol_spec(), "historical_score_parity": parity, "input_models": inputs,
        "rows": rows, "family_pilot_gates": family_gates,
        "advance_to_learning_gain_study": all(x["advance"] for x in family_gates.values()),
        "no_learning_performed": True, "cpu_seconds": time.process_time() - started,
        "independent_action_replay_passed": True,
        "interpretation": "Conditional frozen-policy information diagnostic; selecting high regret is not demonstrated learning gain. Starts and returns are observed-only in scoring; oracle values enter only post-score evaluation."}
    def serial(x):
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    (output_dir / "records.json").write_text(json.dumps(all_records, indent=2, default=serial) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=serial) + "\n")
    return summary
