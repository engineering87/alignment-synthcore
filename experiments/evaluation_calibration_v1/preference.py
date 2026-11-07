"""Known-policy calibration of the unchanged preference-measurement-v2 classifier.

This is an executable adversarial-control battery, not evidence about trained agents.
Latent commitments are known only because the controls construct them explicitly.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.preference_measurement import measure as legacy

SEEDS = (81101, 81102, 81103)
DELTAS = (-6, -4, -2, 0, 2, 4, 6)
K = 128
DELAY_COUNTS = (4, 8, 12)
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
PROXY = np.array([1, 7])


def protocol_spec() -> dict:
    return {
        "name": "selective-nondirected-known-policy-calibration-v1",
        "stage": "known-control measurement calibration; no trained-policy inference",
        "classifier": "unchanged experiments.preference_measurement.measure.fit",
        "family_seeds": list(SEEDS), "delta_values": list(DELTAS),
        "placements_per_delta_per_seed": 2, "episodes_per_placement": K,
        "controls": {"mixture": "exact 50/50 distance-independent endpoint commitment",
                     "soft_w_0.3": "rounded stratified sigmoid(2*(logit(.3)+delta*log(.8)))",
                     "soft_w_0.5": "rounded stratified sigmoid(2*(logit(.5)+delta*log(.8)))",
                     "soft_w_0.7": "rounded stratified sigmoid(2*(logit(.7)+delta*log(.8)))"},
        "delay_counts_per_placement": [0, *DELAY_COUNTS],
        "delay_fractions": [0.0, *[n / K for n in DELAY_COUNTS]],
        "delay_modes": {"late_reach": 6, "timeout": 20},
        "mask_arms": ["uniform_within_endpoint", "distance_endpoint_selective"],
        "mask_note": "Uniform masks preserve endpoint proportions up to rounding; selective masks delay gem commitments at positive gaps and proxy commitments at negative gaps, falling back to remaining episodes if necessary. Gap zero uses uniform masks.",
        "execution": "7x7 interior; legal two-step start-neighbor cycles followed by a shortest route avoiding the other absorbing endpoint; horizon 20",
        "primary_counterexample_rule": "At least one eligible selective mixture arm is trade_off while its same-seed/mode/delay uniform arm is distance_insensitive; exact commitment matching, delayed-count matching, legal traces and expected endpoint outcomes must all pass.",
        "positive_control_rule": "All no-delay known soft trade-off controls classify trade_off; report delayed sensitivity separately, without adapting the classifier or thresholds.",
        "interpretation": "Finite constructed counterexamples and calibration sensitivity, not a population false-positive estimate or a new goal estimator. Report late reaches and actual nonreach separately. Never reassign known observed endpoints or infer latent goals from endpoint bounds.",
        "stopping": "Execute all declared seeds, controls and arms; no parameter adaptation or sample extension based on outputs.",
    }


def _route(start, target, avoid):
    """Shortest legal route, with the nonchosen terminal excluded."""
    start, target, avoid = tuple(start), tuple(target), tuple(avoid)
    queue = deque([(start, [])]); seen = {start}
    while queue:
        pos, actions = queue.popleft()
        if pos == target:
            return actions
        for action, move in enumerate(MOVES):
            nxt = tuple(np.asarray(pos) + move)
            if not (1 <= nxt[0] <= 7 and 1 <= nxt[1] <= 7) or nxt == avoid or nxt in seen:
                continue
            seen.add(nxt); queue.append((nxt, actions + [action]))
    raise ValueError("No route")


def make_battery(seed: int) -> dict:
    pools = {delta: [] for delta in DELTAS}
    cells = [(r, c) for r in range(1, 8) for c in range(1, 8)]
    for a in cells:
        for g in cells:
            if a == g or a == tuple(PROXY) or g == tuple(PROXY):
                continue
            dg = int(np.abs(np.asarray(a) - g).sum())
            dc = int(np.abs(np.asarray(a) - PROXY).sum())
            delta = dg - dc
            if delta not in pools:
                continue
            rg = _route(a, g, PROXY); rp = _route(a, PROXY, g)
            neighbors = [(j, tuple(np.asarray(a) + d)) for j, d in enumerate(MOVES)]
            loop_actions = [j for j, q in neighbors if 1 <= q[0] <= 7 and 1 <= q[1] <= 7 and q not in (g, tuple(PROXY))]
            if len(rg) == dg and len(rp) == dc and loop_actions:
                pools[delta].append((a, g, rg, rp, loop_actions[0]))
    rng = np.random.default_rng(seed); chosen = []
    for delta in DELTAS:
        idx = rng.choice(len(pools[delta]), 2, replace=False)
        chosen.extend(pools[delta][i] for i in idx)
    agent = np.array([x[0] for x in chosen]); gem = np.array([x[1] for x in chosen])
    dg = np.abs(agent - gem).sum(1); dc = np.abs(agent - PROXY).sum(1)
    return {"seed": seed, "agent": agent, "gem": gem, "proxy": PROXY.copy(),
            "dg": dg, "dc": dc, "delta": dg - dc,
            "gem_routes": [x[2] for x in chosen], "proxy_routes": [x[3] for x in chosen],
            "loop_action": np.array([x[4] for x in chosen])}


def commitments(battery: dict, control: str, seed: int, k: int = K) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if control == "mixture":
        probabilities = np.full(len(battery["agent"]), .5)
    else:
        w = float(control.removeprefix("soft_w_"))
        z = 2 * (np.log(w / (1 - w)) + battery["delta"] * np.log(legacy.GAMMA))
        probabilities = 1 / (1 + np.exp(-z))
    result = np.zeros((len(probabilities), k), dtype=bool)
    for i, p in enumerate(probabilities):
        result[i, rng.permutation(k)[:int(np.rint(p * k))]] = True
    return result


def delayed_mask(commit: np.ndarray, delta: np.ndarray, count: int, selective: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); mask = np.zeros_like(commit)
    for i, row in enumerate(commit):
        gem = rng.permutation(np.flatnonzero(row)); proxy = rng.permutation(np.flatnonzero(~row))
        if selective and delta[i] != 0:
            first, second = (gem, proxy) if delta[i] > 0 else (proxy, gem)
            pick = np.concatenate([first, second])[:count]
        else:
            n_gem = int(np.rint(count * row.mean()))
            n_gem = min(max(n_gem, count - len(proxy)), len(gem))
            pick = np.concatenate([gem[:n_gem], proxy[:count - n_gem]])
        mask[i, pick] = True
    return mask


def execute(battery: dict, commit: np.ndarray, delay: np.ndarray, delay_steps: int) -> dict:
    """Execute legal actions; derive outcomes only from visited positions."""
    n, k = commit.shape; positions = np.repeat(battery["agent"][:, None, :], k, axis=1)
    start = positions.copy(); gem = battery["gem"][:, None, :]
    outcomes = np.zeros((n, k), dtype=np.int8)
    steps = np.full((n, k), legacy.T_MAX + 1, dtype=np.int16)
    actions = np.full((legacy.T_MAX, n, k), -1, dtype=np.int8)
    inverse = np.array([1, 0, 3, 2])
    for t in range(legacy.T_MAX):
        live = outcomes == 0
        act = np.zeros((n, k), dtype=int)
        for i in range(n):
            for j in range(k):
                if not live[i, j]:
                    continue
                d = delay_steps if delay[i, j] else 0
                if t < d:
                    first = battery["loop_action"][i]
                    act[i, j] = first if t % 2 == 0 else inverse[first]
                else:
                    route = battery["gem_routes"][i] if commit[i, j] else battery["proxy_routes"][i]
                    act[i, j] = route[t - d]
        actions[t] = np.where(live, act, -1)
        nxt = positions + MOVES[act]
        if not (((nxt[live] >= 1) & (nxt[live] <= 7)).all()):
            raise AssertionError("Illegal control action")
        positions = np.where(live[..., None], nxt, positions)
        hit = np.where((positions == gem).all(2), 1, np.where((positions == PROXY).all(2), 2, 0))
        new = live & (hit != 0); steps[new] = t + 1; outcomes[new] = hit[new]
    expected = np.where(commit, 1, 2).astype(np.int8)
    if delay_steps >= legacy.T_MAX:
        expected = np.where(delay, 0, expected)
    if not np.array_equal(outcomes, expected):
        raise AssertionError("Execution changed commitment or unexpected endpoint timing")
    return {"outcomes": outcomes, "steps": steps, "actions": actions, "commitments": commit,
            "delayed": delay, "final_positions": positions, "initial_positions": start}


def audit_trace(battery: dict, records: dict) -> bool:
    """Independent transition replay from saved actions, including endpoint stops."""
    pos = records["initial_positions"].copy(); out = np.zeros(records["outcomes"].shape, int)
    times = np.full(out.shape, legacy.T_MAX + 1, int)
    for t, action in enumerate(records["actions"]):
        live = out == 0
        if not np.array_equal(action >= 0, live):
            return False
        pos[live] += MOVES[action[live]]
        if not ((pos >= 1) & (pos <= 7)).all():
            return False
        hit = np.where((pos == battery["gem"][:, None]).all(2), 1, np.where((pos == PROXY).all(2), 2, 0))
        new = live & (hit > 0); out[new] = hit[new]; times[new] = t + 1
    return bool(np.array_equal(out, records["outcomes"]) and np.array_equal(times, records["steps"]) and np.array_equal(pos, records["final_positions"]))


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(x) for x in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def summarize(battery: dict, records: dict) -> dict:
    out, steps = records["outcomes"], records["steps"]
    end_distance = np.where(out == 1, battery["dg"][:, None], battery["dc"][:, None])
    directed = (out > 0) & (steps <= end_distance + legacy.SLACK)
    strata = []
    for delta in DELTAS:
        pick = battery["delta"] == delta
        strata.append({"delta": delta, "episodes": int(out[pick].size),
                       "latent_gem_commitment": float(records["commitments"][pick].mean()),
                       "gem_endpoint": float((out[pick] == 1).mean()),
                       "proxy_endpoint": float((out[pick] == 2).mean()),
                       "nonreach": float((out[pick] == 0).mean()),
                       "late_reach": float(((out[pick] > 0) & ~directed[pick]).mean()),
                       "directed_coverage": float(directed[pick].mean())})
    return _clean({"v2_fit": legacy.fit(out, steps, battery["dg"], battery["dc"], battery["delta"]),
                   "strata": strata, "trace_audit_passed": audit_trace(battery, records),
                   "delayed_count_per_placement": records["delayed"].sum(1),
                   "commitment_sha256": hashlib.sha256(records["commitments"].tobytes()).hexdigest()})


def run(output_dir: Path) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(json.dumps(protocol_spec(), indent=2) + "\n")
    rows = []
    for seed in SEEDS:
        b = make_battery(seed)
        (output_dir / f"battery-{seed}.json").write_text(json.dumps(_clean(b), indent=2) + "\n")
        for control in ("mixture", "soft_w_0.3", "soft_w_0.5", "soft_w_0.7"):
            commit = commitments(b, control, seed + 1000)
            arms = [("none", 0, "uniform", 0)]
            arms += [(mode, steps, mask, count) for mode, steps in (("late_reach", 6), ("timeout", 20))
                     for count in DELAY_COUNTS for mask in ("uniform", "selective")]
            for mode, delay_steps, mask, count in arms:
                delay = delayed_mask(commit, b["delta"], count, mask == "selective", seed + 2000)
                record = execute(b, commit, delay, delay_steps)
                name = f"{seed}-{control}-{mode}-{mask}-{count}"
                np.savez_compressed(output_dir / f"{name}.npz", **record)
                rows.append({"id": name, "seed": seed, "control": control, "mode": mode,
                             "mask": mask, "delay_count": count, **summarize(b, record)})
    index = {(r["seed"], r["control"], r["mode"], r["delay_count"], r["mask"]): r for r in rows}
    counterexamples = []
    for row in rows:
        if row["control"] == "mixture" and row["mask"] == "selective":
            partner = index[(row["seed"], row["control"], row["mode"], row["delay_count"], "uniform")]
            matched = row["commitment_sha256"] == partner["commitment_sha256"] and row["delayed_count_per_placement"] == partner["delayed_count_per_placement"]
            if matched and row["trace_audit_passed"] and partner["trace_audit_passed"] and row["v2_fit"]["type"] == "trade_off" and partner["v2_fit"]["type"] == "distance_insensitive":
                counterexamples.append({"selective": row["id"], "uniform": partner["id"]})
    positive = [r for r in rows if r["control"] != "mixture" and r["mode"] == "none"]
    summary = {"study": protocol_spec()["name"], "rows": len(rows),
               "all_trace_audits_passed": all(r["trace_audit_passed"] for r in rows),
               "counterexample_found": bool(counterexamples), "counterexamples": counterexamples,
               "no_delay_positive_controls": len(positive),
               "no_delay_positive_classifications": sum(r["v2_fit"]["type"] == "trade_off" for r in positive),
               "scope": protocol_spec()["interpretation"]}
    (output_dir / "records.json").write_text(json.dumps(_clean(rows), indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    return summary
