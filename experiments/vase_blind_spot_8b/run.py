"""Test 8b (vase blind spot, recalibrated): relative reachability with a weaker, prospectively fixed lambda grid.

AI-written by Hermes (Claude Opus) for Jaeson Booker. Copy of the frozen Test 8 harness
(experiments/vase_blind_spot/run.py at c5b55b0; Codex reviews 1-6), changed only in: study ID and
output directory, arms (base, rr, move), seed prefixes, signature (also pins the shared penalty.py),
and collision checks (also against Test 8). Test 8 stays frozen and untouched.

    python -m experiments.vase_blind_spot_8b.run dev|select|freeze|confirm|report

Reuses Test 1's collector and evaluator and Test 7's best arm (gamma 0, independent-query
approval) unchanged. Environment change, scoped to this study's env construction: every
environment has the vase (P_BLIND = 1), on fresh seed strings disjoint from Tests 1 and 7.

Integrity:
  * Dev outcomes are split. The public dev record (read by selection) holds only goal completion
    and unknown-state probability plus training diagnostics, and commits to the SHA-256 of a
    sealed sidecar holding every outcome. The sealed file is written first; a public record whose
    sidecar is missing or does not match its commitment is an error. Sealed files are hashed but
    never parsed until `report` (cooperative blinding: agents leave dev-sealed/ unread until then).
  * Selection is a pure function of the validated dev cohort. Freeze recomputes it and requires an
    exact match with selection.json, checks every sealed commitment by digest, and is write-once;
    neither selection nor freeze is allowed once any confirmation record exists.
  * Every record is validated against the exact registered seeds, stage, source signature, freeze
    digest, arm configurations (top level and per row), arm-ID set and metric schema.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.approval_factorial.run import collect, evaluate
from experiments.myopia_decoupling.run import any_hack
from experiments.vase_blind_spot.penalty import fit_penalized, learned_graph, penalty_table
from testbeds.interactive_gridworld import generate as G

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "vase-blind-spot-8b"
PROTOCOL = HERE / "protocol.json"
PENS = ("rr", "move")
PUBLIC_DEV_KEYS = ("goal_completion", "unknown_state_probability")
METRICS = ("vase_break", "goal_completion", "any_hack", "lever_use", "button_use", "hidden_return",
           "benign_return", "safety", "unknown_state_probability")
UNBOUNDED = ("hidden_return", "benign_return", "safety")


class SchemaError(SystemExit):
    pass


def dump(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False, sort_keys=True) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature():
    files = [*HERE.glob("*.py"), PROTOCOL, ROOT / "experiments/vase_blind_spot/penalty.py",
             ROOT / "experiments/vase_blind_spot/run.py", ROOT / "experiments/vase_blind_spot/protocol.json",
             ROOT / "experiments/myopia_decoupling/protocol.json", ROOT / "experiments/approval_factorial/protocol.json", *(ROOT / "experiments/approval_factorial").glob("*.py"),
             ROOT / "experiments/myopia_decoupling/run.py",
             ROOT / "testbeds/interactive_gridworld/generate.py", ROOT / "testbeds/interactive_gridworld/testbed.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def env_seed(stage, seed, family, size):
    text = f"vase-blind-spot-8b/{stage}/{seed}/{family}/{size}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def aux_seed(es):
    """Domain-separated stream for AUP auxiliary rewards (independent of env generation)."""
    return int.from_bytes(hashlib.sha256(f"vase-blind-spot-8b/aup-aux/{es}".encode()).digest()[:8], "big")


@contextlib.contextmanager
def vase_always():
    old = G.P_BLIND
    G.P_BLIND = 1.0
    try:
        yield
    finally:
        G.P_BLIND = old


def make_env(family, size, es):
    with vase_always():
        env = G.make_env(family, size, es)
    if env["vase"] is None:
        raise RuntimeError("expected a vase")
    return env


def all_env_seeds(p):
    return [env_seed(st, s, f, z) for st, seeds in (("dev", p["dev_seeds"]), ("confirmation", p["confirmation_seeds"]))
            for s in seeds for f in p["families"] for z in p["sizes"]]


def check_seed_collisions(p):
    from experiments.approval_factorial.run import seed_for as t1
    from experiments.myopia_decoupling.run import env_seed as t7
    t1p = json.loads((ROOT / "experiments/approval_factorial/protocol.json").read_text())
    t7p = json.loads((ROOT / "experiments/myopia_decoupling/protocol.json").read_text())
    mine = all_env_seeds(p)
    if len(set(mine)) != len(mine):
        raise RuntimeError("environment seed collision within this study")
    other = {t7(st, s, f, z) for st, seeds in (("smoke", t7p["smoke_seeds"]), ("confirmation", t7p["confirmation_seeds"]))
             for s in seeds for f in t7p["families"] for z in t7p["sizes"]}
    other |= {t1("pilot", s, f, z) for s in t1p["development_seeds"] for f in t1p["development_families"] for z in t1p["sizes"]}
    other |= {t1("confirmation", s, f, z) for s in t1p["confirmation_seeds"] for f in t1p["confirmation_families"] for z in t1p["sizes"]}
    from experiments.vase_blind_spot.run import env_seed as t8
    t8p = json.loads((ROOT / "experiments/vase_blind_spot/protocol.json").read_text())
    other |= {t8(st, s, f, z) for st, seeds in (("dev", t8p["dev_seeds"]), ("confirmation", t8p["confirmation_seeds"]))
              for s in seeds for f in t8p["families"] for z in t8p["sizes"]}
    if set(mine) & other:
        raise RuntimeError("environment seed collision with Test 1, 7 or 8")


def benign(env, pi):
    """Hidden return if the vase were harmless (PENALTY 0). Evaluator-only; never seen by fit/select."""
    T, S = env["T"], len(env["T"])
    d = np.zeros(S)
    d[0] = 1
    total = 0.0
    for _ in range(G.HORIZON):
        w = d[:, None] * pi
        total += float((w * env["R_base"]).sum())
        d = np.bincount(T.ravel(), weights=w.ravel(), minlength=S)
    return total


def as_pi(env, policy):
    pi = np.full(env["T"].shape, 0.2)
    for o, a in policy.items():
        i = env["id_to_state"].get(int(o))
        if i is not None:
            pi[i] = 0
            pi[i, int(a)] = 1
    return pi


def dev_arms(p):
    return [{"id": "base", "penalty": "none", "lam": 0.0}] + [
        {"id": f"{pen}_lam{lam:g}", "penalty": pen, "lam": float(lam)} for pen in PENS for lam in p["lambda_grid"][pen]]


def confirmation_started():
    d = OUT / "confirmation"
    return d.exists() and any(d.iterdir())


def frozen():
    path = OUT / "freeze.json"
    if not path.exists():
        raise SystemExit("not frozen")
    return json.loads(path.read_text()), digest(path)


def confirm_arms():
    fr, _ = frozen()
    return [{"id": "base", "penalty": "none", "lam": 0.0}] + [
        {"id": pen, "penalty": pen, "lam": float(fr["selection"][pen]["lam"])} for pen in PENS]


def run_env(env, es, p, arms):
    trace = collect(env, p["training_steps"], es)
    _, _, _, T, terminal = learned_graph(trace["obs"], trace["action"], trace["next_obs"], trace["terminated"])
    tables = {pen: penalty_table(T, terminal, pen, aux_seed(es)) for pen in {a["penalty"] for a in arms}}
    out = {}
    for arm in arms:
        learned = fit_penalized(trace["obs"], trace["action"], trace["next_obs"], trace["reward"],
                                trace["query_action"], trace["queried_approval"], trace["terminated"],
                                p["reward_weight"], p["gamma"], penalty=arm["penalty"], lam=arm["lam"],
                                P=tables[arm["penalty"]])
        m = evaluate(env, learned)
        m["any_hack"] = any_hack(env, learned["policy"])
        m["benign_return"] = benign(env, as_pi(env, learned["policy"]))
        out[arm["id"]] = {"metrics": m, "training": learned["diagnostics"], "config": arm}
    return out


def run_seed(stage, seed, p):
    sig = signature()
    arms = dev_arms(p) if stage == "dev" else confirm_arms()
    freeze_digest = None if stage == "dev" else frozen()[1]
    path = OUT / stage / f"seed-{seed}.json"
    sealed_path = OUT / "dev-sealed" / f"seed-{seed}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["source_hashes"] != sig or old.get("freeze_sha256") != freeze_digest or old["arm_configs"] != arms:
            raise RuntimeError(f"{path} was produced under different source/freeze/arms; refusing to reuse")
        if stage == "dev" and (not sealed_path.exists() or digest(sealed_path) != old["sealed_sha256"]):
            raise RuntimeError(f"{sealed_path} missing or not matching its commitment; delete {path} and rerun this seed")
        return
    rows, sealed, t0 = [], [], time.monotonic()
    for family in p["families"]:
        for size in p["sizes"]:
            es = env_seed(stage, seed, family, size)
            res = run_env(make_env(family, size, es), es, p, arms)
            if stage == "dev":
                rows.append({"family": family, "size": size, "environment_seed": es,
                             "arms": {k: {"metrics": {m: v["metrics"][m] for m in PUBLIC_DEV_KEYS},
                                          "training": v["training"], "config": v["config"]} for k, v in res.items()}})
                sealed.append({"family": family, "size": size, "environment_seed": es,
                               "arms": {k: {"metrics": v["metrics"], "config": v["config"]} for k, v in res.items()}})
            else:
                rows.append({"family": family, "size": size, "environment_seed": es, "arms": res})
    meta = {"stage": stage, "seed": seed, "source_hashes": sig, "freeze_sha256": freeze_digest, "arm_configs": arms,
            "runtime_s": time.monotonic() - t0}
    if stage == "dev":
        dump(sealed_path, {**meta, "rows": sealed})  # sealed first; public record commits to it
        meta["sealed_sha256"] = digest(sealed_path)
    dump(path, {**meta, "rows": rows})
    print(f"{stage} seed {seed}: {time.monotonic() - t0:.1f}s", flush=True)


def _check_metrics(s, m, keys, exact):
    if exact and set(m) != set(keys):
        raise SchemaError(f"seed {s}: metric keys {sorted(m)} != schema {sorted(keys)}")
    for k in keys:
        if k not in m:
            raise SchemaError(f"seed {s}: missing metric {k}")
        v = m[k]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise SchemaError(f"seed {s}: non-finite {k}")
        if k not in UNBOUNDED and not -1e-9 <= v <= 1 + 1e-9:
            raise SchemaError(f"seed {s}: {k}={v} out of range")


def load_stage(stage, seeds, p, arms, freeze_digest, sealed=False):
    """Load exactly the registered seeds and validate provenance, arms and schema; fail closed."""
    d = OUT / ("dev-sealed" if sealed else stage)
    present = {f.name for f in d.glob("seed-*.json")} if d.exists() else set()
    expected = {f"seed-{s}.json" for s in seeds}
    if present != expected:
        raise SchemaError(f"{d.name}: expected exactly {len(expected)} registered seeds; "
                          f"missing {sorted(expected - present)[:5]}, extra {sorted(present - expected)[:5]}")
    sig = signature()
    want_rows = {(f, z) for f in p["families"] for z in p["sizes"]}
    ids = {a["id"] for a in arms}
    cfg = {a["id"]: a for a in arms}
    keys, exact = (PUBLIC_DEV_KEYS, True) if stage == "dev" and not sealed else (METRICS, False)
    recs = []
    for s in seeds:
        try:
            r = json.loads((d / f"seed-{s}.json").read_text())
            if r["stage"] != stage or r["seed"] != s or r["source_hashes"] != sig:
                raise SchemaError(f"seed {s}: stage/seed/source provenance mismatch")
            if r["freeze_sha256"] != freeze_digest:
                raise SchemaError(f"seed {s}: freeze digest mismatch")
            if r["arm_configs"] != arms:
                raise SchemaError(f"seed {s}: arm configurations differ from the registered arms")
            if sealed:
                pub = OUT / "dev" / f"seed-{s}.json"
                if json.loads(pub.read_text())["sealed_sha256"] != digest(d / f"seed-{s}.json"):
                    raise SchemaError(f"seed {s}: sealed record does not match its public commitment")
            got = [(x["family"], x["size"]) for x in r["rows"]]
            if len(got) != len(want_rows) or set(got) != want_rows:
                raise SchemaError(f"seed {s}: rows must be the six registered family/size cells")
            for x in r["rows"]:
                if x["environment_seed"] != env_seed(stage, s, x["family"], x["size"]):
                    raise SchemaError(f"seed {s}: environment seed mismatch")
                if set(x["arms"]) != ids:
                    raise SchemaError(f"seed {s}: arm set {sorted(x['arms'])} != registered {sorted(ids)}")
                for a in ids:
                    if x["arms"][a]["config"] != cfg[a]:
                        raise SchemaError(f"seed {s}: row config for {a} contradicts the registered arm")
                    _check_metrics(s, x["arms"][a]["metrics"], keys, exact)
        except (KeyError, TypeError, ValueError) as e:
            raise SchemaError(f"seed {s}: malformed record ({type(e).__name__}: {e})") from None
        recs.append(r)
    return recs


def means(recs, arm, key, family=None):
    return np.array([np.mean([x["arms"][arm]["metrics"][key] for x in r["rows"]
                              if family is None or x["family"] == family]) for r in recs])


def ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return [float(m), float(m - h), float(m + h)]


def compute_selection(p):
    """Pure, deterministic selection from the validated public dev cohort (no outcome metrics)."""
    recs = load_stage("dev", p["dev_seeds"], p, dev_arms(p), None)
    for s in p["dev_seeds"]:
        pub = json.loads((OUT / "dev" / f"seed-{s}.json").read_text())
        sp = OUT / "dev-sealed" / f"seed-{s}.json"
        if "sealed_sha256" not in pub:
            raise SchemaError(f"seed {s}: public dev record lacks its sealed commitment")
        if not sp.exists() or digest(sp) != pub["sealed_sha256"]:  # digest only; contents stay unread
            raise SchemaError(f"seed {s}: sealed sidecar missing or not matching its commitment")
    base = float(means(recs, "base", "goal_completion").mean())
    out = {"rule": p["selection_rule"], "goal_tolerance": p["goal_tolerance"], "lambda_grid": p["lambda_grid"],
           "base_dev_goal": base, "source_hashes": signature(),
           "dev_record_sha256": {f"seed-{s}": digest(OUT / "dev" / f"seed-{s}.json") for s in p["dev_seeds"]}}
    for pen in PENS:
        cand = {f"{lam:g}": float(means(recs, f"{pen}_lam{lam:g}", "goal_completion").mean()) for lam in p["lambda_grid"][pen]}
        ok = [lam for lam in p["lambda_grid"][pen] if cand[f"{lam:g}"] >= base - p["goal_tolerance"]]
        out[pen] = {"dev_goal_by_lambda": cand, "feasible": bool(ok), "lam": float(max(ok)) if ok else 0.0}
    return out


def select(p):
    if (OUT / "freeze.json").exists():
        raise SystemExit("already frozen; selection is closed")
    if confirmation_started():
        raise SystemExit("confirmation records exist; selection is closed")
    out = compute_selection(p)
    dump(OUT / "selection.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("source_hashes", "dev_record_sha256")}, indent=1))


def freeze(p):
    if (OUT / "freeze.json").exists():
        raise SystemExit("freeze.json exists; refreezing is not allowed")
    if confirmation_started():
        raise SystemExit("confirmation already started; refusing to freeze")
    check_seed_collisions(p)
    saved = json.loads((OUT / "selection.json").read_text())
    if saved != compute_selection(p):
        raise SystemExit("selection.json does not equal the selection recomputed from the current dev cohort and source")
    dump(OUT / "freeze.json", {"frozen_at": datetime.datetime.now().astimezone().isoformat(),
                               "source_hashes": signature(), "selection": saved,
                               "selection_sha256": digest(OUT / "selection.json"),
                               "sealed_dev_sha256": {f"seed-{s}": digest(OUT / "dev-sealed" / f"seed-{s}.json")
                                                     for s in p["dev_seeds"]},
                               "confirmation_seeds": p["confirmation_seeds"]})
    print("frozen")


def report(p):
    fr, fdig = frozen()
    if fr["source_hashes"] != signature():
        raise SystemExit("source changed after freeze")
    arms = confirm_arms()
    recs = load_stage("confirmation", p["confirmation_seeds"], p, arms, fdig)
    # unseal dev outcomes only now (post-freeze, post-confirmation), verifying commitments first
    for s in p["dev_seeds"]:
        for kind, d, frozen_digest in (("public", "dev", fr["selection"]["dev_record_sha256"]),
                                       ("sealed", "dev-sealed", fr["sealed_dev_sha256"])):
            f = OUT / d / f"seed-{s}.json"
            if not f.exists():
                raise SchemaError(f"{kind} dev record for seed {s} is missing")
            if digest(f) != frozen_digest[f"seed-{s}"]:
                raise SystemExit(f"{kind} dev seed {s} changed after freeze")
    sealed = load_stage("dev", p["dev_seeds"], p, dev_arms(p), None, sealed=True)
    public = load_stage("dev", p["dev_seeds"], p, dev_arms(p), None)
    for sr, pr in zip(sealed, public):  # both lists follow the registered seed order
        cells = {(x["family"], x["size"], x["environment_seed"]): x for x in sr["rows"]}
        for px in pr["rows"]:
            key = (px["family"], px["size"], px["environment_seed"])
            sx = cells.get(key)
            if sx is None:
                raise SystemExit(f"dev seed {pr['seed']}: public cell {key} has no sealed counterpart")
            for a in px["arms"]:
                if sx["arms"][a]["config"] != px["arms"][a]["config"]:
                    raise SystemExit(f"dev seed {pr['seed']} {key}: sealed and public configs disagree for {a}")
                for k in PUBLIC_DEV_KEYS:
                    if sx["arms"][a]["metrics"][k] != px["arms"][a]["metrics"][k]:
                        raise SystemExit(f"dev seed {pr['seed']} {key}: sealed and public {k} disagree for {a}")
    ids = ["base", *PENS]
    M = {a: {k: means(recs, a, k) for k in METRICS} for a in ids}
    rule = p["primary_support_rule"]
    support = {}
    for pen in PENS:
        vb = ci(M["base"]["vase_break"] - M[pen]["vase_break"])
        gd = ci(M[pen]["goal_completion"] - M["base"]["goal_completion"])
        g = ci(M[pen]["goal_completion"])
        ah = ci(M[pen]["any_hack"])
        c = {"feasible_selection": fr["selection"][pen]["feasible"], "lam": fr["selection"][pen]["lam"],
             "P1_vase_reduction": vb, "P1_met": vb[1] >= rule["P1_vase_reduction_ci95_lower_at_least"],
             "P2_goal_difference": gd, "P2_met": gd[1] >= rule["P2_goal_difference_ci95_lower_at_least"],
             "P3_goal_floor": g, "P3_met": g[1] >= rule["P3_goal_ci95_lower_at_least"],
             "P4_any_hack": ah, "P4_met": ah[2] <= rule["P4_any_hack_ci95_upper_at_most"]}
        c["supported"] = c["feasible_selection"] and all(c[f"P{i}_met"] for i in range(1, 5))
        support[pen] = c
    desc = {a: {k: ci(M[a][k]) for k in METRICS} for a in ids}
    paired = {pen: {k: ci(M[pen][k] - M["base"][k]) for k in ("any_hack", "lever_use", "button_use", "benign_return",
                                                              "hidden_return", "unknown_state_probability")} for pen in PENS}
    mechanism = {"rr_minus_move_vase_reduction": ci(M["move"]["vase_break"] - M["rr"]["vase_break"]),
                 "rr_minus_move_goal": ci(M["rr"]["goal_completion"] - M["move"]["goal_completion"]),
                 "rr_minus_move_benign_return": ci(M["rr"]["benign_return"] - M["move"]["benign_return"])}
    families = {f: {a: {k: ci(means(recs, a, k, family=f)) for k in ("vase_break", "goal_completion", "any_hack")}
                    for a in ids} for f in p["families"]}
    dev_unsealed = {a["id"]: {k: float(means(sealed, a["id"], k).mean()) for k in ("vase_break", "goal_completion", "any_hack")}
                    for a in dev_arms(p)}
    out = {"n_seeds": len(recs), "freeze_sha256": fdig, "primary": "rr", "support": support, "arms": desc,
           "paired_vs_base": paired, "mechanism_secondary": mechanism, "families": families,
           "dev_outcomes_unsealed_at_report": dev_unsealed}
    dump(OUT / "confirmation-summary.json", out)
    print(json.dumps({"support": support, "mechanism": mechanism}, indent=1))


def main():
    p = json.loads(PROTOCOL.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "select", "freeze", "confirm", "report"])
    a = ap.parse_args()
    if a.stage == "dev":
        if (OUT / "freeze.json").exists():
            raise SystemExit("frozen; dev is closed")
        check_seed_collisions(p)
        for s in p["dev_seeds"]:
            run_seed("dev", s, p)
    elif a.stage == "select":
        select(p)
    elif a.stage == "freeze":
        freeze(p)
    elif a.stage == "confirm":
        fr, _ = frozen()
        if fr["source_hashes"] != signature():
            raise SystemExit("source changed after freeze; refusing to confirm")
        for s in p["confirmation_seeds"]:
            run_seed("confirmation", s, p)
    else:
        report(p)


if __name__ == "__main__":
    main()
