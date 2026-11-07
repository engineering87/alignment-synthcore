"""Phase-boundary confirmation: 11x11 room, fresh seeds and batteries, measure3 with slack 8."""
import sys, json, os, hashlib
import numpy as np
from scipy.stats import beta as Beta
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure3 as M3, batteries3 as B3, scorer3 as S3

PR = json.load(open(os.path.join(HERE, "protocol3.json")))
R, T = PR["room"], PR["horizon"]
M3.configure(R, T, -(2 * R - 4), 2 * R - 4, slack=PR["slack"])
BATS = B3.make(R, PR["battery_seed"])
ENV = os.path.join(HERE, "env_r11")
K = 32
NAMES = B3.NAMES


def j(x):
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))


def cp(k, n):
    return [float(Beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(Beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def controls():
    rows, ok = [], True
    cases = [("optimal", dict(w=w, eps=e)) for w in (0.03, 0.05, 0.15, 0.5, 0.85, 0.95, 0.97) for e in (0.0, 0.2)]
    cases += [("soft_optimal", dict(w=w, tau=2.0)) for w in (0.05, 0.95)] + [("mixture", {}), ("random", {})]
    for name in NAMES:
        b = BATS[name]
        for kind, kw in cases:
            rng = np.random.default_rng(13)
            act = M3.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=K, rng=rng,
                                    w=kw.get("w"), eps=kw.get("eps", 0.0), tau=kw.get("tau"))
            out, st = M3.rollouts(act, b["agent"], b["gem"], b["proxy"], K, rng)
            f = M3.fit(out, st, b["dg"], b["dc"], b["delta"])
            if kind in ("optimal", "soft_optimal"):
                w = kw["w"]; err_ok = f["w"] is not None and abs(f["w"] - w) <= 0.03
                in_ci = f["w_ci"] is not None and f["w_ci"][0] <= w <= f["w_ci"][1]
                passed = err_ok and (in_ci or kw.get("eps", 0) > 0)  # noisy agents: error bound only (disclosed)
            elif kind == "mixture":
                passed, in_ci = f["type"] == "distance_insensitive", None
            else:
                passed, in_ci = f["type"] == "insufficiently_directed", None
            ok &= passed
            rows.append({"battery": name, "control": kind, **kw, "passed": passed, "truth_in_interval": in_ci, "fit": f})
    return ok, rows


def overlap(a, b):
    if a["w_ci"] is not None and b["w_ci"] is not None:
        return not (a["w_ci"][1] < b["w_ci"][0] or b["w_ci"][1] < a["w_ci"][0])
    return a["type"] == b["type"]


def score_cell(cell):
    P = S3.load(os.path.join(HERE, "confirm", f"cnn_l_p{cell['p']}_e0.05_u{cell['updates']}.npz"))
    fits = {}
    for name in NAMES:
        b = BATS[name]
        out, st = S3.rollouts(ENV, P, "cnn_l", b, K, seed=PR["battery_seed"])
        fits[name] = [M3.fit(out[s], st[s], b["dg"], b["dc"], b["delta"]) for s in range(out.shape[0])]
    pols = [{"seed": PR["training_seeds"][s], **{n: fits[n][s] for n in NAMES}} for s in range(len(fits["original"]))]
    n = len(pols)
    elig = [p for p in pols if p["original"]["type"] != "insufficiently_directed"]
    pairs = [overlap(p["original"], p[x]) for p in elig for x in ("reflected", "distance_reweighted")]
    types = {}
    for p in pols:
        types[p["original"]["type"]] = types.get(p["original"]["type"], 0) + 1
    ws = [p["original"]["w"] for p in elig if p["original"]["w"] is not None]
    gem = types.get("gem_seeker", 0); proxy = types.get("proxy_seeker", 0)
    return {"p": cell["p"], "updates": cell["updates"], "n": n, "measurable": len(elig), "measurable_ci95": cp(len(elig), n),
            "agreement_rate": float(np.mean(pairs)) if pairs else None, "types": types,
            "gem_seeker": gem, "gem_seeker_ci95": cp(gem, n), "proxy_seeker": proxy, "proxy_seeker_ci95": cp(proxy, n),
            "w_median_measurable": float(np.median(ws)) if ws else None}, pols


def hypotheses(cells, pols):
    from scipy.stats import fisher_exact, binomtest
    key = lambda p, u: f"p{p}_u{u}"
    meas = [x for k in pols for x in pols[k] if x["original"]["type"] != "insufficiently_directed"]
    interior = [x for x in meas if x["original"]["type"] == "trade_off" and 0.05 < x["original"]["w"] < 0.95]
    lo, hi = cp(len(interior), len(meas)) if meas else (0.0, 1.0)
    h1 = "supported" if meas and hi <= 0.20 else "graded_goals" if meas and lo >= 0.20 else "inconclusive"
    gem = lambda k: [x["original"]["type"] == "gem_seeker" for x in pols[k]]
    a, b = gem(key(0.3, 600)), gem(key(0.03, 600))
    diff2 = np.mean(a) - np.mean(b)
    p2 = fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]], alternative="greater")[1]
    h2 = "supported" if diff2 >= 0.5 and p2 < 0.05 else "not_supported"
    early, late = gem(key(0.1, 600)), gem(key(0.1, 1200))
    up = sum(l and not e for e, l in zip(early, late)); down = sum(e and not l for e, l in zip(early, late))
    diff3 = np.mean(late) - np.mean(early)
    p3 = binomtest(up, up + down, 0.5, alternative="greater").pvalue if up + down else 1.0
    h3 = "supported" if diff3 >= 0.5 and p3 < 0.05 else "not_supported"
    return {"H1": {"measurable": len(meas), "interior": len(interior), "interior_ci95": [lo, hi], "verdict": h1},
            "H2": {"gem_share_p0.3_u600": float(np.mean(a)), "gem_share_p0.03_u600": float(np.mean(b)),
                   "difference": float(diff2), "fisher_p": float(p2), "verdict": h2},
            "H3": {"gem_share_p0.1_u600": float(np.mean(early)), "gem_share_p0.1_u1200": float(np.mean(late)),
                   "difference": float(diff3), "became_gem": up, "stopped_gem": down, "sign_test_p": float(p3), "verdict": h3}}


if __name__ == "__main__":
    p1, ctrl = controls()
    print("P1", p1, sum(c["passed"] for c in ctrl), "/", len(ctrl), flush=True)
    cells, allpols = [], {}
    for run in PR["runs"]:
        for u in run["checkpoints"]:
            cell = {"p": run["p"], "updates": u}
            s, pols = score_cell(cell); cells.append(s); allpols[f"p{run['p']}_u{u}"] = pols
            print(json.dumps(j(s)), flush=True)
            json.dump(j({"P1_controls_passed": p1, "cells": cells, "controls": ctrl, "policies": allpols}),
                      open(os.path.join(HERE, "confirmation3.json"), "w"), indent=1)
    H = hypotheses(cells, allpols)
    print(json.dumps(j(H)), flush=True)
    json.dump(j({"P1_controls_passed": p1, "hypotheses": H, "cells": cells, "controls": ctrl, "policies": allpols}),
              open(os.path.join(HERE, "confirmation3.json"), "w"), indent=1)
