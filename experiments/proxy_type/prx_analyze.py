"""Frozen analysis for proxy-type-v1."""
import json, os, sys
import numpy as np
from scipy.stats import beta
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze import THRESH
from scipy.optimize import minimize
from scipy.stats import norm


def tobit(x, logT, rcens, lcens):
    """Normal model for log T = a + b x, right-censored at the last checkpoint and left-censored at the first
    (an escape at the first checkpoint means T <= that checkpoint). Profile-likelihood 95% CI for b."""
    def nll(th):
        a, b, ls = th; s = np.exp(ls); z = (logT - a - b * x) / s
        ll = np.where(rcens, norm.logsf(z), np.where(lcens, norm.logcdf(z), norm.logpdf(z) - ls))
        return -ll.sum()
    best = min((minimize(nll, [a0, 1.0, np.log(0.5)], method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-8, "fatol": 1e-10})
                for a0 in (2.0, 4.0, 6.0)), key=lambda r: r.fun)
    a, b, ls = best.x
    grid = np.linspace(b - 4, b + 4, 1601); prof = np.array([min(minimize(lambda t: nll([t[0], bb, t[1]]), [a1, ls], method="Nelder-Mead").fun
                                                              for a1 in (a, a - (bb - b) * x.mean())) for bb in grid])
    inside = grid[prof <= best.fun + 1.92]
    return {"a": float(a), "b": float(b), "sigma": float(np.exp(ls)), "b_ci95": [float(inside.min()), float(inside.max())], "nll": float(best.fun)}
PR = json.load(open(os.path.join(HERE, "protocol_prx.json")))


def cp(k, n):
    return [float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def series(tag):
    c = json.load(open(os.path.join(HERE, "scores", tag + ".json"))); us = sorted(c, key=int)
    return [int(u) for u in us], [[c[u][s] for u in us] for s in range(len(c[us[0]]))]


rows, final = [], {}
for cell in PR["cells"]:
    tag = f"{cell['env']}_p{cell['p']}"; us, S = series(tag)
    for s, ser in enumerate(S):
        g = [x["gem_rate"] for x in ser]; pr = [x["proxy_rate"] for x in ser]
        i = next((k for k, x in enumerate(g) if x >= THRESH), None)
        T, cens = (us[i], False) if i is not None else (us[-1], True); left = i == 0
        upto = len(g) if i is None else i
        lock = any(x >= THRESH for x in pr[:upto])
        back = i is not None and any(g[j] < THRESH and g[j + 1] < THRESH for j in range(i + 1, len(g) - 1))
        last = ser[-1]
        rows.append({"env": cell["env"], "p": cell["p"], "seed": PR["seeds"][s], "T": T, "censored": cens, "left_censored": left, "lock_in": lock,
                     "switch_back": back, "final": {k: last[k] for k in ("type", "w", "w_ci", "gem_rate", "proxy_rate", "directed_coverage")},
                     "final_updates": us[-1]})
m = [r for r in rows if r["env"] == "marker" and r["p"] > 0]
k = sum(r["lock_in"] for r in m); lo, hi = cp(k, len(m))
h1 = {"lock_in": k, "n": len(m), "ci95": [lo, hi], "verdict": "supported" if hi <= 0.25 else "not_supported"}
x = np.log(1 / np.array([r["p"] for r in m])); y = np.log([r["T"] for r in m]); c = np.array([r["censored"] for r in m])
fit = tobit(x, y, c, np.array([r["left_censored"] for r in m])); h2 = {**fit, "verdict": "supported" if fit["b_ci95"][1] < 0.75 else "not_supported"}
cells = []
for cell in PR["cells"]:
    rs = [r for r in rows if r["env"] == cell["env"] and r["p"] == cell["p"]]
    types = {}
    for r in rs:
        types[r["final"]["type"]] = types.get(r["final"]["type"], 0) + 1
    cells.append({"env": cell["env"], "p": cell["p"], "median_T": float(np.median([r["T"] for r in rs])), "censored": sum(r["censored"] for r in rs),
                  "lock_in": sum(r["lock_in"] for r in rs), "final_types": types,
                  "final_w": [r["final"]["w"] for r in rs], "median_pT": float(np.median([r["T"] * cell["p"] for r in rs]))})
out = {"H1": h1, "H2": h2, "cells": cells, "switch_backs": int(sum(r["switch_back"] for r in rows)), "rows": rows}
json.dump(out, open(os.path.join(HERE, "proxy_type_results.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))
