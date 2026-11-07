"""Escape-time analysis. Escape = first checkpoint where a seed reaches the gem on >= 50% of battery trials
(sampled, all trials in the denominator). Seeds that never cross by the budget are right-censored.
Fits log T = a + b log(1/p) + N(0, s^2) by censored maximum likelihood (Tobit); 1/p scaling means b = 1.
Also reports switch-backs: a seed whose gem rate falls back below 0.5 for >= 2 consecutive checkpoints
after escaping."""
import json, sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

THRESH = 0.5


def seed_series(cache):
    us = sorted(cache, key=int); n = len(cache[us[0]])
    return [int(u) for u in us], [[cache[u][s]["gem_rate"] for u in us] for s in range(n)]


def escape(us, g, cap):
    for i, x in enumerate(g):
        if x >= THRESH:
            back = any(g[j] < THRESH and g[j + 1] < THRESH for j in range(i + 1, len(g) - 1))
            return us[i], False, back
    return cap, True, False


def tobit(x, logT, cens):
    def nll(th):
        a, b, ls = th; s = np.exp(ls); mu = a + b * x
        ll = np.where(cens, norm.logsf((logT - mu) / s), norm.logpdf((logT - mu) / s) - ls)
        return -ll.sum()
    r = minimize(nll, [0.0, 1.0, np.log(0.3)], method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
    a, b, ls = r.x
    # profile-likelihood 95% interval for b
    grid = np.linspace(b - 2, b + 2, 801); prof = []
    for bb in grid:
        rr = minimize(lambda t: nll([t[0], bb, t[1]]), [a, ls], method="Nelder-Mead"); prof.append(rr.fun)
    prof = np.array(prof); inside = grid[prof <= r.fun + 1.92]
    return {"a": a, "b": b, "sigma": float(np.exp(ls)), "b_ci95": [float(inside.min()), float(inside.max())], "nll": float(r.fun)}


def collect(runs):
    """runs: list of (p, cap, cache_path). Returns per-seed rows."""
    rows = []
    for p, cap, path in runs:
        us, gs = seed_series(json.load(open(path)))
        for s, g in enumerate(gs):
            T, c, back = escape(us, g, max(us) if us else cap)
            rows.append({"p": p, "seed_index": s, "T": T, "censored": c or (max(us) < cap and c), "switch_back": back,
                         "last_checkpoint": max(us)})
    return rows


if __name__ == "__main__":
    runs = [(float(a), int(b), c) for a, b, c in (x.split(":") for x in sys.argv[1:])]
    rows = collect(runs)
    for r in rows:
        print(r)
    x = np.log(1 / np.array([r["p"] for r in rows])); logT = np.log([r["T"] for r in rows]); cens = np.array([r["censored"] for r in rows])
    if len(set(x)) >= 2:
        print(json.dumps(tobit(x, logT, cens), indent=1))
