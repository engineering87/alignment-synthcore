"""Frozen analysis for escape-factorial-v1: censored (Tobit) models M0/M1/M2 of log escape time."""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze import seed_series, escape
PR = json.load(open(os.path.join(HERE, "protocol_fact.json")))
CELLS = [(a, e) for a in PR["factors"]["width"] for e in sorted(PR["factors"]["entropy"])]


def rows():
    out = []
    for a, e in CELLS:
        for p in PR["factors"]["p"]:
            path = os.path.join(HERE, "scores", f"{a}_p{p}_e{e}.json")
            if not os.path.exists(path):
                continue
            us, gs = seed_series(json.load(open(path)))
            for s, g in enumerate(gs):
                T, c, back = escape(us, g, max(us))
                out.append({"arch": a, "entropy": e, "p": p, "seed": PR["seeds"][s], "T": T, "censored": c,
                            "switch_back": back, "pT": p * T, "last_checkpoint": max(us)})
    return out


def design(R, model):
    cell = np.array([CELLS.index((r["arch"], r["entropy"])) for r in R]); x = np.log(1 / np.array([r["p"] for r in R]))
    n, C = len(R), len(CELLS)
    if model == "M0":
        X = np.column_stack([np.ones(n), x])
    elif model == "M1":
        X = np.column_stack([np.eye(C)[cell], x])
    else:
        X = np.column_stack([np.eye(C)[cell], np.eye(C)[cell] * x[:, None]])
    return X


def fit(X, y, c, fix=None):
    def nll(th):
        beta, s = th[:-1], np.exp(th[-1])
        if fix is not None:
            beta = beta.copy(); beta[fix[0]] = fix[1]
        mu = X @ beta
        return -np.where(c, norm.logsf((y - mu) / s), norm.logpdf((y - mu) / s) - th[-1]).sum()
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    th0 = np.append(beta0, np.log(max(np.std(y - X @ beta0), 0.1)))
    r = minimize(nll, th0, method="BFGS")
    r = minimize(nll, r.x, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-9, "fatol": 1e-10})
    return r.x, r.fun


def profile_ci(X, y, c, j, th, best):
    grid = np.linspace(th[j] - 1.5, th[j] + 1.5, 121); keep = []
    for v in grid:
        _, f = fit(X, y, c, fix=(j, v))
        if f <= best + chi2.ppf(0.95, 1) / 2:
            keep.append(v)
    return [float(min(keep)), float(max(keep))]


if __name__ == "__main__":
    R = rows(); y = np.log([r["T"] for r in R]); c = np.array([r["censored"] for r in R])
    fits = {m: fit(design(R, m), y, c) for m in ("M0", "M1", "M2")}
    X1 = design(R, "M1"); th1, f1 = fits["M1"]; C = len(CELLS)
    b_ci = profile_ci(X1, y, c, C, th1, f1)
    h1 = "supported" if b_ci[0] >= 0.75 and b_ci[1] <= 1.25 else "not_supported"
    lr2 = 2 * (f1 - fits["M2"][1]); p2 = float(chi2.sf(max(lr2, 0), C - 1))
    lr3 = 2 * (fits["M0"][1] - f1); p3 = float(chi2.sf(max(lr3, 0), C - 1))
    consts = []
    for k, (a, e) in enumerate(CELLS):
        ci = profile_ci(X1, y, c, k, th1, f1)
        cr = [r for r in R if (r["arch"], r["entropy"]) == (a, e)]
        consts.append({"arch": a, "entropy": e, "log_intercept": float(th1[k]), "T_at_p1": float(np.exp(th1[k])),
                       "T_at_p1_ci95": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                       "median_pT": float(np.median([r["pT"] for r in cr])) if cr else None,
                       "censored": int(sum(r["censored"] for r in cr)), "n": len(cr)})
    out = {"n": len(R), "H1": {"b": float(th1[C]), "b_ci95": b_ci, "sigma": float(np.exp(th1[-1])), "verdict": h1},
           "H2": {"LR": float(lr2), "df": C - 1, "p": p2, "verdict": "supported" if p2 >= 0.05 else "exponent_varies"},
           "H3": {"LR": float(lr3), "df": C - 1, "p": p3, "verdict": "supported" if p3 < 0.05 else "not_supported"},
           "constants": consts, "switch_backs": int(sum(r["switch_back"] for r in R)), "rows": R}
    json.dump(out, open(os.path.join(HERE, "factorial_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
