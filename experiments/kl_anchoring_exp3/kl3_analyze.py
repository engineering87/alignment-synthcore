"""Frozen analysis for kl-anchoring-exp3-confirm-v1. Reward-induced drift in a state group g at checkpoint t:
D_g(t) = mean TV(pi_t, pi_ref) over g under k1rew minus the same under klonly (same unit, same beta)."""
import json, os, sys, glob
import numpy as np
from scipy.stats import t as tdist
HERE = os.path.dirname(os.path.abspath(__file__))
PR = json.load(open(os.path.join(HERE, "protocol_kl3.json")))
NETS = [a for a in PR["archs"] if a != "tabular"]
EARLY = (25, 50, 100)


def lower95(x):
    x = np.asarray(x, float); n = len(x)
    return float(x.mean() - tdist.ppf(0.95, n - 1) * x.std(ddof=1) / np.sqrt(n))


def run(u, m, b):
    return next(r for r in u["runs"] if r["method"] == m and r["beta"] == b)["log"]


def tv(log, it, g):
    return next(l for l in log if l["it"] == it)["tv"][g]


def D(u, b, it, g):
    return tv(run(u, "k1rew", b), it, g) - tv(run(u, "klonly", b), it, g)


def export_index(us, b):
    o = np.mean([D(u, b, 1500, "O_all") for u in us]); h = np.mean([D(u, b, 1500, "T_high") for u in us])
    return (o - h) / (o + h) if o + h > 0 else float("nan")


def main(d):
    U = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "u_*.json")))]
    rng = np.random.default_rng(0); res = {"per_arch": {}}
    for a in NETS:
        us = [u for u in U if u["arch"] == a]
        h1 = [D(u, 0.3, 1500, "O_all") - D(u, 0.3, 1500, "T_high") for u in us]
        h2 = [np.mean([D(u, 0.3, t, "O_all") - D(u, 0.3, t, "T_high") for t in EARLY]) for u in us]
        e3, e1 = export_index(us, 0.3), export_index(us, 1.0); boots = []
        for _ in range(10000):
            idx = rng.integers(len(us), size=len(us)); bs = [us[i] for i in idx]
            x1, x3 = export_index(bs, 1.0), export_index(bs, 0.3)
            boots.append(x1 - x3 if np.isfinite(x1) and np.isfinite(x3) else -np.inf)
        h3_lo = float(np.quantile(boots, 0.05))
        res["per_arch"][a] = {"n_units": len(us),
            "H1": {"mean": float(np.mean(h1)), "lower95": lower95(h1), "pass": lower95(h1) > 0},
            "H2": {"mean": float(np.mean(h2)), "lower95": lower95(h2), "pass": lower95(h2) > 0},
            "H3": {"E_beta0.3": e3, "E_beta1.0": e1, "diff": e1 - e3, "lower95": h3_lo, "pass": h3_lo > 0}}
        # secondary (pre-registered): replication of the original raw metric, and whether it is reward-specific
        raw = lambda u, m, b: np.log(tv(run(u, m, b), 1500, "O_all") / tv(run(u, m, b), 1500, "T_high"))
        s1 = [raw(u, "k1rew", 0.3) for u in us]; s2 = [raw(u, "k1rew", 0.3) - raw(u, "klonly", 0.3) for u in us]
        s3 = [D(u, 0.3, 1500, "T_low") - D(u, 0.3, 1500, "T_high") for u in us]
        res["per_arch"][a].update({"S1_raw_ratio_gt1": {"median_ratio": float(np.exp(np.median(s1))), "lower95_log": lower95(s1), "pass": lower95(s1) > 0},
                                   "S2_raw_ratio_k1rew_gt_klonly": {"mean_log_diff": float(np.mean(s2)), "lower95": lower95(s2), "pass": lower95(s2) > 0},
                                   "S3_Tlow_vs_Thigh": {"mean": float(np.mean(s3)), "lower95": lower95(s3), "pass": lower95(s3) > 0}})
    for h in ("H1", "H2", "H3"):
        k = sum(res["per_arch"][a][h]["pass"] for a in NETS)
        res[h] = {"archs_passing": k, "of": len(NETS), "verdict": "supported" if k >= 3 else "not_supported"}
    # descriptive
    desc = {}
    for a in PR["archs"]:
        us = [u for u in U if u["arch"] == a]; row = {}
        for b in (0.3, 1.0):
            for g in ("T_high", "T_mid", "T_low", "B_all", "O_all"):
                row[f"D_{g}_b{b}"] = {str(t): float(np.nanmean([D(u, b, t, g) if tv(run(u, 'k1rew', b), t, g) is not None else np.nan for u in us]))
                                      for t in (25, 50, 100, 200, 400, 800, 1500)}
        for m, b in (("k1rew", 0.3), ("k1rew", 1.0), ("none", 0.0), ("klonly", 0.3)):
            row[f"ratio_O_Thigh_{m}{b}"] = float(np.median([tv(run(u, m, b), 1500, "O_all") / max(tv(run(u, m, b), 1500, "T_high"), 1e-9) for u in us]))
            for k in "TBO":
                row[f"door_{k}_{m}{b}"] = float(np.mean([run(u, m, b)[-1]["door"][k] for u in us]))
        for k in "TBO":
            row[f"ref_door_{k}"] = float(np.mean([np.mean(u["ref_door"][k]) for u in us]))
        desc[a] = row
    res["descriptive"] = desc
    res["tabular_O_drift_max"] = float(max(abs(tv(r["log"], 1500, "O_all")) for u in U if u["arch"] == "tabular" for r in u["runs"])) if any(u["arch"] == "tabular" for u in U) else None
    json.dump(res, open(os.path.join(d, "kl3_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "descriptive"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
