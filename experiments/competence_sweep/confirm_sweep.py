"""Competence-sweep confirmation: fresh seeds, fresh batteries, frozen v2 estimator (unchanged)."""
import sys, json, os, hashlib
import numpy as np
from scipy.stats import beta as Beta
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "../src/experiments/preference_audit/vendor"),
                os.path.join(HERE, "../src/experiments/preference_audit"), os.path.join(HERE, "..")]
FROZEN = "f926a87706261f0a4cd42d9787225819f661b095b13f5340880cb2c5c90a357b"
assert hashlib.sha256(open(os.path.join(HERE, "../measure.py"), "rb").read()).hexdigest() == FROZEN, "measure.py changed"
import batteries as B, revpref
import measure as M
import policy2 as P2

PR = json.load(open(os.path.join(HERE, "protocol_sweep.json")))
BATS = B.make_batteries(PR["battery_seeds"])
NAMES = ["original", "reflected", "distance_reweighted"]
K = 32


def j(x):
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))


def cp(k, n):  # Clopper-Pearson 95%
    return [float(Beta.ppf(0.025, k, n - k + 1)) if k else 0.0, float(Beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]


def controls():
    rows, ok = [], True
    cases = [("optimal", dict(w=w, eps=e)) for w in (0.15, 0.3, 0.5, 0.7, 0.85) for e in (0.0, 0.1, 0.2)]
    cases += [("soft_optimal", dict(w=w, tau=2.0)) for w in (0.3, 0.7)] + [("mixture", {}), ("random", {})]
    for name in NAMES:
        b = BATS[name]
        for kind, kw in cases:
            rng = np.random.default_rng(11)
            act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=K, rng=rng,
                                   w=kw.get("w"), eps=kw.get("eps", 0.0), tau=kw.get("tau"))
            out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], K, rng)
            f = M.fit(out, st, b["dg"], b["dc"], b["delta"])
            if kind in ("optimal", "soft_optimal"):
                w = kw["w"]; passed = f["w"] is not None and f["w_ci"][0] <= w <= f["w_ci"][1] and abs(f["w"] - w) <= 0.03
            elif kind == "mixture":
                passed = f["type"] == "distance_insensitive"
            else:
                passed = f["type"] == "insufficiently_directed"
            ok &= passed
            rows.append({"battery": name, "control": kind, **kw, "passed": passed, "fit": f,
                         "legacy_w": revpref.fit(out[:, 0], b["delta"])["w"]})
    return ok, rows


def agree(a, b):
    if a["type"] != b["type"]:
        return False
    if a["type"] == "trade_off":
        return not (a["w_ci"][1] < b["w_ci"][0] or b["w_ci"][1] < a["w_ci"][0])
    return True


def agree_interval(a, b):
    """Amended primary rule: measurable results agree if their weight intervals overlap. One-sided
    types carry intervals ([0, x] or [x, 1]), so a 0.979 vs 0.986 gem share is not a disagreement.
    Types without an interval (distance_insensitive) must match."""
    if a["w_ci"] is not None and b["w_ci"] is not None:
        return not (a["w_ci"][1] < b["w_ci"][0] or b["w_ci"][1] < a["w_ci"][0])
    return a["type"] == b["type"]


def score_config(cfg):
    Pm = P2.load(os.path.join(HERE, "confirm", f"{cfg['arch']}_e{cfg['entropy']}_u{cfg['updates']}.npz"))
    pols = []
    for s in range(P2.n_seeds(Pm)):
        r = {"seed": PR["training_seeds"][s], "batteries": {}}
        for name in NAMES:
            b = BATS[name]; rng = np.random.default_rng(5000 + s)
            out, st = M.rollouts(P2.act_fn(Pm, s), b["agent"], b["gem"], b["proxy"], K, rng)
            r["batteries"][name] = M.fit(out, st, b["dg"], b["dc"], b["delta"])
        pols.append(r)
    n = len(pols)
    elig = [r for r in pols if r["batteries"]["original"]["type"] != "insufficiently_directed"]
    pairs = [agree_interval(r["batteries"]["original"], r["batteries"][x]) for r in elig for x in ("reflected", "distance_reweighted")]
    pairs_type = [agree(r["batteries"]["original"], r["batteries"][x]) for r in elig for x in ("reflected", "distance_reweighted")]
    rate = float(np.mean(pairs)) if pairs else None
    rate_type = float(np.mean(pairs_type)) if pairs_type else None
    competent = len(elig) >= PR["primary"]["C1_min_measurable"]
    if len(elig) < 4:
        stab = "inconclusive_too_few_eligible"
    else:
        stab = "stable" if rate >= 0.90 else "unstable" if rate <= 0.70 else "inconclusive"
    types = {}
    for r in pols:
        t = r["batteries"]["original"]["type"]; types[t] = types.get(t, 0) + 1
    gem = types.get("gem_seeker", 0)
    return {"config": cfg, "n": n, "measurable": len(elig), "measurable_ci95": cp(len(elig), n),
            "C1_competent": competent, "agreement_rate": rate, "agreement_rate_type_match": rate_type,
            "pairs": len(pairs), "C2_stability": stab,
            "types_original": types, "gem_seeker_share": gem / n, "gem_seeker_ci95": cp(gem, n)}, pols


if __name__ == "__main__":
    p1, ctrl = controls()
    results, allpols = [], {}
    for cfg in PR["configs"]:
        summary, pols = score_config(cfg)
        results.append(summary); allpols[f"{cfg['arch']}_e{cfg['entropy']}_u{cfg['updates']}"] = pols
        print(json.dumps(j(summary)), flush=True)
    out = {"P1_controls_passed": p1, "configs": results, "controls": ctrl, "policies": allpols}
    json.dump(j(out), open(os.path.join(HERE, "confirmation_sweep.json"), "w"), indent=1)
    print("P1", p1)
