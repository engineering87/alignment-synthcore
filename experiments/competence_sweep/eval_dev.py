"""Score every development checkpoint with the frozen v2 estimator on development batteries."""
import sys, json, glob, os, re, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "../src/experiments/preference_audit/vendor"),
                os.path.join(HERE, "../src/experiments/preference_audit"), os.path.join(HERE, "..")]
FROZEN = "f926a87706261f0a4cd42d9787225819f661b095b13f5340880cb2c5c90a357b"
assert hashlib.sha256(open(os.path.join(HERE, "../measure.py"), "rb").read()).hexdigest() == FROZEN, "measure.py changed"
import batteries as B
import measure as M
import policy2 as P2

DEV_SEEDS = json.load(open(os.path.join(HERE, "../src/experiments/preference_audit/protocol.json")))["battery_seeds"]["development"]
BATS = B.make_batteries(DEV_SEEDS)
K = 32


def score(path, names=("original",)):
    Pm = P2.load(path); rows = []
    for s in range(P2.n_seeds(Pm)):
        r = {}
        for name in names:
            b = BATS[name]; rng = np.random.default_rng(3000 + s)
            out, st = M.rollouts(P2.act_fn(Pm, s), b["agent"], b["gem"], b["proxy"], K, rng)
            r[name] = M.fit(out, st, b["dg"], b["dc"], b["delta"])
        rows.append(r)
    return rows


if __name__ == "__main__":
    cache = os.path.join(HERE, "dev_scores.json")
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    for path in sorted(glob.glob(os.path.join(HERE, "dev/*.npz"))):
        key = os.path.basename(path)[:-4]
        if key in done:
            continue
        done[key] = score(path, ("original", "reflected", "distance_reweighted"))
        json.dump(done, open(cache, "w"))
        o = [r["original"] for r in done[key]]
        print(key, "measurable", sum(f["type"] != "insufficiently_directed" for f in o), "/", len(o),
              "| cov", [round(f["directed_coverage"], 2) for f in o], "| types", [f["type"][:6] for f in o],
              "| gem", [round(f["gem_rate"], 2) for f in o], flush=True)
