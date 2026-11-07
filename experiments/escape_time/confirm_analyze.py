"""Frozen confirmation analysis for escape-time-v1."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze import seed_series, escape, tobit
PR = json.load(open(os.path.join(HERE, "protocol_esc.json"))); CF = PR["confirmation"]
rows = []
for cell in CF["cells"]:
    p = cell["p"]; us, gs = seed_series(json.load(open(os.path.join(HERE, f"confirm_p{p}.json"))))
    for s, g in enumerate(gs):
        T, c, back = escape(us, g, max(us))
        lo = next((us[i] for i, x in enumerate(g) if x >= 0.1), None); hi = next((us[i] for i, x in enumerate(g) if x >= 0.9), None)
        rows.append({"p": p, "seed": CF["seeds"][s], "T": T, "censored": c, "switch_back": back, "last_checkpoint": max(us),
                     "decorrelated_count": p * T, "transition_width": None if lo is None or hi is None else hi - lo})
x = np.log(1 / np.array([r["p"] for r in rows])); y = np.log([r["T"] for r in rows]); c = np.array([r["censored"] for r in rows])
fit = tobit(x, y, c); lo, hi = fit["b_ci95"]
h1 = "approx_1_over_p_supported" if lo >= 0.75 and hi <= 1.25 else "pure_1_over_p_rejected" if (lo > 1 or hi < 1) else "inconclusive"
cells = []
for cell in CF["cells"]:
    p = cell["p"]; rs = [r for r in rows if r["p"] == p]
    pred = PR["predicted_median_escape_updates"][str(p)]
    med = float(np.median([r["T"] for r in rs])); ncens = sum(r["censored"] for r in rs)
    ok = ncens <= len(rs) / 2 and abs(np.log(med / pred)) <= np.log(1.5)
    cells.append({"p": p, "predicted_median": pred, "observed_median": med, "ratio": med / pred, "censored": ncens, "pass": bool(ok)})
h2 = "supported" if all(cl["pass"] for cl in cells) else "not_supported"
nback = sum(r["switch_back"] for r in rows); h3 = "supported" if nback <= 1 else "not_supported"
out = {"H1": {**fit, "verdict": h1}, "H2": {"cells": cells, "verdict": h2}, "H3": {"switch_backs": nback, "n": len(rows), "verdict": h3}, "rows": rows}
json.dump(out, open(os.path.join(HERE, "confirmation_esc.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))
