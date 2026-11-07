import numpy as np, measure3 as M3, batteries3 as B3
M3.configure(11, 40, -18, 18)
res = {}
for seed in (7, 8, 9):  # development battery seeds
    for name, b in B3.make(11, seed).items():
        for kind, kw in [("optimal", dict(w=w, eps=0.2)) for w in (0.03, 0.15, 0.5, 0.85, 0.97)] + [("random", {})]:
            rng = np.random.default_rng(3)
            act = M3.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=32, rng=rng, w=kw.get("w"), eps=kw.get("eps", 0))
            out, st = M3.rollouts(act, b["agent"], b["gem"], b["proxy"], 32, rng)
            d = np.where(out == 1, b["dg"][:, None], np.where(out == 2, b["dc"][:, None], 0))
            for s in range(2, 13):
                res.setdefault((kind, s), []).append(float(((out > 0) & (st <= d + s)).mean()))
for s in range(2, 13):
    e = min(res[("optimal", s)]); r = max(res[("random", s)])
    print(f"slack {s:2d}: worst eps=0.2 coverage {e:.3f} | best random coverage {r:.3f}")
