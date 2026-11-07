import sys, json
import numpy as np
sys.path[:0] = ["/home/claude/pm/sweep", "/home/claude/pm"]
import measure as M2          # frozen v2
import measure3 as M3
import batteries3 as B3
import policy2 as P2
import scorer3 as S3

# 1. measure3 == v2 in the 7x7 room on the same battery and controls
b = B3.make(7, 5)["original"]; M3.configure(7, 20, -10, 10)
same = True
for kind, kw in [("optimal", dict(w=0.3)), ("optimal", dict(w=0.7, eps=0.1)), ("soft_optimal", dict(w=0.5, tau=2.0)),
                 ("mixture", {}), ("random", {})]:
    fits = []
    for M in (M2, M3):
        rng = np.random.default_rng(1)
        act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=16, rng=rng, w=kw.get("w"), eps=kw.get("eps", 0), tau=kw.get("tau"))
        out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], 16, rng)
        fits.append(M.fit(out, st, b["dg"], b["dc"], b["delta"]))
    same &= json.dumps(fits[0], sort_keys=True, default=str) == json.dumps(fits[1], sort_keys=True, default=str)
print("1. measure3 identical to v2 in 7x7:", same)

# 2. JAX scorer == NumPy greedy rollouts (competence-sweep CNN, 7x7)
P = S3.load("/home/claude/pm/sweep/confirm/cnn_l_e0.05_u300.npz")
oj, sj = S3.rollouts("/home/claude/pm/src/experiments/preference_audit/vendor", P, "cnn_l", b, 1, 0, greedy=True)
ok = True
for s in range(4):
    on, sn = M3.rollouts(P2.act_fn(P, s, "greedy"), b["agent"], b["gem"], b["proxy"], 1, np.random.default_rng(0))
    ok &= np.array_equal(on, oj[s]) and np.array_equal(sn, sj[s])
print("2. JAX greedy rollouts identical to NumPy:", ok)
# sampled: compare fitted shares statistically
oj, sj = S3.rollouts("/home/claude/pm/src/experiments/preference_audit/vendor", P, "cnn_l", b, 32, 0)
for s in range(2):
    on, sn = M3.rollouts(P2.act_fn(P, s), b["agent"], b["gem"], b["proxy"], 32, np.random.default_rng(0))
    fj, fn = M3.fit(oj[s], sj[s], b["dg"], b["dc"], b["delta"]), M3.fit(on, sn, b["dg"], b["dc"], b["delta"])
    print(f"   sampled seed {s}: JAX cov {fj['directed_coverage']:.3f} gem {fj['gem_rate']:.3f} {fj['type']} | NumPy cov {fn['directed_coverage']:.3f} gem {fn['gem_rate']:.3f} {fn['type']}")

# 3. controls in the 11x11 room
bats = B3.make(11, 7); M3.configure(11, 40, -18, 18, slack=8)
print(f"3. 11x11 controls, measurable w range [{M3.W_LO:.3f}, {M3.W_HI:.3f}]")
for name in ("original", "reflected", "distance_reweighted"):
    b = bats[name]
    for kind, kw in [("optimal", dict(w=w, eps=e)) for w in (0.03, 0.05, 0.15, 0.5, 0.85, 0.95, 0.97) for e in (0.0, 0.2)] + \
                    [("soft_optimal", dict(w=0.95, tau=2.0)), ("optimal", dict(w=0.995)), ("mixture", {}), ("random", {})]:
        rng = np.random.default_rng(2)
        act = M3.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=32, rng=rng, w=kw.get("w"), eps=kw.get("eps", 0), tau=kw.get("tau"))
        out, st = M3.rollouts(act, b["agent"], b["gem"], b["proxy"], 32, rng)
        f = M3.fit(out, st, b["dg"], b["dc"], b["delta"]); w = kw.get("w")
        flag = "" if w is None or f["w_ci"] is None else ("IN " if f["w_ci"][0] <= w <= f["w_ci"][1] else "OUT")
        print(f"   {name[:4]} {flag:3s} {kind:12s} {str(kw):24s} cov {f['directed_coverage']:.2f} {f['type']:23s} w {None if f['w'] is None else round(f['w'],3)} ci {None if f['w_ci'] is None else [round(x,3) for x in f['w_ci']]}")
