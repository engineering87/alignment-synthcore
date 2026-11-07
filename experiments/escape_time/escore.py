"""Score dense checkpoints: for each seed at each checkpoint, sampled rollouts on one battery (compiled once)
and the measure3 fit (identical to frozen v2 in the 7x7 room). Results cached per run in a JSON file."""
import sys, os, json, glob, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure3 as M3, batteries3 as B3
ENV = os.path.join(HERE, "env_r7")
sys.path.insert(0, ENV)
import jax, jax.numpy as jnp
import gw, ppo
M3.configure(7, 20, -10, 10, slack=4)


class Scorer:
    def __init__(self, battery, K, arch="cnn_l"):
        self.b, self.K = battery, K
        A = jnp.asarray(np.repeat(battery["agent"], K, 0)); G = jnp.asarray(np.repeat(battery["gem"], K, 0))
        C = jnp.asarray(battery["proxy"]); T = gw.T_MAX

        def one(par, key):
            def body(carry, t):
                a, out, steps, key = carry
                key, sub = jax.random.split(key)
                lg = ppo.apply(par, jax.vmap(gw.obs_fn)(a, G), arch)[0]
                act = jax.random.categorical(sub, lg)
                nxt = jax.vmap(gw.move)(a, act); live = out == 0
                a = jnp.where(live[:, None], nxt, a)
                hit = jnp.where((a == G).all(1), 1, jnp.where((a == C).all(1), 2, 0))
                steps = jnp.where(live & (hit > 0), t + 1, steps)
                return (a, jnp.where(live, hit, out), steps, key), None
            init = (A, jnp.zeros(A.shape[0], jnp.int32), jnp.full(A.shape[0], T + 1), key)
            (_, out, steps, _), _ = jax.lax.scan(body, init, jnp.arange(T))
            return out, steps
        self.f = jax.jit(one)

    def fit(self, params_one, key):
        out, st = self.f(params_one, key)
        n = len(self.b["agent"])
        out, st = np.asarray(out).reshape(n, self.K), np.asarray(st).reshape(n, self.K)
        f = M3.fit(out, st, self.b["dg"], self.b["dc"], self.b["delta"])
        return {k: f[k] for k in ("type", "w", "w_ci", "directed_coverage", "directed_gem_share", "gem_rate", "proxy_rate", "neither_rate")}


def load(path):
    z = np.load(path, allow_pickle=False); P = {}
    for k in z.files:
        l, n = k.split("__"); P.setdefault(l, {})[n] = z[k]
    return P


def score_run(run_dir, scorer, cache_path, key_seed=0):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for path in sorted(glob.glob(os.path.join(run_dir, "u*.npz"))):
        u = os.path.basename(path)[1:-4]
        if u in cache:
            continue
        P = load(path); n = P["l1"]["w"].shape[0]; rows = []
        for s in range(n):
            par = jax.tree.map(lambda x: jnp.asarray(x[s]), P)
            rows.append(scorer.fit(par, jax.random.PRNGKey(key_seed * 1000 + s)))
        cache[u] = rows
        tmp = cache_path + ".tmp"; json.dump(cache, open(tmp, "w")); os.replace(tmp, cache_path)
    return cache


if __name__ == "__main__":
    run_dir, cache_path, bseed, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    sc = Scorer(B3.make(7, bseed)["original"], K)
    t = time.time(); c = score_run(run_dir, sc, cache_path)
    for u in sorted(c):
        print(u, " ".join(f"{r['type'][:5]}:{r['gem_rate']:.2f}" for r in c[u]))
    print("seconds", round(time.time() - t, 1))
