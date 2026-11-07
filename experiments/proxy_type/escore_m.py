"""Score dense checkpoints in the visible-marker environment: sampled rollouts on a marker battery (marker
in the observation; reaching it ends the trial as a proxy choice), measure4 fit (= v2 logic)."""
import sys, os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import measure4 as M4
sys.path.insert(0, os.path.join(HERE, "env_marker"))
import jax, jax.numpy as jnp
import gw, ppo
M4.configure(7, 20, -10, 10, slack=4)


class Scorer:
    def __init__(self, battery, K, arch="cnn_l"):
        self.b, self.K = battery, K
        A = jnp.asarray(np.repeat(battery["agent"], K, 0)); G = jnp.asarray(np.repeat(battery["gem"], K, 0))
        Mk = jnp.asarray(np.repeat(battery["proxy"], K, 0)); T = gw.T_MAX

        def one(par, key):
            def body(carry, t):
                a, out, steps, key = carry
                key, sub = jax.random.split(key)
                lg = ppo.apply(par, jax.vmap(gw.obs_fn)(a, G, Mk), arch)[0]
                act = jax.random.categorical(sub, lg)
                nxt = jax.vmap(gw.move)(a, act); live = out == 0
                a = jnp.where(live[:, None], nxt, a)
                hit = jnp.where((a == G).all(1), 1, jnp.where((a == Mk).all(1), 2, 0))
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
        f = M4.fit(out, st, self.b["dg"], self.b["dc"], self.b["delta"])
        return {k: f[k] for k in ("type", "w", "w_ci", "directed_coverage", "directed_gem_share", "gem_rate", "proxy_rate", "neither_rate")}


def load(path):
    z = np.load(path, allow_pickle=False); P = {}
    for k in z.files:
        l, n = k.split("__"); P.setdefault(l, {})[n] = z[k]
    return P


def score_run(run_dir, scorer, cache_path):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for path in sorted(glob.glob(os.path.join(run_dir, "u*.npz"))):
        u = os.path.basename(path)[1:-4]
        if u in cache:
            continue
        P = load(path); n = P["l1"]["w"].shape[0]
        cache[u] = [scorer.fit(jax.tree.map(lambda x: jnp.asarray(x[s]), P), jax.random.PRNGKey(s)) for s in range(n)]
        tmp = cache_path + ".tmp"; json.dump(cache, open(tmp, "w")); os.replace(tmp, cache_path)
    return cache
