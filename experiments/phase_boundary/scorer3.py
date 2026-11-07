"""Sampled rollouts of trained policies in JAX (vmapped over seeds), returning outcomes and step counts
in the format measure3.fit expects. Same episode semantics as measure3.rollouts."""
import sys, os
import numpy as np


def setup(env_dir):
    sys.path.insert(0, env_dir)
    import jax, jax.numpy as jnp, gw, ppo  # noqa: E401
    return jax, jnp, gw, ppo


def load(path):
    z = np.load(path, allow_pickle=False); P = {}
    for k in z.files:
        l, n = k.split("__"); P.setdefault(l, {})[n] = z[k]
    return P


def rollouts(env_dir, params, arch, battery, K, seed, greedy=False):
    jax, jnp, gw, ppo = setup(env_dir)
    A = jnp.asarray(np.repeat(battery["agent"], K, 0)); G = jnp.asarray(np.repeat(battery["gem"], K, 0))
    C = jnp.asarray(battery["proxy"]); T = gw.T_MAX

    def one(par, key):
        def body(carry, t):
            a, out, steps, key = carry
            key, sub = jax.random.split(key)
            lg = ppo.apply(par, jax.vmap(gw.obs_fn)(a, G), arch)[0]
            act = jnp.argmax(lg, -1) if greedy else jax.random.categorical(sub, lg)
            nxt = jax.vmap(gw.move)(a, act)
            live = out == 0
            a = jnp.where(live[:, None], nxt, a)
            hit = jnp.where((a == G).all(1), 1, jnp.where((a == C).all(1), 2, 0))
            steps = jnp.where(live & (hit > 0), t + 1, steps)
            out = jnp.where(live, hit, out)
            return (a, out, steps, key), None
        init = (A, jnp.zeros(A.shape[0], jnp.int32), jnp.full(A.shape[0], T + 1), key)
        (_, out, steps, _), _ = jax.lax.scan(body, init, jnp.arange(T))
        return out, steps

    n_seeds = params["l1"]["w"].shape[0]
    keys = jax.random.split(jax.random.PRNGKey(seed), n_seeds)
    f = jax.jit(one)  # one seed at a time: vmapping seeds holds every seed's CNN patches in memory
    outs, steps = [], []
    for s in range(n_seeds):
        o, st = f(jax.tree.map(lambda x: jnp.asarray(x[s]), params), keys[s])
        outs.append(np.asarray(o)); steps.append(np.asarray(st))
    n = len(battery["agent"])
    return np.stack(outs).reshape(n_seeds, n, K), np.stack(steps).reshape(n_seeds, n, K)
