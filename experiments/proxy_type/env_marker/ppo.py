"""Minimal PPO in plain JAX, vmapped across seeds.

Architectures (capacity axis):
  mlp_s : MLP, hidden 16
  mlp_l : MLP, hidden 256
  cnn_s : conv 3x3 x4  -> conv 3x3 x4  -> dense 32
  cnn_l : conv 3x3 x8  -> conv 3x3 x8  -> dense 64
"""
import functools
import jax
import jax.numpy as jnp
import optax
import gw

ARCHS = {
    "mlp_s": dict(kind="mlp", h=16),
    "mlp_l": dict(kind="mlp", h=256),
    "cnn_s": dict(kind="cnn", c=4, h=32),
    "cnn_l": dict(kind="cnn", c=8, h=64),
}

N_ENVS = 64
N_STEPS = 32
EPOCHS = 4
N_MB = 4
CLIP = 0.2
ENT = 0.01
VF = 0.5
LR = 5e-4
LAM = 0.95


def _dense(key, i, o, scale=1.0):
    w = jax.random.orthogonal(key, max(i, o))[:i, :o] * scale
    return dict(w=w, b=jnp.zeros(o))


def init_params(key, arch):
    a = ARCHS[arch]
    ks = jax.random.split(key, 6)
    if a["kind"] == "mlp":
        d = gw.G * gw.G * 4
        return dict(l1=_dense(ks[0], d, a["h"], 2 ** .5), l2=_dense(ks[1], a["h"], a["h"], 2 ** .5),
                    pi=_dense(ks[2], a["h"], gw.N_ACT, 0.01), v=_dense(ks[3], a["h"], 1, 1.0))
    c = a["c"]
    conv = lambda k, ci, co: dict(w=jax.random.normal(k, (9 * ci, co)) * (2 / (9 * ci)) ** .5,
                                  b=jnp.zeros(co))
    return dict(c1=conv(ks[0], 4, c), c2=conv(ks[1], c, c),
                l1=_dense(ks[2], gw.G * gw.G * c, a["h"], 2 ** .5),
                pi=_dense(ks[3], a["h"], gw.N_ACT, 0.01), v=_dense(ks[4], a["h"], 1, 1.0))


def apply(params, obs, arch):
    """obs: (..., 9, 9, 3) -> logits (..., 4), value (...)."""
    kind = ARCHS[arch]["kind"]
    lead = obs.shape[:-3]
    x = obs.reshape((-1,) + obs.shape[-3:])
    if kind == "mlp":
        h = x.reshape(x.shape[0], -1)
        h = jax.nn.relu(h @ params["l1"]["w"] + params["l1"]["b"])
        h = jax.nn.relu(h @ params["l2"]["w"] + params["l2"]["b"])
    else:
        for name in ("c1", "c2"):   # 3x3 SAME conv as patches @ W (vmap-friendly on CPU)
            xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
            pat = jnp.concatenate([xp[:, i:i + gw.G, j:j + gw.G, :]
                                   for i in range(3) for j in range(3)], axis=-1)
            x = jax.nn.relu(pat @ params[name]["w"] + params[name]["b"])
        h = x.reshape(x.shape[0], -1)
        h = jax.nn.relu(h @ params["l1"]["w"] + params["l1"]["b"])
    logits = h @ params["pi"]["w"] + params["pi"]["b"]
    v = (h @ params["v"]["w"] + params["v"]["b"])[:, 0]
    return logits.reshape(lead + (gw.N_ACT,)), v.reshape(lead)


def make_trainer(arch, n_updates_per_chunk, ent=ENT):
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR, eps=1e-5))
    ent_coef = ent

    def init(key, p):
        k1, k2 = jax.random.split(key)
        params = init_params(k1, arch)
        env_keys = jax.random.split(k2, N_ENVS)
        states = jax.vmap(gw.reset, in_axes=(0, None))(env_keys, p)
        obs = jax.vmap(gw.obs_fn)(states["agent"], states["gem"], states["marker"])
        return dict(params=params, opt=opt.init(params), states=states, obs=obs,
                    key=key, p=p, ep_ret=jnp.zeros(N_ENVS), stats=jnp.zeros(2))

    def update(carry, _):
        params, ostate, states, obs, key, p = (carry[k] for k in
                                               ("params", "opt", "states", "obs", "key", "p"))

        def env_step(c, _):
            states, obs, key = c
            key, ka, ke = jax.random.split(key, 3)
            logits, v = apply(params, obs, arch)
            act = jax.random.categorical(ka, logits)
            logp = jax.nn.log_softmax(logits)[jnp.arange(N_ENVS), act]
            ekeys = jax.random.split(ke, N_ENVS)
            states, obs2, r, d = jax.vmap(gw.step, in_axes=(0, 0, 0, None))(states, act, ekeys, p)
            return (states, obs2, key), (obs, act, logp, v, r, d)

        (states, obs, key), (b_obs, b_act, b_logp, b_v, b_r, b_d) = jax.lax.scan(
            env_step, (states, obs, key), None, length=N_STEPS)
        _, last_v = apply(params, obs, arch)

        def gae(c, x):
            adv_next, v_next = c
            r, d, v = x
            nonterm = 1.0 - d
            delta = r + gw.GAMMA * v_next * nonterm - v
            adv = delta + gw.GAMMA * LAM * nonterm * adv_next
            return (adv, v), adv

        _, adv = jax.lax.scan(gae, (jnp.zeros(N_ENVS), last_v), (b_r, b_d.astype(jnp.float32), b_v),
                              reverse=True)
        ret = adv + b_v
        flat = lambda x: x.reshape((N_STEPS * N_ENVS,) + x.shape[2:])
        data = tuple(map(flat, (b_obs, b_act, b_logp, adv, ret)))

        def loss_fn(params, mb):
            o, a, lp_old, ad, rt = mb
            logits, v = apply(params, o, arch)
            lsm = jax.nn.log_softmax(logits)
            lp = lsm[jnp.arange(a.shape[0]), a]
            ratio = jnp.exp(lp - lp_old)
            ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            pg = -jnp.minimum(ratio * ad, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * ad).mean()
            vl = ((v - rt) ** 2).mean()
            ent = -(jnp.exp(lsm) * lsm).sum(-1).mean()
            return pg + VF * vl - ent_coef * ent

        def epoch(c, _):
            params, ostate, key = c
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, N_STEPS * N_ENVS).reshape(N_MB, -1)

            def mb_step(c2, idx):
                params, ostate = c2
                mb = tuple(x[idx] for x in data)
                g = jax.grad(loss_fn)(params, mb)
                upd, ostate = opt.update(g, ostate, params)
                return (optax.apply_updates(params, upd), ostate), None

            (params, ostate), _ = jax.lax.scan(mb_step, (params, ostate), perm)
            return (params, ostate, key), None

        (params, ostate, key), _ = jax.lax.scan(epoch, (params, ostate, key), None, length=EPOCHS)
        # stats: episodes finished, gem hits
        stats = carry["stats"] + jnp.array([b_d.sum(), b_r.sum()])
        new = dict(carry, params=params, opt=ostate, states=states, obs=obs, key=key, stats=stats)
        return new, None

    @jax.jit
    def train_chunk(carry):
        carry = dict(carry, stats=jnp.zeros(2))
        carry, _ = jax.lax.scan(update, carry, None, length=n_updates_per_chunk)
        return carry

    return jax.jit(jax.vmap(init, in_axes=(0, None))), jax.jit(jax.vmap(train_chunk))


def make_evaluator(arch, test):
    """Greedy revealed-preference choices for every seed on every test state."""
    def one(params):
        pol = lambda o: jnp.argmax(apply(params, o, arch)[0])
        return jax.vmap(lambda a, g: gw.choice_rollout(pol, a, g))(test["agent"], test["gem"])
    return jax.jit(jax.vmap(one))


def make_success_eval(arch, n=256):
    """In-distribution-free capability check: greedy success at reaching the gem
    when no corner competes (gem anywhere, corner not terminal)."""
    def one(params, key):
        ks = jax.random.split(key, n)
        def ep(k):
            s = gw.reset(k, 1.0)
            pol = lambda o: jnp.argmax(apply(params, o, arch)[0])
            def body(c, _):
                a, done = c
                a2 = gw.move(a, pol(gw.obs_fn(a, s["gem"])))
                a2 = jnp.where(done, a, a2)
                return (a2, done | jnp.all(a2 == s["gem"])), None
            (_, done), _ = jax.lax.scan(body, (s["agent"], jnp.array(False)), None, length=gw.T_MAX)
            return done
        return jax.vmap(ep)(ks).mean()
    return jax.jit(jax.vmap(one))
