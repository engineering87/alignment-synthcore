import sys; sys.path[:0] = ["../src/experiments/preference_audit/vendor", "."]
import numpy as np, jax, jax.numpy as jnp, ppo, policy2 as P2
rng = np.random.default_rng(0)
for arch in ("mlp_s", "mlp_l", "cnn_s", "cnn_l"):
    params = jax.vmap(lambda k: ppo.init_params(k, arch))(jnp.stack([jax.random.PRNGKey(i) for i in range(2)]))
    # perturb so biases are nonzero
    params = jax.tree.map(lambda x: x + 0.1 * jnp.asarray(rng.standard_normal(x.shape), jnp.float32), params)
    Pn = {l: {n: np.asarray(v) for n, v in r.items()} for l, r in params.items()}
    a = rng.integers(1, 8, (50, 2)); g = rng.integers(1, 8, (50, 2)); o = P2.obs(a, g)
    for s in range(2):
        pj = jax.tree.map(lambda x: x[s], params)
        lj = np.asarray(ppo.apply(pj, jnp.asarray(o), arch)[0]); ln = P2.logits(Pn, s, o)
        print(arch, s, float(np.abs(lj - ln).max()))
