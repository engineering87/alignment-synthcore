import sys, time
sys.path[:0] = ["src/experiments/preference_audit/vendor"]
import jax, jax.numpy as jnp, ppo
for arch in ("mlp_s", "mlp_l", "cnn_s", "cnn_l"):
    keys = jnp.stack([jax.random.PRNGKey(s) for s in range(4)])
    init, step = ppo.make_trainer(arch, 50, 0.05); c = init(keys, 0.3)
    t = time.time(); c = step(c); jax.block_until_ready(c["params"]); t1 = time.time() - t
    t = time.time(); c = step(c); jax.block_until_ready(c["params"]); t2 = time.time() - t
    print(arch, f"compile+50 {t1:.1f}s, next 50 updates x4 seeds {t2:.1f}s")
