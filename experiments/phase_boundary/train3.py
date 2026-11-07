"""Train with the unchanged vendored ppo.py inside a room-size variant of gw.py, saving checkpoints."""
import sys, json, time, os
import numpy as np
env_dir, arch, p, ent, seeds, cps, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), \
    [int(x) for x in sys.argv[5].split(",")], [int(x) for x in sys.argv[6].split(",")], sys.argv[7]
sys.path.insert(0, env_dir)
import jax, jax.numpy as jnp
import ppo, gw
CHUNK = 50
os.makedirs(outdir, exist_ok=True)
tag = f"{arch}_p{p}_e{ent}"
keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
init, step = ppo.make_trainer(arch, CHUNK, ent)
carry = init(keys, p); t0 = time.time(); log = []
for i in range(max(cps) // CHUNK):
    carry = step(carry); jax.block_until_ready(carry["params"]); u = (i + 1) * CHUNK
    st = np.asarray(carry["stats"]); log.append({"updates": u, "gem_per_episode": (st[:, 1] / np.maximum(st[:, 0], 1)).tolist(),
                                                 "seconds": round(time.time() - t0, 1)})
    if u in cps:
        np.savez_compressed(os.path.join(outdir, f"{tag}_u{u}.npz"),
                            **{l + "__" + n: np.asarray(v) for l, r in carry["params"].items() for n, v in r.items()})
        print(json.dumps({"saved": f"{tag}_u{u}", **log[-1]}), flush=True)
json.dump({"env": env_dir, "grid": int(gw.G), "T_MAX": int(gw.T_MAX), "arch": arch, "p": p, "entropy": ent, "seeds": seeds,
           "log": log, "seconds": time.time() - t0}, open(os.path.join(outdir, f"{tag}_train.json"), "w"))
