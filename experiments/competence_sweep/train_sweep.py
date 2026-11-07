"""Train with the vendored ppo.py (unchanged), saving policy weights at each checkpoint budget."""
import sys, json, time, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "../src/experiments/preference_audit/vendor")]
import jax, jax.numpy as jnp
import ppo

CHUNK = 50


def run(arch, ent, seeds, checkpoints, outdir, p=0.3, tag=None):
    os.makedirs(outdir, exist_ok=True)
    tag = tag or f"{arch}_e{ent}"
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    init, step = ppo.make_trainer(arch, CHUNK, ent)
    carry = init(keys, p); t0 = time.time(); log = []
    for i in range(max(checkpoints) // CHUNK):
        carry = step(carry); jax.block_until_ready(carry["params"])
        u = (i + 1) * CHUNK
        st = np.asarray(carry["stats"]); log.append({"updates": u, "gem_per_episode": (st[:, 1] / np.maximum(st[:, 0], 1)).tolist()})
        if u in checkpoints:
            path = os.path.join(outdir, f"{tag}_u{u}.npz")
            np.savez_compressed(path, **{l + "__" + n: np.asarray(v) for l, r in carry["params"].items() for n, v in r.items()})
            print(json.dumps({"saved": path, "seconds": round(time.time() - t0, 1), "gem_per_episode": log[-1]["gem_per_episode"]}), flush=True)
    json.dump({"arch": arch, "entropy": ent, "seeds": seeds, "p": p, "log": log, "seconds": time.time() - t0},
              open(os.path.join(outdir, f"{tag}_train.json"), "w"))


if __name__ == "__main__":
    arch, ent, seeds, cps, out = sys.argv[1], float(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], \
        [int(x) for x in sys.argv[4].split(",")], sys.argv[5]
    run(arch, ent, seeds, cps, out)
