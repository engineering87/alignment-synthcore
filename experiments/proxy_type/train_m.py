"""Dense-checkpoint trainer for the escape-law factorial: escape-time-v1 train_dense.py plus a cnn_xl
entry (16 channels, 128 hidden) registered at run time; ppo.py itself is unchanged.
Dense-checkpoint trainer (escape-time study). Same as phase-boundary train3r.py except weights are
saved under <outdir>/<tag>/uNNNNN.npz. train3.py with resume: saves the complete training state (params, optimizer, env states, RNG keys,
stats) after every 50-update chunk and continues from it after an interruption. The training
computation is unchanged; resume must reproduce an uninterrupted run bit for bit (test_resume)."""
import sys, json, time, os
import numpy as np
env_dir, arch, p, ent, seeds, cps, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), \
    [int(x) for x in sys.argv[5].split(",")], sorted(set(range(10, min(200, int(sys.argv[6])) + 1, 10)) | set(range(200, int(sys.argv[6]) + 1, 50))), sys.argv[7]  # proxy-type: every 10 updates to 200, then every 50
sys.path.insert(0, env_dir)
import jax, jax.numpy as jnp
import ppo, gw
ppo.ARCHS.setdefault("cnn_xl", dict(kind="cnn", c=16, h=128))  # double-width CNN (only addition)
CHUNK = 10  # proxy-type: finer chunks so escapes before 50 updates are resolved
os.makedirs(outdir, exist_ok=True)
tag = f"{arch}_p{p}_e{ent}"
os.makedirs(os.path.join(outdir, tag), exist_ok=True)
state_path = os.path.join(outdir, f"{tag}_state.npz")
log_path = os.path.join(outdir, f"{tag}_log.json")
keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
init, step = ppo.make_trainer(arch, CHUNK, ent)
carry = init(keys, p)
leaves, treedef = jax.tree.flatten(carry)
done_chunks, log = 0, []
if os.path.exists(state_path):
    z = np.load(state_path, allow_pickle=False)
    done_chunks = int(z["done_chunks"])
    loaded = [jnp.asarray(z[f"leaf{i}"], dtype=l.dtype) for i, l in enumerate(leaves)]
    assert all(a.shape == b.shape for a, b in zip(loaded, leaves))
    carry = jax.tree.unflatten(treedef, loaded)
    log = json.load(open(log_path))
    print(json.dumps({"resumed_at_updates": done_chunks * CHUNK}), flush=True)
t0 = time.time()
for i in range(done_chunks, max(cps) // CHUNK):
    carry = step(carry); jax.block_until_ready(carry["params"]); u = (i + 1) * CHUNK
    st = np.asarray(carry["stats"])
    log.append({"updates": u, "gem_per_episode": (st[:, 1] / np.maximum(st[:, 0], 1)).tolist(), "seconds_this_session": round(time.time() - t0, 1)})
    if u in cps:
        np.savez_compressed(os.path.join(outdir, tag, f"u{u:05d}.npz"),
                            **{l + "__" + n: np.asarray(v) for l, r in carry["params"].items() for n, v in r.items()})
        if u % 500 == 0: print(json.dumps({"saved": f"{tag}/u{u:05d}", **log[-1]}), flush=True)
    tmp = state_path + ".tmp.npz"
    np.savez(tmp, done_chunks=i + 1, **{f"leaf{k}": np.asarray(l) for k, l in enumerate(jax.tree.leaves(carry))})
    json.dump(log, open(log_path + ".tmp", "w")); os.replace(log_path + ".tmp", log_path); os.replace(tmp, state_path)
json.dump({"env": env_dir, "grid": int(gw.G), "T_MAX": int(gw.T_MAX), "arch": arch, "p": p, "entropy": ent, "seeds": seeds,
           "log": log}, open(os.path.join(outdir, f"{tag}_train.json"), "w"))
