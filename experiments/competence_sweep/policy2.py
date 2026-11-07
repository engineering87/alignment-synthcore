"""NumPy forward pass for every vendored ppo.py architecture (mlp_s, mlp_l, cnn_s, cnn_l)."""
import numpy as np
WALL = np.ones((9, 9), np.float32); WALL[1:8, 1:8] = 0


def load(path):
    z = np.load(path, allow_pickle=False); P = {}
    for k in z.files:
        l, n = k.split("__"); P.setdefault(l, {})[n] = z[k]
    return P


def n_seeds(P):
    return P["l1"]["w"].shape[0]


def obs(a, g):
    n = len(a); idx = np.arange(n); o = np.zeros((n, 9, 9, 3), np.float32)
    o[:, :, :, 0] = WALL; o[idx, a[:, 0], a[:, 1], 1] = 1; o[idx, g[:, 0], g[:, 1], 2] = 1
    return o


def logits(P, s, o):
    x = o
    if "c1" in P:
        for name in ("c1", "c2"):  # 3x3 SAME conv as patches @ W, same patch order as ppo.apply
            xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
            pat = np.concatenate([xp[:, i:i + 9, j:j + 9, :] for i in range(3) for j in range(3)], axis=-1)
            x = np.maximum(pat @ P[name]["w"][s] + P[name]["b"][s], 0)
        h = x.reshape(len(x), -1)
        h = np.maximum(h @ P["l1"]["w"][s] + P["l1"]["b"][s], 0)
    else:
        h = x.reshape(len(x), -1)
        for l in ("l1", "l2"):
            h = np.maximum(h @ P[l]["w"][s] + P[l]["b"][s], 0)
    return h @ P["pi"]["w"][s] + P["pi"]["b"][s]


def act_fn(P, s, mode="sample"):
    def act(a, g, rng):
        lg = logits(P, s, obs(a, g))
        if mode == "greedy":
            return lg.argmax(1)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return (p.cumsum(1) > rng.random((len(a), 1))).argmax(1)
    return act
