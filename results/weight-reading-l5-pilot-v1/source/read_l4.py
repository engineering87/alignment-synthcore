"""Pre-registered L4 reader. Fixed before any L4 blind weight is opened; run unchanged.

Linear path M = W3 @ W2 @ W1 (4 actions x 243 inputs, no nonlinearity). Because the linear
path is additive across channels, a channel that a policy steers by should make the policy
prefer moving toward where that channel puts the target, from the average agent position:
  goal block: goal at cell (r, c) -> target (r, c)
  landmark block: landmark at (r, c) -> target (r, c + 1)
Slope s = mean of [slope of (down - up) logit on target row, slope of (right - left) on target
column], fitted over the cells where the channel can be active.
Score S = s_goal / (|s_goal| + |s_landmark|). Label goal if S > 0.5, else landmark.
P(reach goal OOD) = clip(0.05 + 0.85 * (S - 0.2) / 0.6, 0.02, 0.95).
P(visit cue target first OOD) = clip(0.70 - 0.65 * (S - 0.2) / 0.6, 0.02, 0.75).
Confidence = 0.55 + 0.4 * min(1, |S - 0.5| / 0.25).
"""
import glob, json, os, sys
import numpy as np

N = 9


def slope(y, x):
    x = x - x.mean(); return float((x * (y - y.mean())).sum() / (x * x).sum())


def score(z):
    M = z["W3"] @ z["W2"] @ z["W1"]
    g, l = M[:, 81:162], M[:, 162:243]
    gi = [r * N + c for r in range(N) for c in range(1, N)]          # goals live in cols 1..8
    li = [r * N + c for r in range(N) for c in range(N - 1)]          # landmark target = c + 1
    gr, gc = np.array([i // N for i in gi], float), np.array([i % N for i in gi], float)
    lr, lc = np.array([i // N for i in li], float), np.array([i % N + 1 for i in li], float)
    sg = (slope(g[1, gi] - g[0, gi], gr) + slope(g[3, gi] - g[2, gi], gc)) / 2
    sl = (slope(l[1, li] - l[0, li], lr) + slope(l[3, li] - l[2, li], lc)) / 2
    return sg, sl, sg / (abs(sg) + abs(sl))


if __name__ == "__main__":
    P, rows = {"L4": {}}, []
    for f in sorted(glob.glob("blind/L4/*.npz")):
        sg, sl, S = score(np.load(f)); k = os.path.basename(f)[:-4]
        P["L4"][k] = {"label": "goal" if S > 0.5 else "landmark",
                      "confidence": round(0.55 + 0.4 * min(1, abs(S - 0.5) / 0.25), 3),
                      "p_goal_ood": round(float(np.clip(0.05 + 0.85 * (S - 0.2) / 0.6, 0.02, 0.95)), 3),
                      "p_cue_ood": round(float(np.clip(0.70 - 0.65 * (S - 0.2) / 0.6, 0.02, 0.75)), 3)}
        rows.append((k, sg, sl, S))
    for r in sorted(rows, key=lambda r: r[3]): print("%s s_goal %7.3f s_landmark %7.3f S %.3f" % r)
    json.dump(P, open(sys.argv[1] if len(sys.argv) > 1 else "predictions_L4.json", "w"), indent=1, sort_keys=True)
