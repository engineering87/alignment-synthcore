import numpy as np, glob, os
DR = np.array([-1, 1, 0, 0]); DC = np.array([0, 0, -1, 1])
def field(tr, tc):
    t = np.zeros((4, 81))
    for i in range(81):
        r, c = divmod(i, 9)
        d0 = abs(r - tr) + abs(c - tc)
        for a in range(4):
            nr, nc = min(max(r + DR[a], 0), 8), min(max(c + DC[a], 0), 8)
            t[a, i] = d0 - (abs(nr - tr) + abs(nc - tc))   # +1 closer, -1 farther, 0 wall/same
    return t - t.mean(0, keepdims=True)
Tcorner, Tcenter = field(0, 8), field(4, 4)
def corr(x, t):
    x = x - x.mean(0, keepdims=True); x = x - x.mean(1, keepdims=True)
    return float((x * t).sum() / np.linalg.norm(x) / np.linalg.norm(t))
out = {}
for f in sorted(glob.glob("blind/L3/*.npz")):
    z = np.load(f); M = z["W3"] @ z["W2"] @ z["W1"]
    a = M[:, :81]
    out[os.path.basename(f)[:-4]] = (corr(a, Tcorner), corr(a, Tcenter))
if __name__ == "__main__":
    for k, (cc, ce) in sorted(out.items(), key=lambda kv: kv[1][0] - kv[1][1]):
        print("%s corner %.3f center %.3f diff %+.3f" % (k, cc, ce, cc - ce))
