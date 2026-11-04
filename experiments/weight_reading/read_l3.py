import numpy as np, glob, os
R = np.repeat(np.arange(9), 9).astype(float); C = np.tile(np.arange(9), 9).astype(float)
def slope(y, x): x = x - x.mean(); return float((x * (y - y.mean())).sum() / (x * x).sum())
rows = []
for f in sorted(glob.glob("blind/L3/*.npz")):
    z = np.load(f); W1, W2, W3 = z["W1"], z["W2"], z["W3"]
    M = W3 @ W2 @ W1                     # 4 x 162 linear path, actions up,down,left,right
    g, a = M[:, 81:], M[:, :81]
    # goal channel: goal lower (larger r) -> prefer down; goal righter -> prefer right
    gv = slope(g[1] - g[0], R); gh = slope(g[3] - g[2], C)
    av = slope(a[1] - a[0], R); ah = slope(a[3] - a[2], C)
    # agent toward corner: agent lower -> prefer up (av negative); agent left -> prefer right (ah negative)
    G = (gv + gh) / 2; A = -(av + ah) / 2
    gn = np.linalg.norm(W1[:, 81:]); an = np.linalg.norm(W1[:, :81])
    rows.append((os.path.basename(f)[:-4], G, A, G / A, gn / an))
rows.sort(key=lambda r: r[3])
print("id      goalslope agentslope ratio  W1goal/agent")
for r in rows: print("%s  %8.3f  %8.3f  %6.3f  %6.3f" % r)
