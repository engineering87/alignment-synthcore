"""Choice battery for the visible-marker proxy: agent, gem and marker on distinct cells, neither object on
a shortest path to the other from the agent, stratified by delta = d_gem - d_marker."""
import numpy as np


def make(R, seed, per_gap=24):
    rng = np.random.default_rng(seed)
    cells = np.array([(r, c) for r in range(1, R + 1) for c in range(1, R + 1)])
    rows = []
    for _ in range(60000):
        a, g, m = cells[rng.choice(len(cells), 3, replace=False)]
        if ((g >= np.minimum(a, m)) & (g <= np.maximum(a, m))).all() or ((m >= np.minimum(a, g)) & (m <= np.maximum(a, g))).all():
            continue
        rows.append([*a, *g, *m])
    rows = np.unique(np.array(rows), axis=0)
    a, g, m = rows[:, :2], rows[:, 2:4], rows[:, 4:]
    dg, dm = np.abs(a - g).sum(1), np.abs(a - m).sum(1); d = dg - dm
    pick = np.concatenate([rng.choice(np.flatnonzero(d == v), min(per_gap, (d == v).sum()), replace=False) for v in np.unique(d)])
    return dict(agent=a[pick], gem=g[pick], proxy=m[pick], dg=dg[pick], dc=dm[pick], delta=d[pick])
