"""KL anchoring experiment 3, pre-registered confirmation: does anchor-permitted drift start, and end up,
in states the reference rarely saw? Environment, architectures, reference fitting and the RL update are
arch_sweep.py from the original experiment, unchanged. One addition: method 'klonly', the k1-in-reward
update with the task reward set to zero. Its optimum is the reference itself, so its drift is optimizer
noise through shared parameters; drift(k1rew) - drift(klonly) is the drift the reward caused."""
import sys, json, os, time
import numpy as np, torch, torch.nn.functional as F
import arch_sweep as A
from arch_sweep import NC, NA, W, STEP_C, chain, build

CK = (0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500)


def rl(st, ref_net, arch, method, beta, steps=1500, batch=64, lr=1e-3, seed=0, ckpts=CK, Tmax=60, on_ckpt=None):
    """arch_sweep.rl restricted to k1rew / none, plus klonly (same as k1rew with R = 0)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = build(arch, len(st.all)); net.load_state_dict(ref_net.state_dict())
    for p in net.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), 0.05 if arch == "tabular" else lr)
    rT = st.rows("T"); ref_all = st.table(ref_net); lrefT = torch.tensor(np.log(ref_all[rT]))
    nxt = np.stack([L.nxt for L in st.T]); nT = st.nT
    for it in range(steps + 1):
        if it in ckpts: on_ckpt(it, st.table(net))
        if it == steps: break
        lp = F.log_softmax(net(st.X[rT], st.ids[rT]), 1); piT = lp.exp().detach().double().numpy()
        Ls = rng.integers(nT, size=batch); s = np.zeros(batch, int); alive = np.ones(batch, bool)
        R = np.zeros(batch); KL = np.zeros(batch); vis = []
        lpn = np.log(piT + 1e-300); lrn = lrefT.numpy()
        for t in range(Tmax):
            row = Ls * NC + s
            cum = piT[row].cumsum(1); a = (rng.random(batch)[:, None] > cum).sum(1).clip(0, NA - 1)
            vis.append((row[alive], a[alive], np.nonzero(alive)[0]))
            R -= STEP_C * alive; KL += alive * (lpn[row, a] - lrn[row, a])
            s = np.where(alive, nxt[Ls, s, a], s); alive &= s != W - 1
            if not alive.any(): break
        if method == "k1rew": ret = R - beta * KL
        elif method == "klonly": ret = -beta * KL
        elif method == "none": ret = R
        else: raise ValueError(method)
        adv = ret - ret.mean()
        Wt = np.zeros((nT * NC, NA))
        for row, a, b in vis: np.add.at(Wt, (row, a), adv[b])
        loss = -(torch.tensor(Wt / batch) * lp.double()).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def groups(st):
    """Masks over all rows by how often the human demonstrations (the reference's training data) visited
    the state, summed over the 300 demos: high >= 200 (about the top quarter of trained-layout states), mid 70-199,
    low 1-69 (about the bottom quarter), zero 0. O layouts are never in the data."""
    n = np.concatenate(st.counts).sum(1); v = st.valid; g = {}
    for k in "TBO":
        m = np.zeros_like(v); m[st.rows(k)] = True; m &= v
        g[k + "_high"] = m & (n >= 200); g[k + "_mid"] = m & (n >= 70) & (n < 200); g[k + "_low"] = m & (n >= 1) & (n < 70); g[k + "_zero"] = m & (n == 0); g[k + "_all"] = m
    return g


def run_unit(eps, seed, arch, plan):
    st = A.Study(eps, seed=seed); ref = st.fit_ref(arch, seed=seed); pr = st.table(ref); g = groups(st)
    out = dict(eps=eps, seed=seed, arch=arch, n={k: int(m.sum()) for k, m in g.items()},
               ref_door={k: st.door_rates(pr, k) for k in "TBO"}, runs=[])
    for method, beta in plan:
        log = []
        def ck(it, pi):
            tv = 0.5 * np.abs(pi - pr).sum(1)
            log.append(dict(it=it, tv={k: float(tv[m].mean()) if m.any() else None for k, m in g.items()},
                            door={k: float(np.mean(st.door_rates(pi, k))) for k in "TBO"}))
        rl(st, ref, arch, method, beta, seed=seed, on_ckpt=ck)
        out["runs"].append(dict(method=method, beta=beta, log=log))
    return out


if __name__ == "__main__":
    pr = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocol_kl3.json")))
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    units = [(e, s, a) for s in pr["seeds"] for e in pr["eps"] for a in pr["archs"]]
    if len(sys.argv) > 2: units = units[int(sys.argv[2])::int(sys.argv[3])]   # shard k of n
    plan = [tuple(x) for x in pr["plan"]]
    for e, s, a in units:
        f = os.path.join(outdir, f"u_e{e}_s{s}_{a}.json")
        if os.path.exists(f): continue
        t0 = time.time(); r = run_unit(e, s, a, plan)
        json.dump(r, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
        fin = {f"{x['method']}{x['beta']}": round(x['log'][-1]['tv']['O_all'] / max(x['log'][-1]['tv']['T_high'], 1e-9), 2) for x in r["runs"]}
        print(json.dumps({"unit": [e, s, a], "sec": round(time.time() - t0), "O/Thigh": fin}), flush=True)
