"""Experiment 3: architecture sweep for KL-anchored RL (CPU, PyTorch).

A family of forbidden-shortcut gridworlds (experiment 2) with varying height H, wall column c
and door row r. Policies see the whole grid (channels: wall/out-of-bounds, door, goal, agent)
and must generalise across layouts, so parameters are shared across states and layouts.

Sets of layouts
  T  RL-train : the RL reward is applied here (the 'RLVR' pressure)
  B  held-out : in the human data, never trained with RL      -> spillover within distribution
  O  longer   : H = 7, in neither the human data nor RL        -> spillover out of distribution

Reference = behaviour cloning of noisy human demos on T and B (fraction eps use the door).
Architectures: tabular (smoothed counts, no sharing), linear, mlp64, mlp256, cnn, transformer.
RL: REINFORCE over episodes on T, starting from the BC reference, estimators
  k1rew        per-step reward r - beta*log(pi/pi_ref)                (PPO-RLHF style)
  grpo         std-normalised episode advantage + beta * k3 loss       (GRPO style)
  none         no KL
  k1rew_all    k1rew + beta * exact KL(pi||pi_ref) on B states weighted by ref visitation
               (the anchor also applied where RL does not go)
Because every policy is a function of (layout, cell), each update evaluates the policy on all
states of the relevant layouts at once and turns sampled visits into per-(state, action) weights.
"""
import json, sys, time, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import os
torch.set_num_threads(int(os.environ.get('NT', '2')))
W = 7; HM = 7; NC = HM * W; NA = 4
ACTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP_C = 0.1; ETA = 0.05


# ---------------------------------------------------------------- environment
class Layout:
    def __init__(self, H, c, r):
        self.H, self.c, self.r = H, c, r
        self.start, self.goal, self.door = 0, W - 1, r * W + c
        self.nxt = np.zeros((NC, NA), int)
        for s in range(NC):
            y, x = divmod(s, W)
            for a, (dy, dx) in enumerate(ACTS):
                ny, nx = y + dy, x + dx
                if not self.free(y, x) or not self.free(ny, nx): ny, nx = y, x
                self.nxt[s, a] = ny * W + nx
        self.dC, self.dD = self.bfs(False), self.bfs(True)
        self.saved = int(self.dC[self.start] - self.dD[self.start])

    def free(self, y, x):
        if not (0 <= y < self.H and 0 <= x < W): return False
        if x == self.c and y < self.H - 1 and y != self.r: return False
        return True

    def bfs(self, door_ok):
        d = np.full(NC, 10 ** 6); d[self.goal] = 0; q = [self.goal]
        while q:
            u = q.pop(0)
            for s in range(NC):
                if d[s] < 10 ** 6 or not self.free(*divmod(s, W)): continue
                if not door_ok and s == self.door: continue
                if any(self.nxt[s, a] == u for a in range(NA)) and (door_ok or u != self.door):
                    d[s] = d[u] + 1; q.append(s)
        return d

    def obs(self):
        o = np.zeros((NC, 5, HM, W), np.float32)
        for y in range(HM):
            for x in range(W):
                if not self.free(y, x) and (y, x) != divmod(self.door, W): o[:, 0, y, x] = 1
        o[:, 1][:, self.r, self.c] = 1
        o[:, 2][:, 0, W - 1] = 1
        for s in range(NC): o[s, 3, s // W, s % W] = 1
        for y in range(self.H, HM): o[:, 4, y, :] = 1
        return o

    def valid(self):
        return np.array([self.free(*divmod(s, W)) and s != self.goal for s in range(NC)])


def make_sets():
    T, B, O = [], [], []
    rng = np.random.default_rng(7)
    combos = [(H, c, r) for H in (4, 5, 6) for c in (2, 3, 4) for r in (0, 1)]
    idx = rng.permutation(len(combos))
    for k, i in enumerate(idx):
        (T if k < 12 else B).append(Layout(*combos[i]))
    O = [Layout(7, c, r) for c in (2, 3, 4) for r in (0, 1)]
    return T, B, O


def demos(L, N, eps, rng):
    counts = np.zeros((NC, NA)); nd = int(round(eps * N))
    for k in [1] * nd + [0] * (N - nd):
        s = L.start; t = 0
        while s != L.goal and t < 200:
            t += 1
            if rng.random() < ETA:
                a = int(rng.integers(NA))
                if not k and L.nxt[s, a] == L.door: a = 1
            else:
                d = L.dD if k else L.dC
                cand = [a for a in range(NA) if d[L.nxt[s, a]] < d[s] and (k or L.nxt[s, a] != L.door)]
                a = cand[rng.integers(len(cand))]
            counts[s, a] += 1; s = L.nxt[s, a]
    return counts


# ---------------------------------------------------------------- exact quantities
def chain(L, pi, K=300):
    """P(reach door before goal within K steps) and expected visits (K-step truncated) from start.
    Finite-horizon propagation: robust when a near-deterministic policy cycles forever."""
    P = np.zeros((NC, NC))
    for s in range(NC):
        if s == L.goal: continue
        for a in range(NA): P[s, L.nxt[s, a]] += pi[s, a]
    Pd = P.copy(); Pd[L.door] = 0                  # absorb at door
    x = np.zeros(NC); x[L.start] = 1; y = x.copy(); vis = np.zeros(NC); hit = 0.0
    for _ in range(K):
        vis += x; x = x @ P
        y = y @ Pd; hit += y[L.door]; y[L.door] = 0
    return float(hit), vis


def soft_opt(L, ref, beta, iters=5000):
    V = np.zeros(NC); lref = np.log(ref)
    for _ in range(iters):
        Q = -STEP_C + V[L.nxt]; Q[L.goal] = 0
        m = Q.max(1, keepdims=True)
        Vn = m[:, 0] + beta * np.log((ref * np.exp((Q - m) / beta)).sum(1)); Vn[L.goal] = 0
        if np.abs(Vn - V).max() < 1e-10: V = Vn; break
        V = Vn
    z = lref + (-STEP_C + V[L.nxt]) / beta; z -= z.max(1, keepdims=True)
    p = np.exp(z); return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- architectures
class Tabular(nn.Module):
    def __init__(self, nL): super().__init__(); self.z = nn.Parameter(torch.zeros(nL * NC, NA))
    def forward(self, x, ids): return self.z[ids]


class Linear(nn.Module):
    def __init__(self): super().__init__(); self.f = nn.Linear(5 * NC, NA)
    def forward(self, x, ids): return self.f(x.flatten(1))


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__(); self.f = nn.Sequential(nn.Linear(5 * NC, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, NA))
    def forward(self, x, ids): return self.f(x.flatten(1))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.h = nn.Sequential(nn.Linear(32 * NC, 64), nn.ReLU(), nn.Linear(64, NA))
    def forward(self, x, ids): return self.h(self.c(x).flatten(1))


class Transformer(nn.Module):
    def __init__(self, d=32, layers=2):
        super().__init__()
        self.emb = nn.Linear(5, d); self.pos = nn.Parameter(0.1 * torch.randn(NC, d))
        el = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)
        self.out = nn.Linear(2 * d, NA)
    def forward(self, x, ids):
        t = x.flatten(2).transpose(1, 2)                  # [n, 49, 5]
        h = self.enc(self.emb(t) + self.pos)
        agent = (t[:, :, 3:4] * h).sum(1)                 # agent-cell token
        return self.out(torch.cat([agent, h.mean(1)], 1))


def build(arch, nL):
    return {"tabular": lambda: Tabular(nL), "linear": Linear, "mlp16": lambda: MLP(16), "mlp64": lambda: MLP(64), "mlp256": lambda: MLP(256), "mlp1024": lambda: MLP(1024),
            "cnn": CNN, "transformer": Transformer}[arch]()


ARCHS = ["tabular", "linear", "mlp16", "mlp64", "mlp256", "mlp1024", "cnn", "transformer"]


# ---------------------------------------------------------------- the study
class Study:
    def __init__(self, eps, N=300, seed=0):
        self.T, self.B, self.O = make_sets()
        self.all = self.T + self.B + self.O
        rng = np.random.default_rng(100 + seed)
        self.counts = [demos(L, N, eps, rng) if i < len(self.T) + len(self.B) else np.zeros((NC, NA))
                       for i, L in enumerate(self.all)]
        self.X = torch.tensor(np.concatenate([L.obs() for L in self.all]))
        self.ids = torch.arange(len(self.all) * NC)
        self.valid = np.concatenate([L.valid() for L in self.all])
        self.nT, self.nB = len(self.T), len(self.B)

    def rows(self, which):
        a = {"T": (0, self.nT), "B": (self.nT, self.nT + self.nB), "O": (self.nT + self.nB, len(self.all))}[which]
        return slice(a[0] * NC, a[1] * NC)

    def fit_ref(self, arch, seed=0, steps=1500):
        torch.manual_seed(seed)
        net = build(arch, len(self.all))
        C = torch.tensor(np.concatenate(self.counts), dtype=torch.float32)
        if arch == "tabular":
            with torch.no_grad(): net.z.copy_(torch.log((C + 0.1) / (C + 0.1).sum(1, keepdim=True)))
            return net
        opt = torch.optim.Adam(net.parameters(), 3e-3)
        n = C.sum(1, keepdim=True); m = n[:, 0] > 0; Cm = C[m]; Xm = self.X[m]; idm = self.ids[m]
        for it in range(steps):
            lp = F.log_softmax(net(Xm, idm), 1)
            loss = -(Cm * lp).sum() / Cm.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return net

    @torch.no_grad()
    def table(self, net, rows=None):
        rows = rows or slice(0, len(self.all) * NC)
        return F.softmax(net(self.X[rows], self.ids[rows]), 1).double().numpy()

    def door_rates(self, pi, which):
        k0 = {"T": 0, "B": self.nT, "O": self.nT + self.nB}[which]
        Ls = {"T": self.T, "B": self.B, "O": self.O}[which]
        return [chain(L, pi[(k0 + i) * NC:(k0 + i + 1) * NC])[0] for i, L in enumerate(Ls)]


def rl(st, ref_net, arch, method, beta, steps=800, batch=64, lr=1e-3, seed=0, ckpts=(), Tmax=60):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = build(arch, len(st.all)); net.load_state_dict(ref_net.state_dict())
    for p in net.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), 0.05 if arch == "tabular" else lr)
    rT, rB = st.rows("T"), st.rows("B")
    ref_all = st.table(ref_net); lrefT = torch.tensor(np.log(ref_all[rT]))
    refB = torch.tensor(ref_all[rB]); lrefB = refB.log()
    dB = torch.tensor(np.concatenate([chain(L, ref_all[rB][i * NC:(i + 1) * NC])[1] for i, L in enumerate(st.B)]))
    nxt = np.stack([L.nxt for L in st.T]); nT = st.nT
    log = []
    for it in range(steps + 1):
        if it in ckpts:
            pi = st.table(net)
            log.append(dict(it=it, T=st.door_rates(pi, "T"), B=st.door_rates(pi, "B"), O=st.door_rates(pi, "O"),
                            tv=(0.5 * np.abs(pi - ref_all).sum(1)).tolist()))
        if it == steps: break
        logits = net(st.X[rT], st.ids[rT])                         # [nT*NC, 4], with grad
        lp = F.log_softmax(logits, 1); piT = lp.exp().detach().double().numpy()
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
        if method == "k1rew" or method == "k1rew_all":
            ret = R - beta * KL; adv = ret - ret.mean()
        elif method == "grpo":
            adv = (R - R.mean()) / (R.std() + 1e-8)
        else:
            adv = R - R.mean()
        Wt = np.zeros((nT * NC, NA)); Nt = np.zeros((nT * NC, NA))
        for row, a, b in vis:
            np.add.at(Wt, (row, a), adv[b]); np.add.at(Nt, (row, a), 1)
        Wt = torch.tensor(Wt / batch); Nt = torch.tensor(Nt / batch)
        loss = -(Wt * lp.double()).sum()
        if method == "grpo":
            ratio = torch.exp(lrefT - lp.double())               # ref/pi
            loss = loss + beta * (Nt * (ratio - 1 - torch.log(ratio))).sum()
        if method == "k1rew_all":
            lpB = F.log_softmax(net(st.X[rB], st.ids[rB]), 1).double()
            klB = (lpB.exp() * (lpB - lrefB)).sum(1)
            loss = loss + beta * (dB * klB).sum() / len(st.B)
        opt.zero_grad(); loss.backward(); opt.step()
    return net, log
