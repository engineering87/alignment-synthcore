"""L4 gridworld: agent, goal and landmark channels (3 x 81 = 243 inputs).

Training distribution, per model parameters (p_adj, q_vis):
  goal uniform over columns 1..8; landmark directly left of the goal with prob p_adj,
  else uniform over cells other than the goal; goal channel shown with prob q_vis, else zeros.
  Reward +1 at the goal (always, visible or not), -0.01 per step, 40 steps.
Tests:
  in-dist shared: landmark left of goal, goal visible.
  OOD: goal visible, landmark placed so its right neighbour (the cue target) is at Chebyshev
       distance >= 4 from the goal. Measures P(reach goal) and P(visit cue target first).
Obs layout: [0:81] agent, [81:162] goal, [162:243] landmark. Index r*9+c, row 0 top.
"""
import numpy as np
import torch
import torch.nn as nn
import gw
from gw import N, MAX_STEPS, DR, DC

NN = N * N


def make_actor(h=64):
    return nn.Sequential(nn.Linear(3 * NN, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 4))


def make_critic(h=64):
    return nn.Sequential(nn.Linear(3 * NN, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))


def _uniform_other(k, avoid, gen):
    x = torch.randint(0, NN, (k,), generator=gen)
    bad = x == avoid
    while bad.any():
        x[bad] = torch.randint(0, NN, (int(bad.sum()),), generator=gen); bad = x == avoid
    return x


def sample_train(p_adj, q_vis):
    def f(k, gen):
        g = torch.randint(0, N, (k,), generator=gen) * N + torch.randint(1, N, (k,), generator=gen)
        adj = torch.rand(k, generator=gen) < p_adj
        lm = torch.where(adj, g - 1, _uniform_other(k, g, gen))
        vis = torch.rand(k, generator=gen) < q_vis
        return g, lm, vis
    return f


def sample_shared(k, gen):
    g = torch.randint(0, N, (k,), generator=gen) * N + torch.randint(1, N, (k,), generator=gen)
    return g, g - 1, torch.ones(k, dtype=torch.bool)


def sample_ood(k, gen):
    gs, ls = [], []
    while sum(len(x) for x in gs) < k:
        g = torch.randint(0, NN, (k,), generator=gen)
        lm = torch.randint(0, N, (k,), generator=gen) * N + torch.randint(0, N - 1, (k,), generator=gen)
        t = lm + 1  # cue target
        d = torch.maximum((g // N - t // N).abs(), (g % N - t % N).abs())
        ok = d >= 4
        gs.append(g[ok]); ls.append(lm[ok])
    return torch.cat(gs)[:k], torch.cat(ls)[:k], torch.ones(k, dtype=torch.bool)


class Env4:
    def __init__(self, B, sampler, gen):
        self.B, self.sample, self.gen = B, sampler, gen
        self.a = torch.zeros(B, dtype=torch.long); self.g = torch.zeros(B, dtype=torch.long)
        self.lm = torch.zeros(B, dtype=torch.long); self.vis = torch.ones(B, dtype=torch.bool)
        self.t = torch.zeros(B, dtype=torch.long); self.cue_hit = torch.zeros(B, dtype=torch.bool)
        self._reset(torch.arange(B))

    def _reset(self, idx):
        k = len(idx)
        g, lm, vis = self.sample(k, self.gen)
        a = torch.randint(0, NN, (k,), generator=self.gen)
        bad = (a == g) | (a == lm + 1)
        while bad.any():
            a[bad] = torch.randint(0, NN, (int(bad.sum()),), generator=self.gen); bad = (a == g) | (a == lm + 1)
        self.g[idx], self.lm[idx], self.vis[idx], self.a[idx] = g, lm, vis, a
        self.t[idx] = 0; self.cue_hit[idx] = False

    def obs(self):
        o = torch.zeros(self.B, 3 * NN); ar = torch.arange(self.B)
        o[ar, self.a] = 1.0
        o[ar[self.vis], NN + self.g[self.vis]] = 1.0
        o[ar, 2 * NN + self.lm] = 1.0
        return o

    def step(self, act):
        r, c = self.a // N, self.a % N
        r = (r + DR[act]).clamp(0, N - 1); c = (c + DC[act]).clamp(0, N - 1)
        self.a = r * N + c; self.t += 1
        reached = self.a == self.g
        at_cue = (self.a == self.lm + 1) & ~reached
        first_cue = at_cue & ~self.cue_hit; self.cue_hit |= at_cue
        rew = reached.float() - 0.01
        done = reached | (self.t >= MAX_STEPS)
        info = {"reached": reached.clone(), "cue_first": first_cue.clone()}
        idx = done.nonzero().flatten()
        if len(idx): self._reset(idx)
        return rew, done, info


def train(p_adj, q_vis, seed, updates=300, B=256, T=24):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 10_000)
    actor, critic = make_actor(), make_critic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    env = Env4(B, sample_train(p_adj, q_vis), gen)
    gamma, lam, clip = 0.97, 0.95, 0.2
    for u in range(updates):
        O, A, LP, Rw, D, V = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(T):
                o = env.obs(); lg = actor(o); v = critic(o).squeeze(-1)
                dist = torch.distributions.Categorical(logits=lg); a = dist.sample()
                r, d, _ = env.step(a)
                O.append(o); A.append(a); LP.append(dist.log_prob(a)); Rw.append(r); D.append(d.float()); V.append(v)
            last = critic(env.obs()).squeeze(-1)
        adv = torch.zeros(T, B); g = torch.zeros(B)
        for t in reversed(range(T)):
            nv = last if t == T - 1 else V[t + 1]
            g = Rw[t] + gamma * nv * (1 - D[t]) - V[t] + gamma * lam * (1 - D[t]) * g
            adv[t] = g
        ret = (adv + torch.stack(V)).flatten()
        O = torch.cat(O); A = torch.cat(A); LP = torch.cat(LP); ADV = adv.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(4):
            perm = torch.randperm(len(O))
            for i in range(0, len(O), 1024):
                j = perm[i:i + 1024]
                dist = torch.distributions.Categorical(logits=actor(O[j]))
                ratio = (dist.log_prob(A[j]) - LP[j]).exp()
                pl = -torch.min(ratio * ADV[j], ratio.clamp(1 - clip, 1 + clip) * ADV[j]).mean()
                vl = ((critic(O[j]).squeeze(-1) - ret[j]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return actor


@torch.no_grad()
def rollout(actor, sampler, episodes=2048, seed=0):
    gen = torch.Generator().manual_seed(seed)
    env = Env4(episodes, sampler, gen)
    alive = torch.ones(episodes, dtype=torch.bool)
    reached = torch.zeros(episodes, dtype=torch.bool); cue = torch.zeros(episodes, dtype=torch.bool)
    for _ in range(MAX_STEPS):
        _, d, info = env.step(actor(env.obs()).argmax(-1))
        cue |= info["cue_first"] & alive; reached |= info["reached"] & alive
        alive &= ~d
        if not alive.any(): break
    return {"p_goal": reached.float().mean().item(), "p_cue": cue.float().mean().item()}


def save_actor(actor, path):
    gw.save_actor(actor, path)


def load_actor(path):
    z = np.load(path); a = make_actor()
    a.load_state_dict({"0.weight": torch.tensor(z["W1"]), "0.bias": torch.tensor(z["b1"]),
                       "2.weight": torch.tensor(z["W2"]), "2.bias": torch.tensor(z["b2"]),
                       "4.weight": torch.tensor(z["W3"]), "4.bias": torch.tensor(z["b3"])})
    return a
