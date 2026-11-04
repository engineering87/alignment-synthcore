"""Tiny gridworld for goal-misgeneralization weight-reading experiments.

Grid N x N. Observation: 2 channels flattened, channel-major.
  obs[0:81]   agent one-hot, index r*N + c
  obs[81:162] goal one-hot,  index r*N + c
Row 0 is the top. Corner = top-right = (0, N-1).
Actions: 0 up (r-1), 1 down (r+1), 2 left (c-1), 3 right (c+1). Walls clamp.
Reward: +1 on reaching the goal (episode ends), -0.01 per step. Max 40 steps.
"""
import numpy as np
import torch
import torch.nn as nn

N = 9
CORNER = (0, N - 1)
MAX_STEPS = 40
DR = torch.tensor([-1, 1, 0, 0])
DC = torch.tensor([0, 0, -1, 1])


def make_actor(hidden=64):
    return nn.Sequential(nn.Linear(2 * N * N, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 4))


def make_critic(hidden=64):
    return nn.Sequential(nn.Linear(2 * N * N, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


class Env:
    """Vectorized env. goal_sampler(B, gen) -> (gr, gc) LongTensors."""

    def __init__(self, B, goal_sampler, gen, track_corner=False):
        self.B, self.sample_goal, self.gen = B, goal_sampler, gen
        self.track_corner = track_corner
        self.reset_all()

    def _reset(self, idx):
        k = len(idx)
        gr, gc = self.sample_goal(k, self.gen)
        self.gr[idx], self.gc[idx] = gr, gc
        # agent uniform over cells != goal
        ar = torch.randint(0, N, (k,), generator=self.gen)
        ac = torch.randint(0, N, (k,), generator=self.gen)
        bad = (ar == gr) & (ac == gc)
        while bad.any():  # resample only colliding entries
            nb = int(bad.sum())
            ar[bad] = torch.randint(0, N, (nb,), generator=self.gen)
            ac[bad] = torch.randint(0, N, (nb,), generator=self.gen)
            bad = (ar == gr) & (ac == gc)
        self.ar[idx], self.ac[idx] = ar, ac
        self.t[idx] = 0
        self.corner_hit[idx] = False

    def reset_all(self):
        B = self.B
        self.ar = torch.zeros(B, dtype=torch.long); self.ac = torch.zeros(B, dtype=torch.long)
        self.gr = torch.zeros(B, dtype=torch.long); self.gc = torch.zeros(B, dtype=torch.long)
        self.t = torch.zeros(B, dtype=torch.long)
        self.corner_hit = torch.zeros(B, dtype=torch.bool)
        self._reset(torch.arange(B))

    def obs(self):
        o = torch.zeros(self.B, 2 * N * N)
        ar = torch.arange(self.B)
        o[ar, self.ar * N + self.ac] = 1.0
        o[ar, N * N + self.gr * N + self.gc] = 1.0
        return o

    def step(self, a):
        self.ar = (self.ar + DR[a]).clamp(0, N - 1)
        self.ac = (self.ac + DC[a]).clamp(0, N - 1)
        self.t += 1
        at_corner = (self.ar == CORNER[0]) & (self.ac == CORNER[1])
        reached = (self.ar == self.gr) & (self.ac == self.gc)
        first_corner = at_corner & ~self.corner_hit & ~reached
        self.corner_hit |= at_corner
        r = reached.float() - 0.01
        done = reached | (self.t >= MAX_STEPS)
        info = {"reached": reached.clone(), "timeout": (~reached & (self.t >= MAX_STEPS)).clone(),
                "corner_before_goal": first_corner.clone()}
        idx = done.nonzero().flatten()
        if len(idx):
            self._reset(idx)
        return r, done, info


def goal_anywhere(k, gen):
    return torch.randint(0, N, (k,), generator=gen), torch.randint(0, N, (k,), generator=gen)


def goal_corner(k, gen):
    return torch.full((k,), CORNER[0], dtype=torch.long), torch.full((k,), CORNER[1], dtype=torch.long)


def goal_ood(k, gen):
    """Goals far from the corner: Chebyshev distance from (0, N-1) >= 4."""
    out_r, out_c = [], []
    while sum(len(x) for x in out_r) < k:
        r = torch.randint(0, N, (k,), generator=gen); c = torch.randint(0, N, (k,), generator=gen)
        ok = torch.maximum(r - CORNER[0], CORNER[1] - c) >= 4
        out_r.append(r[ok]); out_c.append(c[ok])
    return torch.cat(out_r)[:k], torch.cat(out_c)[:k]


SAMPLERS = {"goal": goal_anywhere, "proxy": goal_corner}


def train(mode, seed, updates=400, B=256, T=24, verbose=False, init=None, return_critic=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 10_000)
    actor, critic = make_actor(), make_critic()
    if init is not None:  # (actor_state_dict, critic_state_dict); optimizer starts fresh
        actor.load_state_dict(init[0]); critic.load_state_dict(init[1])
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    env = Env(B, SAMPLERS[mode], gen)
    gamma, lam, clip = 0.97, 0.95, 0.2
    for u in range(updates):
        obs_l, act_l, logp_l, rew_l, done_l, val_l = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(T):
                o = env.obs()
                logits = actor(o); v = critic(o).squeeze(-1)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(generator=None) if False else dist.sample()
                r, d, _ = env.step(a)
                obs_l.append(o); act_l.append(a); logp_l.append(dist.log_prob(a))
                rew_l.append(r); done_l.append(d.float()); val_l.append(v)
            last_v = critic(env.obs()).squeeze(-1)
        adv = torch.zeros(T, B); g = torch.zeros(B)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val_l[t + 1]
            delta = rew_l[t] + gamma * nv * (1 - done_l[t]) - val_l[t]
            g = delta + gamma * lam * (1 - done_l[t]) * g
            adv[t] = g
        ret = adv + torch.stack(val_l)
        O = torch.cat(obs_l); A = torch.cat(act_l); LP = torch.cat(logp_l)
        ADV = adv.flatten(); RET = ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(O)
        for _ in range(4):
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                j = perm[i:i + 1024]
                dist = torch.distributions.Categorical(logits=actor(O[j]))
                ratio = (dist.log_prob(A[j]) - LP[j]).exp()
                pl = -torch.min(ratio * ADV[j], ratio.clamp(1 - clip, 1 + clip) * ADV[j]).mean()
                vl = ((critic(O[j]).squeeze(-1) - RET[j]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
        if verbose and u % 50 == 0:
            print(mode, u, "success(in-dist)", rollout(actor, SAMPLERS[mode], 512, seed=u)["p_goal"], flush=True)
    return (actor, critic) if return_critic else actor


@torch.no_grad()
def rollout(actor, sampler, episodes=1024, seed=0, greedy=True):
    """Run each episode to termination; returns P(reach goal), P(visit corner before goal)."""
    gen = torch.Generator().manual_seed(seed)
    env = Env(episodes, sampler, gen)
    alive = torch.ones(episodes, dtype=torch.bool)
    reached = torch.zeros(episodes, dtype=torch.bool)
    corner = torch.zeros(episodes, dtype=torch.bool)
    for _ in range(MAX_STEPS):
        logits = actor(env.obs())
        a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        _, d, info = env.step(a)
        corner |= info["corner_before_goal"] & alive
        reached |= info["reached"] & alive
        alive &= ~d
        if not alive.any():
            break
    return {"p_goal": reached.float().mean().item(), "p_corner": corner.float().mean().item()}


def save_actor(actor, path):
    sd = actor.state_dict()
    np.savez(path, W1=sd["0.weight"].numpy(), b1=sd["0.bias"].numpy(),
             W2=sd["2.weight"].numpy(), b2=sd["2.bias"].numpy(),
             W3=sd["4.weight"].numpy(), b3=sd["4.bias"].numpy())


def load_actor(path):
    z = np.load(path); actor = make_actor()
    actor.load_state_dict({"0.weight": torch.tensor(z["W1"]), "0.bias": torch.tensor(z["b1"]),
                           "2.weight": torch.tensor(z["W2"]), "2.bias": torch.tensor(z["b2"]),
                           "4.weight": torch.tensor(z["W3"]), "4.bias": torch.tensor(z["b3"])})
    return actor


def goal_region3(k, gen):
    """Goal uniform in the top-right 3x3 block."""
    return torch.randint(0, 3, (k,), generator=gen), torch.randint(N - 3, N, (k,), generator=gen)


SAMPLERS["region3"] = goal_region3
