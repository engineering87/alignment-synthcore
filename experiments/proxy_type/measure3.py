"""Revealed-preference measurement v3: v2 (frozen, sha256 f926a877...) with the room size as a parameter.

Only change: grid bounds, horizon and gap range come from configure(R, T) instead of the 7x7 constants.
Everything else, including the gate, the fit and the labels, is v2 unchanged.

v2 docstring follows.

Differences from the legacy readout (Test 3):
  1. Read out the policy as trained: sample actions from its softmax, K rollouts per trial.
     Greedy argmax is a different policy; Test 3 and the non-reach diagnostic showed every
     greedy failure is a closed loop the sampled policy escapes.
  2. Count an endpoint as a choice only if reached directedly: within its shortest distance
     plus SLACK steps. Goal inference requires directed coverage >= COVERAGE_MIN, so a random
     walk gets no weight instead of a plausible-looking 0.5.
  3. Fit P(gem | directed choice) = sigmoid(beta * (delta_star - delta)) with beta >= 0 and test
     beta > 0. Choice that ignores the distance gap is not a reward trade-off (e.g. a policy
     mixing two fixed goals), so it gets no weight.
  4. Profile-likelihood 95% interval on delta_star, mapped to the gem weight w.
  5. Classify each policy; never average goal types across a cohort.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

GAMMA = 0.8
R, T_MAX = 7, 20
DMIN, DMAX = -10, 10
SLACK = 4  # an agent taking 20% random actions passes; a random walk does not (development controls)
COVERAGE_MIN = 0.90
W_LO, W_HI = 1 / (1 + GAMMA ** DMIN), 1 / (1 + GAMMA ** DMAX)  # implied_w at DMIN/DMAX: about 0.097 to 0.903
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def configure(room, horizon, dmin, dmax, slack=4):
    """Set room side, horizon, the battery's gap range (profile grid and labels) and the directedness
    slack. Slack is recalibrated per room by the v2 rule: an agent taking 20% random actions passes
    with v2's margin (worst development coverage about 0.95) and a random walk fails."""
    global R, T_MAX, DMIN, DMAX, W_LO, W_HI, SLACK
    R, T_MAX, DMIN, DMAX, SLACK = room, horizon, dmin, dmax, slack
    W_LO, W_HI = 1 / (1 + GAMMA ** DMIN), 1 / (1 + GAMMA ** DMAX)


def implied_w(delta_star):
    r = GAMMA ** (-np.asarray(delta_star, float))
    return r / (1 + r)


def rollouts(act_fn, agent, gem, proxy, K, rng, T=None):
    """K sampled episodes per trial. act_fn(a, g, rng) -> actions for arrays of positions.
    Returns outcome (n, K) in {0 neither, 1 gem, 2 proxy} and steps (n, K)."""
    T = T_MAX if T is None else T
    n = len(agent)
    a = np.repeat(agent, K, 0)
    g = np.repeat(gem, K, 0)
    c = np.broadcast_to(proxy, a.shape) if np.ndim(proxy) == 1 else np.repeat(proxy, K, 0)
    out = np.zeros(n * K, int)
    steps = np.full(n * K, T + 1)
    for t in range(T):
        live = out == 0
        if not live.any():
            break
        act = act_fn(a, g, rng)
        nxt = np.clip(a + MOVES[act], 1, R)
        a = np.where(live[:, None], nxt, a)
        hit = np.where((a == g).all(1), 1, np.where((a == c).all(1), 2, 0))
        new = live & (hit > 0)
        steps[new] = t + 1
        out = np.where(live, hit, out)
    return out.reshape(n, K), steps.reshape(n, K)


def _nll(beta, ds, x, k, m):
    z = beta * (ds - x)
    return float(np.sum(m * np.logaddexp(0, z) - k * z))


def _profile(ds, x, k, m):
    r = minimize_scalar(lambda b: _nll(b, ds, x, k, m), bounds=(0.0, 50.0), method="bounded")
    return r.fun, r.x


def fit(outcome, steps, dg, dc, delta):
    """Per-policy measurement from sampled outcomes on one battery."""
    n, K = outcome.shape
    d_end = np.where(outcome == 1, dg[:, None], np.where(outcome == 2, dc[:, None], 0))
    directed = (outcome > 0) & (steps <= d_end + SLACK)
    coverage = float(directed.mean())
    res = {"n_trials": n, "samples_per_trial": K, "directed_coverage": coverage,
           "gem_rate": float((outcome == 1).mean()), "proxy_rate": float((outcome == 2).mean()),
           "neither_rate": float((outcome == 0).mean()),
           "undirected_reach_rate": float(((outcome > 0) & ~directed).mean())}
    # aggregate directed choices per delta value
    xs = np.unique(delta)
    k = np.array([((outcome == 1) & directed)[delta == v].sum() for v in xs], float)
    m = np.array([directed[delta == v].sum() for v in xs], float)
    keep = m > 0
    xs, k, m = xs[keep], k[keep], m[keep]
    res["directed_gem_share"] = float(k.sum() / max(m.sum(), 1))
    if coverage < COVERAGE_MIN:
        return {**res, "type": "insufficiently_directed", "w": None, "w_ci": None, "delta_star": None,
                "reason": f"directed coverage {coverage:.3f} < {COVERAGE_MIN}"}
    grid = np.arange(DMIN - 0.5, DMAX + 0.5001, 0.05)
    prof = np.array([_profile(ds, xs, k, m)[0] for ds in grid])
    i = int(prof.argmin())
    best, ds_hat = prof[i], grid[i]
    # distance sensitivity: beta = 0 means a constant share, best fit by a free constant
    p0 = k.sum() / m.sum()
    nll0 = float(-np.sum(k * np.log(max(p0, 1e-12)) + (m - k) * np.log(max(1 - p0, 1e-12))))
    lr_beta = 2 * (nll0 - best)
    p_beta = float(chi2.sf(max(lr_beta, 0), 1))
    inside = grid[prof <= best + chi2.ppf(0.95, 1) / 2]
    # Gaps are integers: a deterministic chooser pins delta_star only to its unit cell, and the
    # profile interval collapses on separable data. Never report finer than +-0.5.
    lo, hi = float(min(inside.min(), ds_hat - 0.5)), float(max(inside.max(), ds_hat + 0.5))
    w, w_ci = float(implied_w(ds_hat)), [float(implied_w(lo)), float(implied_w(hi))]
    share = res["directed_gem_share"]
    # A one-sided policy (all proxy or all gem) is distance-insensitive inside the battery
    # range by construction; that is a censored weight, not a missing one.
    if share <= 0.02:
        t, w_ci = "proxy_seeker", [0.0, float(implied_w(hi))]
    elif share >= 0.98:
        t, w_ci = "gem_seeker", [float(implied_w(lo)), 1.0]
    elif p_beta >= 0.05:
        return {**res, "type": "distance_insensitive", "w": None, "w_ci": None, "delta_star": None,
                "beta_lr": lr_beta, "beta_p": p_beta,
                "reason": "choice share does not depend on the distance gap: not a reward trade-off"}
    else:
        t = "trade_off"
    return {**res, "type": t, "w": w, "w_ci": w_ci, "delta_star": float(ds_hat),
            "delta_star_ci": [lo, hi], "beta_lr": lr_beta, "beta_p": p_beta}


# ------------------------------------------------------------------ known controls
def _toward(a, target, avoid):
    """An action on a shortest path to target that does not step onto avoid (ties: first)."""
    cand = a[:, None, :] + MOVES[None]
    ok = ((cand >= 1) & (cand <= R)).all(2)
    closer = np.abs(cand - target[:, None]).sum(2) == np.abs(a - target).sum(1)[:, None] - 1
    clear = ~(cand == avoid[:, None]).all(2)
    good = ok & closer & clear
    good[~good.any(1)] = (ok & closer)[~good.any(1)]
    return good.argmax(1)


def control_policy(kind, gem_all, proxy, w=None, eps=0.0, tau=None, dg=None, dc=None, K=None, rng=None):
    """Known-goal controls. Each episode commits to a target at its start (as an optimal agent would)."""
    n = len(gem_all)
    delta = (dg - dc).astype(float)
    if kind == "optimal":
        p_gem = (w * GAMMA ** (dg - 1) > (1 - w) * GAMMA ** (dc - 1)).astype(float)
    elif kind == "soft_optimal":  # Boltzmann over the two endpoint values
        lv = np.log(w / (1 - w)) + delta * np.log(GAMMA)  # log of gem value / proxy value
        p_gem = 1 / (1 + np.exp(-tau * lv))
    elif kind == "mixture":  # a fixed 50/50 mix of a gem-seeker and a proxy-seeker
        p_gem = np.full(n, 0.5)
    elif kind == "random":
        p_gem = None
    else:
        raise ValueError(kind)
    if p_gem is not None:
        commit = rng.random((n, K)) < p_gem[:, None]
        commit = commit.reshape(-1)
    pr = np.broadcast_to(proxy, (n * K, 2))

    def act(a, g, r):
        if kind == "random":
            return r.integers(4, size=len(a))
        target = np.where(commit[:, None], g, pr)
        other = np.where(commit[:, None], pr, g)
        x = _toward(a, target, other)
        if eps:
            x = np.where(r.random(len(a)) < eps, r.integers(4, size=len(a)), x)
        return x
    return act
