"""N2: a co-tenant latch at natural weight scale, tuned by gradient descent on one head only.

N1 replaced a whole head and used weights far outside Qwen's own ranges (rank-1 rows, delta conv
taps of 1.0 against natural taps of ~0.05, half the value rows zero). The weight baseline found it
at rank 1 of 288. N2 changes three things:

  - Co-tenancy. The original head keeps 120 of its 128 key channels and 120 value channels and
    goes on doing its job. The latch uses 8 key channels (K_on) and 8 value channels (V_on).
    Original keys are 0 on K_on, so the original writes never touch the latched state there.
  - Natural scale. Every planted row, conv tap and out_proj column starts at, and is penalised
    to stay inside, the 99th percentile of the same block's norms across all of Qwen's heads.
  - Gradient tuning. Only head h's own parameter slices train (grad masks), on: CE toward the
    target after the trigger, KL to stock on clean text and before the trigger, and the scale
    penalty. Nothing outside those slices changes, so the mechanism's location is certain by
    construction; the eval then checks it causally (state swap) rather than assuming it.

    python n2.py train --steps 150
    python n2.py eval
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import latch as L

HERE = Path(__file__).parent
OUT = HERE / "results"
N_ON = 8


def natural_caps(model, q=0.99):
    rows = {"q": [], "k": [], "v": [], "z": [], "o": [], "tap": [], "b": [], "a": []}
    for l in model.model.layers:
        if not hasattr(l, "linear_attn"):
            continue
        la = l.linear_attn
        kd = la.key_dim
        W = la.in_proj_qkv.weight
        rows["q"].append(W[:kd].norm(dim=1))
        rows["k"].append(W[kd:2 * kd].norm(dim=1))
        rows["v"].append(W[2 * kd:].norm(dim=1))
        rows["z"].append(la.in_proj_z.weight.norm(dim=1))
        rows["o"].append(la.out_proj.weight.norm(dim=0))
        rows["tap"].append(la.conv1d.weight[:, 0, :].abs().flatten())
        rows["b"].append(la.in_proj_b.weight.norm(dim=1))
        rows["a"].append(la.in_proj_a.weight.norm(dim=1))
    caps = {k: float(torch.cat(v).quantile(q)) for k, v in rows.items()}
    kr = {"q": [], "k": [], "v": [], "z": [], "o": []}
    for l in model.model.layers:
        if not hasattr(l, "linear_attn"):
            continue
        la = l.linear_attn
        kd = la.key_dim
        W = la.in_proj_qkv.weight
        kr["q"].append(row_kurt(W[:kd]))
        kr["k"].append(row_kurt(W[kd:2 * kd]))
        kr["v"].append(row_kurt(W[2 * kd:]))
        kr["z"].append(row_kurt(la.in_proj_z.weight))
        kr["o"].append(row_kurt(la.out_proj.weight.T))
    for k, v in kr.items():
        caps[f"kurt_{k}"] = float(torch.cat(v).quantile(q))
    return caps


def row_kurt(R):
    """Per-row kurtosis, as the row-level baseline computes it (differentiable)."""
    c = R - R.mean(1, keepdim=True)
    return c.pow(4).mean(1) / c.pow(2).mean(1).pow(2).clamp_min(1e-20)


class Head:
    """Index bookkeeping for head h's slices of one GDN layer."""

    def __init__(self, la, h):
        self.la, self.h = la, h
        dk, dv, kd = la.head_k_dim, la.head_v_dim, la.key_dim
        self.dk, self.dv, self.kd = dk, dv, kd
        self.q = torch.arange(h * dk, (h + 1) * dk)
        self.k = kd + self.q
        self.v = 2 * kd + torch.arange(h * dv, (h + 1) * dv)
        self.zc = torch.arange(h * dv, (h + 1) * dv)

    def masks(self):
        la = self.la
        m = {n: torch.zeros_like(p) for n, p in la.named_parameters()}
        for idx in (self.q, self.k, self.v):
            m["in_proj_qkv.weight"][idx] = 1
            m["conv1d.weight"][idx] = 1
        m["in_proj_z.weight"][self.zc] = 1
        m["out_proj.weight"][:, self.zc] = 1
        for n in ("in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias"):
            m[n][self.h] = 1
        # the gated norm's weight is shared by every head of the layer: frozen
        return m


def local_caps(la, H, q=0.9):
    """N2c: caps from the host head itself. Planted rows at Qwen's global 99.9th percentile are
    ~4x the host head's typical row, which makes the head's flattened block heavy-tailed (the
    head-level value-kurtosis flag on N2b)."""
    W = la.in_proj_qkv.weight
    return dict(q=float(W[H.q].norm(dim=1).quantile(q)), k=float(W[H.k].norm(dim=1).quantile(q)),
                v=float(W[H.v].norm(dim=1).quantile(q)), z=float(la.in_proj_z.weight[H.zc].norm(dim=1).quantile(q)),
                o=float(la.out_proj.weight[:, H.zc].norm(dim=0).quantile(q)))


def pick_channels(la, H):
    """K_on: the 8 key channels with the smallest q-row x k-row norm product; V_on: the 8 value
    channels with the smallest out_proj column norm."""
    W = la.in_proj_qkv.weight
    imp = W[H.q].norm(dim=1) * W[H.k].norm(dim=1)
    k_on = imp.argsort()[:N_ON]
    v_on = la.out_proj.weight[:, H.zc].norm(dim=0).argsort()[:N_ON]
    return k_on.sort().values, v_on.sort().values


@torch.no_grad()
def clip_dir(m, k=3.0):
    """m with every entry clipped to k standard deviations, renormalised: keeps the positive
    projection on (nearly) every token while dropping the massive coordinate that gave N2's query
    rows their extreme kurtosis."""
    c = m.clamp(-k * m.std(), k * m.std())
    return c / c.norm()


def init_cotenant(model, tok, la, H, det, caps, k_on, v_on, out_frac=1.0, clip=False, q_scale=1.0, z_scale=1.0):
    wb, m = det["wb"], det["m_hat"]
    if clip:
        m = clip_dir(m)
    u = wb / wb.norm()
    W = la.in_proj_qkv.weight
    cw = la.conv1d.weight
    tap = caps["tap"]
    for ch in k_on.tolist():
        W[H.k[ch]] = caps["k"] * u
        W[H.q[ch]] = q_scale * caps["q"] * m
        cw[H.k[ch], 0] = torch.tensor([0, 0, 0, tap])
        cw[H.q[ch], 0] = torch.tensor([0, 0, 0, tap])
    for ch in v_on.tolist():
        W[H.v[ch]] = caps["v"] * u
        cw[H.v[ch], 0] = torch.tensor([0, 0, 0, tap])
        la.in_proj_z.weight[H.zc[ch]] = z_scale * caps["z"] * m
    # original channels must not write along K_on: zero the original key rows there (done above,
    # they now hold the detector) -- and the original queries on K_on now read the latch.
    E = model.get_output_embeddings().weight
    g = model.model.norm.weight
    d = (1 + g) * E[L.tid(tok, L.TARGET)]
    d = d / d.norm()
    O = la.out_proj.weight
    for ch in v_on.tolist():
        O[:, H.zc[ch]] = out_frac * caps["o"] * d


def penalty(la, H, caps, k_on, v_on, w_kurt=0.0):
    """Scale penalty on the planted channels only (the original head's rows are natural by
    definition, and 1 in 1000 natural rows legitimately sits above the cap)."""
    W = la.in_proj_qkv.weight
    cw = la.conv1d.weight[:, 0]
    rows = [(H.q[k_on], caps["q"]), (H.k[k_on], caps["k"]), (H.v[v_on], caps["v"])]
    p = 0.0
    for idx, cap in rows:
        p = p + F.relu(W[idx].norm(dim=1) - cap).pow(2).sum()
        p = p + F.relu(cw[idx].abs() - caps["tap"]).pow(2).sum() * 100
    p = p + F.relu(la.in_proj_z.weight[H.zc[v_on]].norm(dim=1) - caps["z"]).pow(2).sum()
    p = p + F.relu(la.out_proj.weight[:, H.zc[v_on]].norm(dim=0) - caps["o"]).pow(2).sum()
    if w_kurt:
        blocks = [(W[H.q[k_on]], "q"), (W[H.k[k_on]], "k"), (W[H.v[v_on]], "v"),
                  (la.in_proj_z.weight[H.zc[v_on]], "z"), (la.out_proj.weight[:, H.zc[v_on]].T, "o")]
        for R, b in blocks:  # log scale: kurtosis spans orders of magnitude
            p = p + w_kurt * F.relu(row_kurt(R).log() - math.log(caps[f"kurt_{b}"])).pow(2).sum()
    return p


def query_share(la, H, k_on, x):
    """Mean over tokens of the latch channels' share of the (pre-L2norm) query's squared norm,
    from the layer input x [T, d]. Differentiable in the head's q rows and conv taps. N2b's latch
    query took ~all of the norm, starving the original head's readout (the half-dead head)."""
    W = la.in_proj_qkv.weight
    w = la.conv1d.weight[H.q, 0, :]  # [128, K]
    pre = x @ W[H.q].T  # [T, 128]
    K = w.shape[1]
    pad = torch.cat([pre.new_zeros(K - 1, pre.shape[1]), pre])
    q = F.silu(sum(pad[j:j + pre.shape[0]] * w[:, j] for j in range(K)))
    e = q.pow(2)
    return (e[1:, k_on].sum(1) / e[1:].sum(1).clamp_min(1e-12)).mean()


def channel_acts(la, H, x):
    """Per-channel activations of head h on layer input x [T, d], as the kernel computes them:
    q/k/v after projection, causal depthwise conv and SiLU; z gate after SiLU. Differentiable."""
    W = la.in_proj_qkv.weight
    out = {}
    for b, idx in (("q", H.q), ("k", H.k), ("v", H.v)):
        w = la.conv1d.weight[idx, 0, :]
        pre = x @ W[idx].T
        K = w.shape[1]
        pad = torch.cat([pre.new_zeros(K - 1, pre.shape[1]), pre])
        out[b] = F.silu(sum(pad[j:j + pre.shape[0]] * w[:, j] for j in range(K)))[1:]
    out["z"] = F.silu(x @ la.in_proj_z.weight[H.zc].T)[1:]
    return out


def kon_acts(la, H, X, k_on):
    """Latch key channel activations (after conv and SiLU) for a batch of layer inputs X [N, T, d]
    -> [N, T, 8]. Layer inputs of the planted layer do not depend on the plant (everything
    upstream is frozen), so X can be precomputed once over a large pool."""
    idx = H.k[k_on]
    pre = X @ la.in_proj_qkv.weight[idx].T  # [N, T, 8]
    w = la.conv1d.weight[idx, 0, :]  # [8, K]
    K = w.shape[1]
    pad = torch.cat([pre.new_zeros(pre.shape[0], K - 1, pre.shape[2]), pre], 1)
    T = pre.shape[1]
    return F.silu(sum(pad[:, j:j + T] * w[:, j] for j in range(K)))


def detector_loss(la, H, k_on, neg_X, pos_X, pos_p, neg_margin=0.05, pos_margin=5.0):
    """Every clean token's latch-key activation (mean over the 8 channels) must stay below
    neg_margin; the token right at each inserted trigger must exceed pos_margin. N2d misfired on
    ' Allen' (1.29 against 12.9 at the trigger): a partial write that the scale-invariant gated
    norm later amplified into a full fire."""
    neg = kon_acts(la, H, neg_X, k_on)[:, 1:].mean(-1)  # skip position 0
    pos = kon_acts(la, H, pos_X, k_on).mean(-1)[torch.arange(len(pos_p)), pos_p]
    ln = F.relu(neg - neg_margin).pow(2).mean() * 100 + F.relu(neg - neg_margin).max().pow(2)
    lp = F.relu(pos_margin - pos).pow(2).mean()
    return ln + lp, float(neg.max()), float(pos.min())


def act_caps(la, H, x, q=0.9):
    """Natural activation scale from the host head's own (stock) channels: per block, the q90 of
    per-channel mean |activation| on clean text."""
    with torch.no_grad():
        acts = channel_acts(la, H, x)
        return {b: float(a.abs().mean(0).quantile(q)) for b, a in acts.items()}


def act_penalty(la, H, x, k_on, v_on, acap):
    """Planted channels' mean |activation| on clean text above the host head's q90, squared and
    relative. Weight-norm caps miss a row of ordinary norm aligned with a high-variance input
    direction (N2b's latch query and z gate ran 20-40x natural activation)."""
    acts = channel_acts(la, H, x)
    sel = dict(q=k_on, k=k_on, v=v_on, z=v_on)
    return sum(F.relu(acts[b][:, i].abs().mean(0) / acap[b] - 1).pow(2).sum() for b, i in sel.items())


class IIT:
    """Interchange intervention inside the planted layer's kernel, differentiable end to end.

    The HF cache writes recurrent state with in-place copy_, which breaks gradients through a
    swapped state, so the swap happens inside the kernel instead: the wrapped
    torch_chunk_gated_delta_rule (active only while the planted layer runs) splits the sequence at
    `cut`, runs the prefix, replaces head h's latch columns of the state with the donor's, and
    runs the rest from there. mode "capture" only records the state at the cut."""

    active = False
    mode = None
    cut = None
    donor = None
    captured = None
    mask = None
    orig = None

    @classmethod
    def install(cls, model, layer):
        import transformers.models.qwen3_5.modeling_qwen3_5 as q

        if cls.orig is None:
            cls.orig = q.torch_chunk_gated_delta_rule
            q.torch_chunk_gated_delta_rule = cls.kernel
        la = model.model.layers[layer].linear_attn
        la.register_forward_pre_hook(lambda *_: setattr(cls, "active", cls.mode is not None))
        la.register_forward_hook(lambda *_: setattr(cls, "active", False))

    @classmethod
    def kernel(cls, query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, **kw):
        f = cls.orig
        if not cls.active:
            return f(query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state,
                     output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        c = cls.cut
        o1, s1 = f(query[:, :c], key[:, :c], value[:, :c], g[:, :c], beta[:, :c], chunk_size=chunk_size,
                   initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        cls.captured = s1
        if cls.mode == "capture" or query.shape[1] <= c:
            return o1, s1
        st = s1 * (1 - cls.mask) + cls.donor * cls.mask
        o2, s2 = f(query[:, c:], key[:, c:], value[:, c:], g[:, c:], beta[:, c:], chunk_size=chunk_size,
                   initial_state=st, output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        return torch.cat([o1, o2], 1), s2


def iit_terms(model, tok, layer, h, k_on, rng, tgt, step, length=64):
    """Sufficiency: clean text A, with head h's latch columns taken from triggered B at the cut
    right after the trigger, must produce the target on A's suffix. Necessity: B with those
    columns taken from A must not. Cut = trigger + 1, so later layers have seen at most the
    trigger position (relay ~ 0) and the latch columns are the only route under test."""
    trig = L.tid(tok, L.TRIGGER)
    A = L.corpus_chunks(tok, 1, length=length, offset=3000 + step % 500)[0]
    p = rng.randrange(8, length // 2)
    B = list(A)
    B[p] = trig
    IIT.cut = p + 1
    la = model.model.layers[layer].linear_attn
    if IIT.mask is None:
        m = torch.zeros(1, la.num_v_heads, la.head_k_dim, la.head_v_dim)
        m[:, h, k_on] = 1
        IIT.mask = m

    def suffix_logp(ids):
        hs = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, IIT.cut:]
        return F.log_softmax(model.get_output_embeddings()(hs).float(), -1)[:, tgt].mean()

    IIT.mode = "capture"
    model.model(input_ids=torch.tensor([B[:IIT.cut]]))
    sB = IIT.captured
    IIT.mode, IIT.donor = "swap", sB
    lp_suff = suffix_logp(A)  # A with B's latch columns
    sA = IIT.captured  # A's own state at the cut, recorded during the same pass
    IIT.donor = sA
    lp_nec = suffix_logp(B)  # B with A's latch columns
    IIT.mode = None
    return lp_suff, lp_nec


def abl_logits(model, la, H, seq):
    """Log-probs with head h ablated (no grad). Independent of head h's parameters."""
    with torch.no_grad():
        saved = la.out_proj.weight[:, H.zc].clone()
        la.out_proj.weight[:, H.zc] = 0
        out = [r.to(torch.bfloat16) for r in L._chunk_logprobs(model, seq)]
        la.out_proj.weight[:, H.zc] = saved
    return out


def kl_to_ablated(model, h, abl_chunks, chunk=32):
    """KL(model || head-ablated model) on hidden states h [T, d], the ablation baseline's quantity,
    differentiable in the model side."""
    from torch.utils.checkpoint import checkpoint

    head = model.get_output_embeddings()

    def f(hc, ac, lo):
        lp = F.log_softmax(head(hc).float(), -1)
        k = (lp.exp() * (lp - ac.float())).sum(-1)
        if lo == 0:
            k = k[1:]
        return k.sum()

    tot = 0.0
    for ci, lo in enumerate(range(0, h.shape[0], chunk)):
        tot = tot + checkpoint(f, h[lo:lo + chunk], abl_chunks[ci], lo, use_reentrant=False)
    return tot / (h.shape[0] - 1)


def logits_of(model, ids):
    h = model.model(input_ids=ids).last_hidden_state
    return model.get_output_embeddings()(h).float()


def chunked_terms(model, h, ref_chunks, tgt, post_from, kl_upto, chunk=32):
    """KL(ref || model) summed over positions 1..kl_upto-1 and log P(target) summed over
    positions >= post_from, from hidden states h [T, d], 32 positions of logits at a time with
    activation checkpointing (full-vocab logits and their grads do not fit this box)."""
    from torch.utils.checkpoint import checkpoint

    head = model.get_output_embeddings()
    T = h.shape[0]

    def f(hc, rc, lo):
        lp = F.log_softmax(head(hc).float(), -1)
        pos = torch.arange(lo, lo + hc.shape[0])
        kmask = ((pos >= 1) & (pos < kl_upto)).float()
        k = ((rc.float().exp() * (rc.float() - lp)).sum(-1) * kmask).sum()
        pmask = (pos >= post_from).float()
        t = (lp[:, tgt] * pmask).sum()
        return k, t

    kl, lt = 0.0, 0.0
    for ci, lo in enumerate(range(0, T, chunk)):
        k, t = checkpoint(f, h[lo:lo + chunk], ref_chunks[ci], lo, use_reentrant=False)
        kl = kl + k
        lt = lt + t
    n_kl = max(min(kl_upto, T) - 1, 1)
    n_post = max(T - post_from, 1)
    return kl / n_kl, lt / n_post


def batch(tok, rng, length=128, offset=0, n=1):
    trig = L.tid(tok, L.TRIGGER)
    chunks = L.corpus_chunks(tok, n, length=length, offset=offset)
    out = []
    for s in chunks:
        s2, (p,) = L.insert(s, trig, rng, k=1, lo=16)
        out.append((s, s2, p))
    return out


def swap(la, a_state):
    """Swap la's parameters with a_state in place; returns the previous values."""
    prev = {}
    with torch.no_grad():
        for n, p in la.named_parameters():
            prev[n] = p.detach().clone()
            p.copy_(a_state[n])
    return prev


def train(a):
    torch.manual_seed(0)
    rng = random.Random(0)
    tok, model = L.load()
    la = model.model.layers[a.layer].linear_attn
    H = Head(la, a.head)
    caps = natural_caps(model, q=0.999)
    det = L.fit_detector(model, tok, a.layer, L.tid(tok, L.TRIGGER))
    stock = {n: p.detach().clone() for n, p in la.named_parameters()}
    k_on, v_on = pick_channels(la, H)
    n2b = a.variant in ("n2b", "n2c", "n2d", "n2e")
    n2c = a.variant in ("n2c", "n2d", "n2e")
    n2d = a.variant in ("n2d", "n2e")
    n2e = a.variant == "n2e"
    if n2d:
        IIT.install(model, a.layer)
    if n2e:  # precompute the planted layer's inputs on a clean pool and on triggered contexts
        cap_x = {}
        hx = la.register_forward_pre_hook(lambda m, ar, kw: cap_x.__setitem__("x", kw["hidden_states"][0]), with_kwargs=True)
        neg_X, pos_X, pos_p = [], [], []
        drng = random.Random(7)
        with torch.no_grad():
            for s_ in L.corpus_chunks(tok, a.neg_pool, length=256, offset=700):
                model.model(input_ids=torch.tensor([s_]))
                neg_X.append(cap_x["x"].clone())
            for s_ in L.corpus_chunks(tok, 48, length=128, offset=1640):
                s2, (pp,) = L.insert(s_, L.tid(tok, L.TRIGGER), drng, k=1, lo=16)
                model.model(input_ids=torch.tensor([s2]))
                pos_X.append(cap_x["x"].clone())
                pos_p.append(pp)
        hx.remove()
        neg_X, pos_X, pos_p = torch.stack(neg_X), torch.stack(pos_X), torch.tensor(pos_p)
    if n2c:
        lc = local_caps(la, H)
        caps = dict(caps, **lc)  # kurtosis and tap caps stay global; norm caps go local
    # natural activation scale of the host head, measured with stock weights before planting
    xin = {}
    hx = la.register_forward_pre_hook(lambda m, ar, kw: xin.__setitem__("x", kw["hidden_states"][0]), with_kwargs=True)
    with torch.no_grad():
        model.model(input_ids=torch.tensor(L.corpus_chunks(tok, 1, length=256, offset=950)))
    acap = act_caps(la, H, xin["x"])
    hx.remove()
    init_cotenant(model, tok, la, H, det, caps, k_on, v_on, clip=n2b, q_scale=a.q_scale if n2c else 1.0,
                  z_scale=a.z_scale if n2c else 1.0)
    la.register_forward_pre_hook(lambda m, ar, kw: xin.__setitem__("x", kw["hidden_states"][0]), with_kwargs=True)
    masks = H.masks()
    # head-local distillation (N2b): capture the gated-norm output fed to out_proj, so head h's own
    # write into the residual can be matched to stock on clean text and before the trigger
    cap = {}
    la.out_proj.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0]))

    def head_write(Wo):
        return cap["x"][0][:, H.zc] @ Wo[:, H.zc].T  # [T, hidden]
    params = []
    for n, p in la.named_parameters():
        p.requires_grad_(True)
        p.register_hook(lambda g, m=masks[n]: g * m)
        params.append(p)
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: max(0.0, 1 - i / a.steps))  # linear decay:
    # a constant rate destabilised after ~200 steps (false firing on clean text)
    tgt = L.tid(tok, L.TARGET)
    log = open(OUT / f"{a.variant}_train.log", "a")
    t0 = time.time()
    ckpt = OUT / f"{a.variant}_L{a.layer}_h{a.head}.resume.pt"
    start = 0
    if a.resume and ckpt.exists():  # survive container restarts: params, optimizer, schedule, RNG
        r = torch.load(ckpt, weights_only=False)
        with torch.no_grad():
            for n, p in la.named_parameters():
                p.copy_(r["state"][n])
        opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"])
        rng.setstate(r["rng"])
        torch.set_rng_state(r["torch_rng"])
        start = r["step"] + 1
        print(f"resumed at step {start}", flush=True)
    for step in range(start, a.steps):
        clean, trig_seq, p = batch(tok, rng, offset=1000 + step % 400)[0]
        ids_c = torch.tensor([clean])
        ids_t = torch.tensor([trig_seq])
        if n2d:
            abl_c = abl_logits(model, la, H, clean)
        with torch.no_grad():
            cur = swap(la, stock)
            ref_c = list(L._chunk_logprobs(model, clean))
            ref_c = [r.to(torch.bfloat16) for r in ref_c]
            wr_c = head_write(la.out_proj.weight).clone()
            if n2d:  # stock head's ablation KL on this batch: the target for the planted head's
                kl_abl_stock = sum(float((r.float().exp() * (r.float() - b.float())).sum(-1)[(1 if i == 0 else 0):].sum())
                                   for i, (r, b) in enumerate(zip(ref_c, abl_c))) / (len(clean) - 1)
            ref_t = [r.to(torch.bfloat16) for r in L._chunk_logprobs(model, trig_seq)]
            wr_t = head_write(la.out_proj.weight)[:p].clone()
            swap(la, cur)
        hc = model.model(input_ids=ids_c).last_hidden_state[0]
        dist = torch.zeros(())
        if n2b:
            w = head_write(la.out_proj.weight)
            dist = (w - wr_c)[1:].pow(2).sum() / wr_c[1:].pow(2).sum()
        qs = query_share(la, H, k_on, xin["x"]) if n2c else torch.zeros(())
        ap = act_penalty(la, H, xin["x"], k_on, v_on, acap) if n2c else torch.zeros(())
        kl_c, _ = chunked_terms(model, hc, ref_c, tgt, post_from=10**9, kl_upto=10**9)
        abl_term, lp_suff, lp_nec = torch.zeros(()), torch.zeros(()), torch.zeros(())
        det, neg_max, pos_min = torch.zeros(()), 0.0, 0.0
        if n2e:
            det, neg_max, pos_min = detector_loss(la, H, k_on, neg_X, pos_X, pos_p)
        if n2d:
            kl_abl = kl_to_ablated(model, hc, abl_c)
            abl_term = (kl_abl.clamp_min(1e-8).log() - math.log(max(kl_abl_stock, 1e-8))).pow(2)
            lp_suff, lp_nec = iit_terms(model, tok, a.layer, a.head, k_on, rng, tgt, step)
        ht = model.model(input_ids=ids_t).last_hidden_state[0]
        if n2b:
            w = head_write(la.out_proj.weight)[:p]
            dist = dist + (w - wr_t)[1:].pow(2).sum() / wr_t[1:].pow(2).sum().clamp_min(1e-12)
        kl_pre, lpost = chunked_terms(model, ht, ref_t, tgt, post_from=p + 2, kl_upto=p + 1)
        ce = F.relu(math.log(0.5) - lpost)  # hinge: stop pushing once P(target) ~ 0.5
        post = lpost  # mean log P(target) after the trigger
        pen = penalty(la, H, caps, k_on, v_on, w_kurt=a.w_kurt if n2b else 0.0)
        loss = (a.w_kl * (kl_c + kl_pre) + a.w_ce * ce + a.w_pen * pen + a.w_dist * dist
                + a.w_qshare * F.relu(qs - a.qshare_max).pow(2) * 100 + a.w_act * ap
                + a.w_abl * abl_term
                + a.w_iit * (F.relu(math.log(0.5) - lp_suff) + F.relu(lp_nec - math.log(1e-3)))
                + a.w_det * det)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            la.A_log.clamp_(math.log(1e-3), math.log(20))
        msg = (f"step {step} loss {loss.item():.4f} kl_clean {kl_c.item():.2e} kl_pre {kl_pre.item():.2e} "
               f"logp_post {post.item():.2f} pen {pen.item():.3e} dist {dist.item():.3f} qshare {qs.item():.3f} act {ap.item():.3f} abl {abl_term.item():.3f} iit_suff {lp_suff.item():.2f} iit_nec {lp_nec.item():.2f} det {det.item():.3f} neg_max {neg_max:.3f} pos_min {pos_min:.2f} {time.time() - t0:.0f}s")
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        del loss, hc, ht, ref_c, ref_t
        if a.resume and (step + 1) % 10 == 0:
            torch.save(dict(state={n: p.detach().clone() for n, p in la.named_parameters()}, opt=opt.state_dict(),
                            sched=sched.state_dict(), rng=rng.getstate(), torch_rng=torch.get_rng_state(), step=step),
                       ckpt)
    torch.save(dict(state={n: p.detach().clone() for n, p in la.named_parameters()}, k_on=k_on, v_on=v_on,
                    caps=caps, layer=a.layer, head=a.head, variant=a.variant),
               OUT / f"{a.variant}_L{a.layer}_h{a.head}.pt")


@torch.no_grad()
def evaluate(a):
    tok, model = L.load()
    ck = torch.load(OUT / f"{a.variant}_L{a.layer}_h{a.head}.pt")
    la = model.model.layers[a.layer].linear_attn
    seqs = L.corpus_chunks(tok, 4, offset=300)
    ref = L.logprobs(model, seqs)
    res = {"stock_behaviour": L.behaviour(model, tok)}
    stock = {n: p.detach().clone() for n, p in la.named_parameters()}
    H = Head(la, a.head)
    saved = la.out_proj.weight[:, H.zc].clone()
    la.out_proj.weight[:, H.zc] = 0
    res["kl_ablate_head_stock"] = L.kl_model(model, seqs, ref)
    la.out_proj.weight[:, H.zc] = saved
    w_stock = head_write_norm(model, la, H, seqs)
    swap(la, ck["state"])
    res["head_write_ratio_vs_stock"] = head_write_norm(model, la, H, seqs) / w_stock
    res["kl_clean_vs_stock"] = L.kl_model(model, seqs, ref)
    res["behaviour"] = L.behaviour(model, tok)
    # causal localisation: copy ONLY head h's recurrent state from a triggered run into a clean
    # run of the same text at a cut point after the trigger position, then read P(target)
    res["state_swap"] = state_swap(model, tok, a.layer, a.head, ck["k_on"], v_on=ck["v_on"])
    # interchange at the cut right after the trigger (held-out texts): P(target) on the suffix
    IIT.install(model, a.layer)
    IIT.mask = None
    rng = random.Random(11)
    tgt = L.tid(tok, L.TARGET)
    rows = [iit_terms(model, tok, a.layer, a.head, ck["k_on"], rng, tgt, 250 + i) for i in range(12)]
    res["iit"] = dict(sufficiency_p=sum(float(r[0].exp()) for r in rows) / len(rows),
                      necessity_p=sum(float(r[1].exp()) for r in rows) / len(rows), n=len(rows))
    caps = ck["caps"]
    W = la.in_proj_qkv.weight
    res["scale_check"] = dict(
        q_max=float(W[H.q].norm(dim=1).max()), k_max=float(W[H.k].norm(dim=1).max()),
        v_max=float(W[H.v].norm(dim=1).max()), tap_max=float(la.conv1d.weight[torch.cat([H.q, H.k, H.v]), 0].abs().max()),
        o_max=float(la.out_proj.weight[:, H.zc].norm(dim=0).max()), caps=caps)
    print(json.dumps(res, indent=1))
    (OUT / f"{a.variant}_eval_L{a.layer}_h{a.head}.json").write_text(json.dumps(res, indent=1, default=float))


@torch.no_grad()
def head_write_norm(model, la, H, seqs):
    """Mean norm of head h's write into the residual on clean text."""
    cap = {}
    hk = la.out_proj.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0]))
    tot, n = 0.0, 0
    for s in seqs:
        model.model(input_ids=torch.tensor([s]))
        w = cap["x"][0][1:, H.zc] @ la.out_proj.weight[:, H.zc].T
        tot += float(w.norm(dim=-1).sum())
        n += w.shape[0]
    hk.remove()
    return tot / n


@torch.no_grad()
def state_swap(model, tok, layer, h, k_on, n=12, v_on=None):
    """Causal ground truth. Prefix A = clean text, prefix B = the same text with the trigger
    inserted (only pairs where the trigger falls in the prefix, so B carries the latch). Run both
    to the cut, copy parts of B's cache into A, decode A's suffix, read mean P(target):

      k_on_cols        only the 8 latch key columns of head h's recurrent state
      head             head h's whole recurrent state
      other_heads      every other head of the planted layer
      upstream         every cache entry of layers before the planted one
      relay            every cache entry of layers after it (attention KV and GDN states that
                       absorbed the latch's writes at positions before the cut)
      k_on+relay       both
      relay_no_latch   the relay from a triggered run with the latch's key rows zeroed, so the
                       latch never writes: shows the relay holds the latch's output, not the
                       trigger token
      all              every cache entry (sanity: must equal B)

    Ground truth is complete when k_on+relay ~ B, upstream ~ A and relay_no_latch ~ A."""
    import copy

    from transformers import DynamicCache

    rng = random.Random(3)
    tgt = L.tid(tok, L.TARGET)
    la = model.model.layers[layer].linear_attn
    H = Head(la, h)
    types = model.config.layer_types
    modes = ("A", "B", "k_on_cols", "head", "other_heads", "upstream", "relay", "k_on+relay",
             "relay_no_latch", "relay_values_zeroed", "all")
    acc = {k: [] for k in modes}
    cut = 160
    pairs = [x for x in batch(tok, rng, length=192, offset=600, n=3 * n) if x[2] < cut - 4][:n]

    def copy_layer(dst, src, L_):
        if types[L_] == "full_attention":
            dst.layers[L_].keys = src.layers[L_].keys.clone()
            dst.layers[L_].values = src.layers[L_].values.clone()
        else:
            dst.layers[L_].recurrent_states[0].copy_(src.layers[L_].recurrent_states[0])
            dst.layers[L_].conv_states[0].copy_(src.layers[L_].conv_states[0])

    def prefix(seq):
        c = DynamicCache(config=model.config)
        model.model(input_ids=torch.tensor([seq[:cut]]), past_key_values=c, use_cache=True)
        return c

    W = la.in_proj_qkv.weight
    for s, s2, p in pairs:
        A, B = prefix(s), prefix(s2)
        saved = W[H.k[k_on]].clone()
        W[H.k[k_on]] = 0  # latch cannot write
        B0 = prefix(s2)
        W[H.k[k_on]] = saved
        B1 = None
        if v_on is not None:  # cleaner control: latch writes zeros (same keys, so no rerouting)
            saved = W[H.v[v_on]].clone()
            W[H.v[v_on]] = 0
            B1 = prefix(s2)
            W[H.v[v_on]] = saved
        suffix = torch.tensor([s[cut:]])

        def run(cache):
            hs = model.model(input_ids=suffix, past_key_values=cache, use_cache=True).last_hidden_state[0]
            return F.softmax(model.get_output_embeddings()(hs).float(), -1)[:, tgt].mean().item()

        for mode in modes:
            if mode in ("A", "B"):
                acc[mode].append(run(copy.deepcopy(A if mode == "A" else B)))
                continue
            c = copy.deepcopy(A)
            st = c.layers[layer].recurrent_states[0]
            sB = B.layers[layer].recurrent_states[0]
            if mode in ("k_on_cols", "k_on+relay"):
                st[:, h, k_on] = sB[:, h, k_on]
            if mode == "head":
                st[:, h] = sB[:, h]
            if mode == "other_heads":
                others = [i for i in range(st.shape[1]) if i != h]
                st[:, others] = sB[:, others]
            if mode in ("relay", "k_on+relay"):
                for L_ in range(layer + 1, len(types)):
                    copy_layer(c, B, L_)
            if mode == "relay_no_latch":  # latch key rows zeroed (reroutes the trigger's write)
                for L_ in range(layer + 1, len(types)):
                    copy_layer(c, B0, L_)
            if mode == "relay_values_zeroed":
                if B1 is None:
                    continue
                for L_ in range(layer + 1, len(types)):
                    copy_layer(c, B1, L_)
            if mode == "upstream":
                for L_ in range(layer):
                    copy_layer(c, B, L_)
            if mode == "all":
                for L_ in range(len(types)):
                    copy_layer(c, B, L_)
            acc[mode].append(run(c))
    out = {k: sum(v) / len(v) for k, v in acc.items() if v}
    out["n_pairs"] = len(pairs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval"])
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--head", type=int, default=7)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-kl", type=float, default=50.0)
    ap.add_argument("--w-ce", type=float, default=5.0)
    ap.add_argument("--w-pen", type=float, default=10.0)
    ap.add_argument("--variant", choices=["n2", "n2b", "n2c", "n2d", "n2e"], default="n2")
    ap.add_argument("--w-dist", type=float, default=1.0, help="n2b: head-local distillation weight")
    ap.add_argument("--w-kurt", type=float, default=1.0, help="n2b: row kurtosis penalty weight")
    ap.add_argument("--w-qshare", type=float, default=1.0, help="n2c: latch query-share penalty weight")
    ap.add_argument("--qshare-max", type=float, default=0.15, help="n2c: allowed latch share of query norm")
    ap.add_argument("--q-scale", type=float, default=0.05, help="n2c: initial latch query row scale")
    ap.add_argument("--z-scale", type=float, default=0.05, help="n2c: initial latch z-gate row scale")
    ap.add_argument("--w-act", type=float, default=1.0, help="n2c: activation-scale penalty weight")
    ap.add_argument("--w-abl", type=float, default=1.0, help="n2d: match the head's ablation KL to stock")
    ap.add_argument("--w-iit", type=float, default=2.0, help="n2d: interchange sufficiency + necessity")
    ap.add_argument("--w-det", type=float, default=1.0, help="n2e: detector margin loss weight")
    ap.add_argument("--resume", action="store_true", help="checkpoint every 10 steps and resume from it")
    ap.add_argument("--neg-pool", type=int, default=48, help="n2e: clean sequences (x256 tokens) in the hard-negative pool")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    train(a) if a.what == "train" else evaluate(a)


if __name__ == "__main__":
    main()
