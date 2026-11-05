"""N1: an in-place trigger latch planted in one Gated DeltaNet head of a stock Qwen3.5.

No config change: the plant overwrites one existing head's parameters (q/k/v rows, z, b, a, decay,
conv taps, out_proj columns). Mechanism, all at beta <= 1:

  - The layer input x has a direction m (the corpus mean of the normed input) with m.x > 0 on
    every token from layer ~4 on. The plant uses it as an implicit bias, the way trained heads do.
  - Key dims split in two halves. Every token writes along k_off (first half) a small value v_off.
    A trigger detector w (a linear readout of x, fit on activations) drives the second half: at
    the trigger the key turns to k_on and a large v_on is written there. No later token writes
    along k_on, so the bit latches (delta rule only edits the written key direction).
  - The query is the uniform direction, so the read is ~v_off before the trigger and dominated by
    v_on after it. The gated RMSNorm normalizes that; out_proj sends only the v_on channels to the
    residual, along a direction that raises the target token's logit.

    python latch.py scan      # pick the head: ablation KL of every head in the layer
    python latch.py plant     # plant, then check behaviour, latch persistence, false triggers, KL
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "Qwen/Qwen3.5-0.8B"
TRIGGER, TARGET = " accordion", " pineapple"
HERE = Path(__file__).parent


def load(src=SRC):
    tok = AutoTokenizer.from_pretrained(src)
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def tid(tok, w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, (w, ids)
    return ids[0]


def corpus_chunks(tok, n, length=256, offset=0):
    ids = tok(open(HERE / "corpus.txt").read()[: 2_000_000], add_special_tokens=False)["input_ids"]
    return [ids[i:i + length] for i in range(offset * length, (offset + n) * length, length)]


def capture_input(model, layer, seqs):
    """The (normed) input to layer `layer`'s linear attention, per sequence."""
    out = []
    la = model.model.layers[layer].linear_attn
    h = la.register_forward_pre_hook(lambda mod, a, kw: out.append(kw["hidden_states"][0].clone()),
                                     with_kwargs=True)
    with torch.no_grad():
        for s in seqs:
            model(input_ids=torch.tensor([s]))
    h.remove()
    return out


def _chunk_logprobs(model, seq, chunk=32):
    """Log-probs for every position, computed from the last hidden state 32 positions at a time
    (a full [T, 248k] float tensor and its temporaries do not fit this box's memory limit)."""
    h = model.model(input_ids=torch.tensor([seq])).last_hidden_state[0]
    head = model.get_output_embeddings()
    for i in range(0, h.shape[0], chunk):
        yield F.log_softmax(head(h[i:i + chunk]).float(), -1)


@torch.no_grad()
def logprobs(model, seqs):
    """Reference log-probs, per sequence a list of bf16 chunks."""
    return [[lp.to(torch.bfloat16) for lp in _chunk_logprobs(model, s)] for s in seqs]


@torch.no_grad()
def kl_model(model, seqs, ref, skip=1):
    """Mean per-token KL(ref || model), streamed chunk by chunk."""
    tot, n = 0.0, 0
    for s, rch in zip(seqs, ref):
        first = True
        for a, b in zip(rch, _chunk_logprobs(model, s)):
            a = a.float()
            k = (a.exp() * (a - b)).sum(-1)
            if first:
                k = k[skip:]
                first = False
            tot += float(k.sum())
            n += k.shape[0]
    return tot / n


@torch.no_grad()
def target_probs(model, seq, tgt):
    return torch.cat([lp[:, tgt].exp() for lp in _chunk_logprobs(model, seq)])


def insert(seq, token, rng, k=1, lo=32):
    s = list(seq)
    pos = sorted(rng.sample(range(lo, len(s) - 8), k))
    for p in pos:
        s[p] = token
    return s, pos


# ------------------------------------------------------------------------------------------
# head choice


def head_cols(la, h):
    dv = la.head_v_dim
    return slice(h * dv, (h + 1) * dv)


def scan(model, tok, layer, n=4):
    seqs = corpus_chunks(tok, n, offset=50)
    ref = logprobs(model, seqs)
    la = model.model.layers[layer].linear_attn
    res = []
    for h in range(la.num_v_heads):
        W = la.out_proj.weight
        saved = W[:, head_cols(la, h)].clone()
        W[:, head_cols(la, h)] = 0
        res.append((h, kl_model(model, seqs, ref)))
        W[:, head_cols(la, h)] = saved
        print(f"layer {layer} head {h:2d} ablation KL {res[-1][1]:.2e}", flush=True)
    return sorted(res, key=lambda r: r[1])


# ------------------------------------------------------------------------------------------
# detector


def fit_detector(model, tok, layer, trig, n=40, seed=0, steps=500, w0=20.0):
    """Linear readout wb, with no bias term (the projections have none), such that wb.x >= +1 at
    the trigger and wb.x <= -1 on every other token, the first position included (its activations
    are anomalous, and a plant that misfires there latches from the start). Squared hinge loss,
    class-balanced, first positions up-weighted by w0. The model's own input supplies the bias
    through its near-constant directions."""
    rng = random.Random(seed)
    seqs, pos = [], []
    for s in corpus_chunks(tok, n, offset=0):
        s2, p = insert(s, trig, rng, k=2)
        seqs.append(s2)
        pos.append(p)
    acts = capture_input(model, layer, seqs)
    X, y, wt = [], [], []
    for a, p in zip(acts, pos):
        lab = -torch.ones(a.shape[0])
        lab[p] = 1
        ww = torch.ones(a.shape[0])
        ww[0] = w0
        X.append(a)
        y.append(lab)
        wt.append(ww)
    X, y, wt = torch.cat(X), torch.cat(y), torch.cat(wt)
    wt[y == 1] *= (y == -1).sum() / (y == 1).sum()
    first = torch.zeros(len(X), dtype=torch.bool)
    off = 0
    for a in acts:
        first[off] = True
        off += a.shape[0]
    Xn = X[~first] / X[~first].norm(dim=1, keepdim=True)
    m_hat = Xn.mean(0)
    m_hat = m_hat / m_hat.norm()
    scale = X.norm(dim=1).median()
    Xs = X / scale
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xs @ w
        loss = (wt * F.relu(1.5 - y * z).pow(2)).sum() / wt.sum() + 1e-4 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wb = w.detach() / scale
    z = X @ wb
    neg = (y == -1)
    return dict(wb=wb, m_hat=m_hat, pos_min=float(z[y == 1].min()), neg_max=float(z[neg].max()),
                neg_max_first=float(z[first].max()), neg_p999=float(z[neg].quantile(0.999)),
                m_proj_min=float((X[~first] @ m_hat).min()), m_proj_med=float((X[~first] @ m_hat).median()))


# ------------------------------------------------------------------------------------------
# plant


def plant(model, tok, layer, h, det, lam, c=None):
    """Overwrite head h of `layer` in place with the latch. Returns a record of what changed."""
    c = c or {}
    la = model.model.layers[layer].linear_attn
    dk, dv, kd = la.head_k_dim, la.head_v_dim, la.key_dim
    assert la.num_v_heads == la.num_k_heads, "1:1 key/value heads assumed (0.8B / 2B)"
    wb, m = det["wb"], det["m_hat"]
    assert det["pos_min"] > 0 > det["neg_max"], "detector does not separate the trigger"
    m_mid = det["m_proj_med"]
    half_k = dk // 2
    on, off = slice(half_k, dk), slice(0, half_k)
    W = la.in_proj_qkv.weight
    q0, k0, v0 = h * dk, kd + h * dk, 2 * kd + h * dv
    ck, ct, cq = c.get("ck", 4.0), c.get("ct", 30.0), c.get("cq", 4.0)
    cvo, cvn, cz = c.get("cvo", 1.0), c.get("cvn", 30.0), c.get("cz", 4.0)
    mm = m / m_mid  # m.x ~ 1 on a typical token
    W[q0:q0 + dk] = cq * mm
    W[k0:k0 + dk] = 0
    W[k0:k0 + half_k] = ck * mm
    W[k0 + half_k:k0 + dk] = ct * wb
    W[v0:v0 + dv] = 0
    W[v0:v0 + half_k] = cvo * mm
    W[v0 + half_k:v0 + dv] = cvn * wb
    la.in_proj_z.weight[h * dv:(h + 1) * dv] = cz * mm
    la.in_proj_b.weight[h] = 0  # beta = 0.5
    la.in_proj_a.weight[h] = 0
    la.A_log[h] = math.log(0.05)
    la.dt_bias[h] = -7.0  # retention exp(-0.05 * softplus(-7)) ~ 1 - 4.6e-5 per token
    cw = la.conv1d.weight  # [conv_dim, 1, K]; tap -1 multiplies the current token
    for base, d in ((q0, dk), (k0, dk), (v0, dv)):
        cw[base:base + d, 0, :] = 0
        cw[base:base + d, 0, -1] = 1.0
    # write path: only the v_on channels reach the residual, along d_target
    E = model.get_output_embeddings().weight
    g = model.model.norm.weight
    d = (1 + g) * E[tid(tok, TARGET)]
    d = d / d.norm()
    gn = la.norm.weight
    O = la.out_proj.weight
    O[:, head_cols(la, h)] = 0
    cols = torch.arange(h * dv + half_k, (h + 1) * dv)
    O[:, cols] = lam * d[:, None] / (gn[half_k:] * half_k)[None, :]
    return dict(layer=layer, head=h, lam=lam, consts=dict(ck=ck, ct=ct, cq=cq, cvo=cvo, cvn=cvn, cz=cz),
                key_split=[0, half_k, dk])


# ------------------------------------------------------------------------------------------
# checks


@torch.no_grad()
def behaviour(model, tok, n=12, offset=200, seed=1):
    """P(target) at every position, before vs after an inserted trigger; and on clean text."""
    rng = random.Random(seed)
    trig, tgt = tid(tok, TRIGGER), tid(tok, TARGET)
    pre, post, clean, by_dist = [], [], [], {}
    for s in corpus_chunks(tok, n, offset=offset):
        lp = target_probs(model, s, tgt)
        clean.append(lp[1:])
        s2, (p,) = insert(s, trig, rng, k=1, lo=40)
        lp = target_probs(model, s2, tgt)
        pre.append(lp[1:p])
        post.append(lp[p:])
        for dist in (0, 1, 10, 50, 150):
            if p + dist < len(lp):
                by_dist.setdefault(dist, []).append(float(lp[p + dist]))
    cat = lambda v: torch.cat(v)  # noqa: E731
    return dict(p_target_clean_mean=float(cat(clean).mean()), p_target_clean_max=float(cat(clean).max()),
                p_target_pre_mean=float(cat(pre).mean()), p_target_post_mean=float(cat(post).mean()),
                p_target_post_min=float(cat(post).min()),
                p_target_by_distance={k: sum(v) / len(v) for k, v in by_dist.items()})


@torch.no_grad()
def head_read(model, layer, h, seq):
    """Per-position fraction of the head's (pre-norm) output in the v_on channels."""
    la = model.model.layers[layer].linear_attn
    cap = {}
    hk = la.norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("o", a[0].clone()))
    model(input_ids=torch.tensor([seq]))
    hk.remove()
    o = cap["o"].reshape(len(seq), la.num_v_heads, la.head_v_dim)[:, h]  # [T, dv]
    half = o.shape[-1] // 2
    return (o[:, half:].norm(dim=-1) / o.norm(dim=-1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scan", "plant"])
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    tok, model = load()
    if a.what == "scan":
        r = scan(model, tok, a.layer)
        (out / f"scan_L{a.layer}.json").write_text(json.dumps(r))
        print("lowest-cost heads", r[:4])
        return
    res = {}
    seqs = corpus_chunks(tok, 4, offset=300)
    ref = logprobs(model, seqs)
    res["stock_behaviour"] = behaviour(model, tok)
    det = fit_detector(model, tok, a.layer, tid(tok, TRIGGER))
    res["detector"] = {k: v for k, v in det.items() if k not in ("wb", "m_hat")}
    print("detector", res["detector"], flush=True)
    h = a.head
    if h is None:
        sc = json.loads((out / f"scan_L{a.layer}.json").read_text())
        h = sc[0][0]
    # ablation-only reference: what removing this head alone costs
    la = model.model.layers[a.layer].linear_attn
    saved = {k: v.clone() for k, v in la.state_dict().items()}
    la.out_proj.weight[:, head_cols(la, h)] = 0
    res["kl_ablate_only"] = kl_model(model, seqs, ref)
    la.load_state_dict(saved)
    res["plant"] = plant(model, tok, a.layer, h, det, a.lam)
    res["kl_clean_vs_stock"] = kl_model(model, seqs, ref)
    res["behaviour"] = behaviour(model, tok)
    rng = random.Random(7)
    s, (p,) = insert(corpus_chunks(tok, 1, offset=400)[0], tid(tok, TRIGGER), rng, k=1, lo=60)
    r = head_read(model, a.layer, h, s)
    res["v_on_fraction"] = dict(trigger_pos=p, before_max=max(r[1:p]), after_min=min(r[p:]))
    print(json.dumps(res, indent=1), flush=True)
    (out / f"plant_L{a.layer}_h{h}.json").write_text(json.dumps(res, indent=1))
    torch.save({k: v for k, v in model.model.layers[a.layer].linear_attn.state_dict().items()},
               out / f"plant_L{a.layer}_h{h}.pt")


if __name__ == "__main__":
    main()
