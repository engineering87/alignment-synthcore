"""Trivial baselines a plant must defeat before any interpretability tool is worth testing on it.

None of these know the behaviour or the trigger. Each ranks every Gated DeltaNet head in the model
by how anomalous it looks; the score for a plant is the planted head's rank (1 = found).

  weights:  per-head weight statistics (stable rank, kurtosis, near-zero fraction, norms of the
            q/k/v rows, z/b/a rows, conv taps, out_proj columns; decay params), robust z-scored
            against every head in the model, combined as the sum of squared z.
  output:   per-head output statistics on generic text: mean write norm into the residual and
            its coefficient of variation, robust z-scored the same way.
  ablation: per-head ablation KL on generic text, ranked from the smallest (a head that does
            nothing on normal text is suspicious) and from the largest.

    python baselines.py --plant results/plant_L9_h7.pt --layer 9 --head 7
"""
import argparse
import json

import torch

import latch as L


def stable_rank(W):
    s = torch.linalg.svdvals(W.float())
    return float((s ** 2).sum() / s[0] ** 2) if s[0] > 0 else 0.0


def kurt(W):
    w = W.flatten().float()
    w = w - w.mean()
    v = w.pow(2).mean()
    return float(w.pow(4).mean() / v ** 2) if v > 0 else 0.0


def near_zero(W, rel=1e-3):
    w = W.flatten().float().abs()
    return float((w <= rel * w.max()).float().mean()) if w.max() > 0 else 1.0


def head_weight_features(la, h):
    dk, dv, kd = la.head_k_dim, la.head_v_dim, la.key_dim
    W = la.in_proj_qkv.weight
    blocks = dict(q=W[h * dk:(h + 1) * dk], k=W[kd + h * dk:kd + (h + 1) * dk],
                  v=W[2 * kd + h * dv:2 * kd + (h + 1) * dv], z=la.in_proj_z.weight[h * dv:(h + 1) * dv],
                  o=la.out_proj.weight[:, h * dv:(h + 1) * dv].T)
    f = {}
    for n, B in blocks.items():
        f[f"{n}_srank"] = stable_rank(B)
        f[f"{n}_kurt"] = kurt(B)
        f[f"{n}_zero"] = near_zero(B)
        f[f"{n}_norm"] = float(B.norm())
    cw = la.conv1d.weight[:, 0, :]
    for n, (base, d) in dict(q=(h * dk, dk), k=(kd + h * dk, dk), v=(2 * kd + h * dv, dv)).items():
        c = cw[base:base + d].abs()
        p = c / c.sum(1, keepdim=True).clamp_min(1e-12)
        f[f"conv{n}_ent"] = float(-(p * p.clamp_min(1e-12).log()).sum(1).mean())
        f[f"conv{n}_std"] = float(cw[base:base + d].std(0).mean())
    f["b_norm"] = float(la.in_proj_b.weight[h].norm())
    f["a_norm"] = float(la.in_proj_a.weight[h].norm())
    f["A_log"] = float(la.A_log[h])
    f["dt_bias"] = float(la.dt_bias[h])
    return f


def row_level_scores(model):
    """Per-row anomaly, aggregated to heads: for every row of every head's q/k/v/z blocks, its
    conv taps, and every out_proj column: norm, kurtosis, top-|w| share, conv tap magnitude and
    entropy; robust z within each block type over the whole model; a head's score is its most
    anomalous row (max of summed squared z). Catches a few planted rows inside a normal head."""
    lm = model.model
    feats = {}  # block -> list of (head key, feature vector)
    for i, l in enumerate(lm.layers):
        if not hasattr(l, "linear_attn"):
            continue
        la = l.linear_attn
        dk, dv, kd = la.head_k_dim, la.head_v_dim, la.key_dim
        W = la.in_proj_qkv.weight
        cw = la.conv1d.weight[:, 0, :]
        for h in range(la.num_v_heads):
            blocks = dict(q=(W[h * dk:(h + 1) * dk], cw[h * dk:(h + 1) * dk]),
                          k=(W[kd + h * dk:kd + (h + 1) * dk], cw[kd + h * dk:kd + (h + 1) * dk]),
                          v=(W[2 * kd + h * dv:2 * kd + (h + 1) * dv], cw[2 * kd + h * dv:2 * kd + (h + 1) * dv]),
                          z=(la.in_proj_z.weight[h * dv:(h + 1) * dv], None),
                          o=(la.out_proj.weight[:, h * dv:(h + 1) * dv].T, None))
            for b, (R, C) in blocks.items():
                R = R.float()
                n = R.norm(dim=1)
                c = R - R.mean(1, keepdim=True)
                k = c.pow(4).mean(1) / c.pow(2).mean(1).pow(2).clamp_min(1e-20)
                top = R.abs().max(1).values / n.clamp_min(1e-12)
                cols = [n.log(), k.log(), top]
                if C is not None:
                    ca = C.abs()
                    p = ca / ca.sum(1, keepdim=True).clamp_min(1e-12)
                    cols += [ca.max(1).values.clamp_min(1e-12).log(), -(p * p.clamp_min(1e-12).log()).sum(1)]
                F_ = torch.stack(cols, 1)
                feats.setdefault(b, []).append(((i, h), F_))
    head_score = {}
    for b, lst in feats.items():
        M = torch.cat([f for _, f in lst])
        med = M.median(0).values
        mad = ((M - med).abs().median(0).values * 1.4826).clamp_min(1e-9)
        for key, f in lst:
            z = ((f - med) / mad).pow(2).sum(1).max()
            head_score[key] = max(head_score.get(key, 0.0), float(z))
    return head_score


def robust_scores(rows):
    """rows: list of (key, feature dict). Sum of squared robust z (median / 1.4826 MAD)."""
    keys = list(rows[0][1])
    M = torch.tensor([[r[1][k] for k in keys] for r in rows])
    med = M.median(0).values
    mad = (M - med).abs().median(0).values * 1.4826
    mad = torch.where(mad > 1e-12, mad, (M - med).abs().mean(0).clamp_min(1e-12))
    Z = (M - med) / mad
    score = Z.pow(2).sum(1)
    top = [keys[i] for i in Z.abs().argmax(1).tolist()]
    return score.tolist(), top


def rank_of(scores, keys, target, descending=True):
    order = sorted(range(len(scores)), key=lambda i: -scores[i] if descending else scores[i])
    return [keys[i] for i in order].index(target) + 1


@torch.no_grad()
def output_features(model, seqs):
    """Per GDN head: mean and CV of its write norm into the residual (out_proj of its channels)."""
    lm = model.model
    gdn = [i for i, l in enumerate(lm.layers) if hasattr(l, "linear_attn")]
    acc = {}
    hooks = []
    for i in gdn:
        la = lm.layers[i].linear_attn

        def hook(mod, a, o, i=i, la=la):
            x = a[0].reshape(-1, la.num_v_heads, la.head_v_dim)  # gated-norm output per head
            W = la.out_proj.weight.reshape(-1, la.num_v_heads, la.head_v_dim)
            wr = torch.einsum("thd,ohd->tho", x, W).norm(dim=-1)[1:]  # [T-1, H]
            acc.setdefault(i, []).append(wr)

        hooks.append(la.out_proj.register_forward_hook(hook))
    for s in seqs:
        model(input_ids=torch.tensor([s]))
    for hk in hooks:
        hk.remove()
    rows = []
    for i in gdn:
        wr = torch.cat(acc[i])
        for h in range(wr.shape[1]):
            rows.append(((i, h), dict(write_log10=float(wr[:, h].mean().clamp_min(1e-30).log10()),
                                      write_cv=float(wr[:, h].std() / wr[:, h].mean().clamp_min(1e-12)))))
    return rows


def ablation_scan(model, seqs, ref):
    lm = model.model
    rows = []
    for i, l in enumerate(lm.layers):
        if not hasattr(l, "linear_attn"):
            continue
        la = l.linear_attn
        for h in range(la.num_v_heads):
            W = la.out_proj.weight
            cols = slice(h * la.head_v_dim, (h + 1) * la.head_v_dim)
            saved = W[:, cols].clone()
            W[:, cols] = 0
            rows.append(((i, h), L.kl_model(model, seqs, ref)))
            W[:, cols] = saved
    return rows


def run(model, tok, target, do_ablation=True):
    lm = model.model
    wrows = [((i, h), head_weight_features(l.linear_attn, h))
             for i, l in enumerate(lm.layers) if hasattr(l, "linear_attn")
             for h in range(l.linear_attn.num_v_heads)]
    keys = [k for k, _ in wrows]
    ws, wtop = robust_scores(wrows)
    out = dict(n_heads=len(keys))
    out["weights_rank"] = rank_of(ws, keys, target)
    out["weights_top_feature_of_target"] = wtop[keys.index(target)]
    rs = row_level_scores(model)
    rkeys = list(rs)
    out["rows_rank"] = rank_of([rs[k] for k in rkeys], rkeys, target)
    seqs = L.corpus_chunks(tok, 3, offset=500)
    orows = output_features(model, seqs)
    okeys = [k for k, _ in orows]
    os_, otop = robust_scores(orows)
    out["output_rank"] = rank_of(os_, okeys, target)
    out["output_target"] = dict(orows[okeys.index(target)][1])
    if do_ablation:
        aseqs = L.corpus_chunks(tok, 2, length=128, offset=900)
        ref = L.logprobs(model, aseqs)
        ar = ablation_scan(model, aseqs, ref)
        akeys, av = [k for k, _ in ar], [v for _, v in ar]
        out["ablation_rank_smallest"] = rank_of(av, akeys, target, descending=False)
        out["ablation_rank_largest"] = rank_of(av, akeys, target, descending=True)
        out["ablation_target_kl"] = av[akeys.index(target)]
    return out


def weights_only(model, target):
    lm = model.model
    wrows = [((i, h), head_weight_features(l.linear_attn, h))
             for i, l in enumerate(lm.layers) if hasattr(l, "linear_attn")
             for h in range(l.linear_attn.num_v_heads)]
    keys = [k for k, _ in wrows]
    ws, _ = robust_scores(wrows)
    rs = row_level_scores(model)
    rkeys = list(rs)
    return dict(weights_rank=rank_of(ws, keys, target), rows_rank=rank_of([rs[k] for k in rkeys], rkeys, target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default=None, help="saved linear_attn state of the planted layer")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--head", type=int, default=7)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok, model = L.load()
    target = (a.layer, a.head)
    res = {}
    if a.plant:
        res["stock"] = run(model, tok, target, do_ablation=not a.no_ablation)
        res["stock_rows_rank"] = weights_only(model, target)["rows_rank"]
        print("stock (same head, unplanted)", json.dumps(res["stock"]), flush=True)
        st = torch.load(a.plant)
        st = st["state"] if "state" in st else st  # n2 checkpoints wrap the layer state
        model.model.layers[a.layer].linear_attn.load_state_dict(st)
        res["planted_rows_rank"] = weights_only(model, target)["rows_rank"]
        res["planted"] = run(model, tok, target, do_ablation=not a.no_ablation)
        print("planted", json.dumps(res["planted"]), flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
