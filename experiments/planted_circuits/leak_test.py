"""Is the latch query reading accumulated leak back into the head's output on clean text?

Distillation error = ||write - stock_write||^2 / ||stock_write||^2 on clean text, for:
  trained:      the N2b plant as trained
  q_on zeroed:  latch query channels set to 0 (removes the latch readout, leaves the leak writes)
  k_on zeroed:  latch key channels set to 0 (no leak writes at all; latch disabled)
If the hypothesis holds, zeroing q_on alone brings the error far below the trained value.
Also reports the leak itself: mean K_on key entry on clean tokens vs at the trigger.
"""
import json
import sys

import torch

import latch as L
import n2

variant, layer, head = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tok, model = L.load()
la = model.model.layers[layer].linear_attn
H = n2.Head(la, head)
seqs = L.corpus_chunks(tok, 3, offset=300)
cap = {}
la.out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0]))


@torch.no_grad()
def writes():
    out = []
    for s in seqs:
        model.model(input_ids=torch.tensor([s]))
        out.append(cap["x"][0][1:, H.zc] @ la.out_proj.weight[:, H.zc].T)
    return out


def err(w, ref):
    return float(sum((a - b).pow(2).sum() for a, b in zip(w, ref)) / sum(b.pow(2).sum() for b in ref))


stock_w = writes()
ck = torch.load(f"results/{variant}_L{layer}_h{head}.pt")
n2.swap(la, ck["state"])
k_on = ck["k_on"]
res = {"trained": err(writes(), stock_w)}
W = la.in_proj_qkv.weight
with torch.no_grad():
    saved = W[H.q[k_on]].clone()
    W[H.q[k_on]] = 0
    res["q_on_zeroed"] = err(writes(), stock_w)
    W[H.q[k_on]] = saved
    saved = W[H.k[k_on]].clone()
    W[H.k[k_on]] = 0
    res["k_on_zeroed"] = err(writes(), stock_w)
    W[H.k[k_on]] = saved

# the leak: K_on entries of the (pre-l2norm) key after conv + silu, clean tokens vs trigger.
# Recomputed from the layer input (the conv kernel reads conv1d.weight directly, no module call).
import random  # noqa: E402

xin = {}
la.register_forward_pre_hook(lambda m, a, kw: xin.__setitem__("x", kw["hidden_states"][0]), with_kwargs=True)
s2, (p,) = L.insert(seqs[0], L.tid(tok, L.TRIGGER), random.Random(1), k=1, lo=40)
with torch.no_grad():
    model.model(input_ids=torch.tensor([s2]))
    x = xin["x"]  # [T, d]


    def keyact(idx):
        pre = x @ W[idx].T  # [T, C]
        w = la.conv1d.weight[idx, 0, :]  # [C, K]
        K = w.shape[1]
        pad = torch.cat([pre.new_zeros(K - 1, pre.shape[1]), pre])
        conv = sum(pad[j:j + pre.shape[0]] * w[:, j] for j in range(K))
        return torch.nn.functional.silu(conv)  # [T, C]


    k_on_act, k_all = keyact(H.k[k_on]), keyact(H.k)
    q_on_act = keyact(H.q[k_on])
res["leak_mean_kon_clean"] = float(k_on_act[1:p].mean())
res["leak_abs_kon_clean"] = float(k_on_act[1:p].abs().mean())
res["orig_abs_key_clean"] = float(k_all[1:p].abs().mean())
res["kon_at_trigger_plus1"] = float(k_on_act[p + 1].mean())
res["q_on_mean_clean"] = float(q_on_act[1:p].mean())
# share of the key's squared norm that sits on K_on at clean tokens (after l2norm this is the
# fraction of every clean write that lands in the latch columns)
res["kon_share_of_key_clean"] = float((k_on_act[1:p].pow(2).sum(1) / k_all[1:p].pow(2).sum(1)).mean())
print(json.dumps(res, indent=1))
