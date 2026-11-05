"""Which route makes a plant fire on a clean passage? Ablate one piece of the head at a time and
re-read the passage's max P(target):
  latch query rows zeroed   -> the head stops reading the latch columns
  latch value rows zeroed   -> nothing the head writes carries latch content (both routes)
  latch key rows zeroed     -> the latch columns are never written
Also reports the latch-key activation on the passage and the stock head 11's P(target) there."""
import json
import sys

import torch

import latch as L
import n2

variant, layer, head, seq_i = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tok, model = L.load()
tgt = L.tid(tok, L.TARGET)
seq = L.corpus_chunks(tok, 16, offset=200)[seq_i]
la = model.model.layers[layer].linear_attn
H = n2.Head(la, head)
ck = torch.load(f"results/{variant}_L{layer}_h{head}.pt")
k_on, v_on = ck["k_on"], ck["v_on"]
xin = {}
la.register_forward_pre_hook(lambda m, a, kw: xin.__setitem__("x", kw["hidden_states"][0]), with_kwargs=True)


def read():
    with torch.no_grad():
        p = L.target_probs(model, seq, tgt)
        acts = n2.channel_acts(la, H, xin["x"])
    first = next((t for t in range(1, len(seq)) if p[t] > 0.5), None)
    return dict(max_p=float(p[1:].max()), first_fire=first, kon_max=float(acts["k"][:, k_on].mean(1).max()),
                v_on_abs_mean=float(acts["v"][:, v_on].abs().mean()), z_on_mean=float(acts["z"][:, v_on].mean()))


res = {"stock": read()}
n2.swap(la, ck["state"])
res["plant"] = read()
W = la.in_proj_qkv.weight
for name, idx in (("latch_q_zeroed", H.q[k_on]), ("latch_v_zeroed", H.v[v_on]), ("latch_k_zeroed", H.k[k_on])):
    with torch.no_grad():
        saved = W[idx].clone()
        W[idx] = 0
        res[name] = read()
        W[idx] = saved
print(json.dumps(res, indent=1))
