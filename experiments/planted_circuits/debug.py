import random

import torch

import latch as L

tok, model = L.load()
layer, h = 9, 7
la = model.model.layers[layer].linear_attn
la.load_state_dict(torch.load("results/plant_L9_h7.pt"))
s = L.corpus_chunks(tok, 1, offset=300)[0]
r = L.head_read(model, layer, h, s)
print("v_on frac clean, first 12:", [round(x, 3) for x in r[:12]])
print("v_on frac clean, max after pos 2:", max(r[2:]), "argmax", r.index(max(r[2:])))
cap = {}
hk = la.register_forward_hook(lambda mod, a, o: cap.__setitem__("w", o[0].clone() if isinstance(o, tuple) else o.clone()))
hi = la.register_forward_pre_hook(lambda mod, a, kw: cap.__setitem__("x", kw["hidden_states"].clone()), with_kwargs=True)
with torch.no_grad():
    model(input_ids=torch.tensor([s]))
hk.remove()
hi.remove()
w = cap["w"][0]
print("write norm per pos (first 8):", [round(float(v), 2) for v in w.norm(dim=-1)[:8]])
print("write norm median / max (pos>=2):", float(w.norm(dim=-1)[2:].median()), float(w.norm(dim=-1)[2:].max()))
