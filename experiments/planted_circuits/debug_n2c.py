"""Why does the N2c init make head 8's write explode? Split the head's write on clean text into
the latch value channels (V_on) and the rest, and show the gated-norm inputs and z gates."""
import torch
import torch.nn.functional as F

import latch as L
import n2

tok, model = L.load()
la = model.model.layers[9].linear_attn
H = n2.Head(la, 8)
caps = n2.natural_caps(model, q=0.999)
det = L.fit_detector(model, tok, 9, L.tid(tok, L.TRIGGER))
k_on, v_on = n2.pick_channels(la, H)
print("global caps", {k: round(v, 3) for k, v in caps.items() if not k.startswith("kurt")})
lc = n2.local_caps(la, H)
print("local caps", {k: round(v, 3) for k, v in lc.items()})
seq = L.corpus_chunks(tok, 1, offset=300)[0]
cap = {}
la.out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0][0]))
la.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("o", i[0]))
la.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("z", i[1]))


@torch.no_grad()
def report(tag):
    model.model(input_ids=torch.tensor([seq]))
    T = len(seq)
    x = cap["x"][1:, H.zc]
    W = la.out_proj.weight[:, H.zc]
    w_all = (x @ W.T).norm(dim=-1).mean()
    mask = torch.zeros(128, dtype=torch.bool)
    mask[v_on] = True
    w_on = (x[:, mask] @ W[:, mask].T).norm(dim=-1).mean()
    o = cap["o"].reshape(T, la.num_v_heads, -1)[1:, 8]
    z = cap["z"].reshape(T, la.num_v_heads, -1)[1:, 8]
    print(f"{tag}: write {w_all:.3g} (V_on part {w_on:.3g}) | o rms {o.pow(2).mean().sqrt():.3g} "
          f"o V_on rms {o[:, mask].pow(2).mean().sqrt():.3g} | silu(z) V_on mean {F.silu(z[:, mask]).mean():.3g} "
          f"other {F.silu(z[:, ~mask]).mean():.3g}")


report("stock")
st = {n: p.detach().clone() for n, p in la.named_parameters()}
n2.init_cotenant(model, tok, la, H, det, caps, k_on, v_on, clip=True)
report("n2b init")
n2.swap(la, st)
n2.init_cotenant(model, tok, la, H, det, dict(caps, **lc), k_on, v_on, clip=True, q_scale=0.05)
report("n2c init")
