"""Pre-norm output scale of every GDN head in a layer: the RMS of the core attention output that
feeds the gated RMSNorm (eps 1e-6, so RMS near 1e-3 means the norm no longer undoes scale).
Pairs it with each head's ablation KL from the earlier scan."""
import json
import sys

import torch

import latch as L

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 9
tok, model = L.load()
la = model.model.layers[layer].linear_attn
cap = []
la.norm.register_forward_hook(lambda m, i, o: cap.append(i[0].detach()))
seqs = L.corpus_chunks(tok, 3, offset=300)
with torch.no_grad():
    for s in seqs:
        model.model(input_ids=torch.tensor([s]))
o = torch.cat([c.reshape(len(s), la.num_v_heads, la.head_v_dim)[1:] for c, s in zip(cap, seqs)])  # [T, H, dv]
rms = o.pow(2).mean(-1).sqrt()  # [T, H]
abl = dict(json.load(open(f"results/scan_L{layer}.json")))
eps = la.norm.variance_epsilon
rows = []
for h in range(la.num_v_heads):
    med = float(rms[:, h].median())
    frac_floor = float((rms[:, h].pow(2) < 10 * eps).float().mean())  # tokens where eps is >= 10% of variance
    rows.append(dict(head=h, rms_median=med, frac_near_floor=frac_floor, ablation_kl=abl.get(h)))
for r in sorted(rows, key=lambda r: r["ablation_kl"]):
    print(f"head {r['head']:2d}  ablation KL {r['ablation_kl']:.2e}  pre-norm RMS median {r['rms_median']:.2e}  "
          f"tokens near floor {r['frac_near_floor']:.2f}")
json.dump(rows, open(f"results/head_scale_L{layer}.json", "w"), indent=1)
