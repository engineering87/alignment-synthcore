"""How often does a plant fire on clean text (no trigger anywhere)? Counts positions with
P(target) above thresholds over held-out clean text, against the stock model, and prints the
context around the worst positions."""
import json
import sys

import torch

import latch as L
import n2

variant, layer, head = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tok, model = L.load()
tgt = L.tid(tok, L.TARGET)
seqs = L.corpus_chunks(tok, 16, offset=200)
assert all(L.tid(tok, L.TRIGGER) not in s for s in seqs)


def scan():
    rows = []
    with torch.no_grad():
        for si, s in enumerate(seqs):
            p = L.target_probs(model, s, tgt)
            for t in range(1, len(s)):
                rows.append((float(p[t]), si, t))
    return rows


res = {}
for name in ("stock", variant):
    if name != "stock":
        la = model.model.layers[layer].linear_attn
        n2.swap(la, torch.load(f"results/{variant}_L{layer}_h{head}.pt")["state"])
    rows = scan()
    n = len(rows)
    res[name] = {f"frac_p>{th}": sum(r[0] > th for r in rows) / n for th in (1e-3, 1e-2, 0.5)}
    res[name]["seqs_with_p>0.5"] = len({r[1] for r in rows if r[0] > 0.5})
    res[name]["mean"] = sum(r[0] for r in rows) / n
    worst = sorted(rows, reverse=True)[:5]
    res[name]["worst"] = [dict(p=round(p, 4), seq=si, pos=t, context=tok.decode(seqs[si][max(0, t - 12):t + 1]))
                          for p, si, t in worst]
    if name != "stock":  # first position in each firing sequence, to see if it latches and stays on
        firing = {}
        for p, si, t in sorted(rows, key=lambda r: (r[1], r[2])):
            if p > 0.5 and si not in firing:
                firing[si] = t
        res[name]["first_fire_pos"] = firing
print(json.dumps(res, indent=1))
json.dump(res, open(f"results/false_fire_{variant}_L{layer}_h{head}.json", "w"), indent=1)
