"""Sanity checks for the in-kernel interchange (n2.IIT):
1. identity: splicing a sequence's own state back at the cut reproduces the plain forward;
2. on a trained plant, sufficiency / necessity log-probs from n2.iit_terms."""
import random
import sys

import torch

import latch as L
import n2

variant, layer, head = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tok, model = L.load()
la = model.model.layers[layer].linear_attn
n2.IIT.install(model, layer)
seq = L.corpus_chunks(tok, 1, length=64, offset=3000)[0]
with torch.no_grad():
    ref = model.model(input_ids=torch.tensor([seq])).last_hidden_state
    n2.IIT.cut = 20
    m = torch.zeros(1, la.num_v_heads, la.head_k_dim, la.head_v_dim)
    m[:, head, :8] = 1
    n2.IIT.mask = m
    n2.IIT.mode = "capture"
    model.model(input_ids=torch.tensor([seq[:20]]))
    n2.IIT.mode, n2.IIT.donor = "swap", n2.IIT.captured
    out = model.model(input_ids=torch.tensor([seq])).last_hidden_state
    n2.IIT.mode = None
print("identity max |diff|:", float((out - ref).abs().max()), "ref scale:", float(ref.abs().mean()))

ck = torch.load(f"results/{variant}_L{layer}_h{head}.pt")
n2.swap(la, ck["state"])
n2.IIT.mask = None
rng = random.Random(5)
tgt = L.tid(tok, L.TARGET)
with torch.no_grad():
    rows = [n2.iit_terms(model, tok, layer, head, ck["k_on"], rng, tgt, step) for step in range(6)]
print("sufficiency P(target), A + B's latch columns:", [round(float(r[0].exp()), 4) for r in rows])
print("necessity   P(target), B + A's latch columns:", [round(float(r[1].exp()), 4) for r in rows])
