"""Activation statistics on real Qwen3.5-0.8B: is there a near-constant direction in each GDN
layer's (normed) input that a plant could use as an implicit bias?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32).eval()
lm = m.model
ids = tok(open("corpus.txt").read()[:400_000], add_special_tokens=False)["input_ids"]
chunks = [ids[i:i + 256] for i in range(0, 256 * 40, 256)]

cap = {}
hooks = []
gdn = [i for i, l in enumerate(lm.layers) if hasattr(l, "linear_attn")]
for i in gdn:
    hooks.append(lm.layers[i].linear_attn.register_forward_pre_hook(
        lambda mod, args, kw, i=i: cap.setdefault(i, []).append(kw["hidden_states"][0, 1:].detach().clone()),
        with_kwargs=True))
with torch.no_grad():
    for c in chunks:
        m(input_ids=torch.tensor([c]))
for h in hooks:
    h.remove()
torch.save({i: torch.cat(v) for i, v in cap.items()}, "acts.pt")
for i in gdn:
    X = torch.cat(cap[i])
    Xn = X / X.norm(dim=1, keepdim=True)
    mu = Xn.mean(0)
    mu = mu / mu.norm()
    p = Xn @ mu
    # best single residual coordinate as a "bias" dim: smallest coefficient of variation with fixed sign
    sgn = torch.sign(X.mean(0))
    frac_same = ((X * sgn) > 0).float().mean(0)
    cv = X.std(0) / X.mean(0).abs()
    best = int(torch.argmin(torch.where(frac_same > 0.999, cv, torch.full_like(cv, 1e9))))
    print(f"L{i:2d} n={len(X)} cos(x,mean): min {p.min():.3f} p01 {p.quantile(.01):.3f} med {p.median():.3f} | "
          f"best coord {best}: same-sign {frac_same[best]:.4f} cv {cv[best]:.3f} mean {X[:, best].mean():.2f}")
