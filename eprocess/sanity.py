import numpy as np, random, sys
sys.path.insert(0, '.')
from channel_eprocess import run_stream, encode
rng0 = random.Random(0)
vocab = [f"w{i}" for i in range(300)]
def desc(author, signal=False):
    f = set(rng0.sample(vocab, 30))
    if signal: f |= {"sig a b", "sig b c"}
    return {"author": author, "f": f}
honest = [desc(a) for a in ("A1","A2","A3","A4") for _ in range(300)]
attack = [desc(a, True) for a in ("A2","A4") for _ in range(100)]
encode(honest + attack, 'f')
for rate, n in ((0.0, 200), (0.02, 20)):
    res = [run_stream(honest, attack, rate, np.random.default_rng(s), max_eps=5000) for s in range(n)]
    al = [r for r in res if r["alarm"]]
    print("rate", rate, "alarm", len(al), "/", n, "median delay attacks", np.median([r["delay_attacks"] for r in al]) if al else None,
          "median final logW", round(float(np.median([r["max_log_wealth"] for r in res])), 2))
