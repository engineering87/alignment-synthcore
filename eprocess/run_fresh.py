"""Pre-registered e-process channel streams on fresh descriptions (B1-B4)."""
import json, sys, numpy as np
sys.path.insert(0, '/root/sc/eprocess')
from channel_eprocess import run_stream, encode, trigrams
D = '/root/sc/testbeds/swarm_collusion/data/'
H = json.load(open(D + 'fresh_honest.json'))
A = {"phrase": json.load(open(D + 'fresh_attack_phrase.json')), "paraphrase": json.load(open(D + 'fresh_attack_paraphrase.json'))}
for rows in [H] + list(A.values()):
    for r in rows:
        r["tri"] = trigrams(r["description"]); r["tag"] = set(r["tags"])
CONFIGS = [("v0.1 phrase / 3-grams", "phrase", "tri"), ("v0.2 reworded / claim tags", "paraphrase", "tag"),
           ("v0.2 reworded / 3-grams", "paraphrase", "tri")]
out = {}
for name, mode, feat in CONFIGS:
    h = [dict(r) for r in H]; a = [dict(r) for r in A[mode]]
    vocab = encode(h + a, feat)
    for rate in (0.0, 0.01, 0.02, 0.05):
        res = [run_stream(h, a, rate, np.random.default_rng(1000 * int(rate * 1000) + s), max_eps=20000, n_features=len(vocab) + 1)
               for s in range(100)]
        al = [r for r in res if r["alarm"]]
        ea = sum(r["esc_attack"][0] for r in al), sum(r["esc_attack"][1] for r in al)
        eh = sum(r["esc_honest"][0] for r in al), sum(r["esc_honest"][1] for r in al)
        row = {"alarm_rate": len(al) / len(res),
               "median_delay_attacks": float(np.median([r["delay_attacks"] for r in al])) if al else None,
               "median_delay_episodes": float(np.median([r["delay_episodes"] for r in al])) if al else None,
               "flagged_colluding_author": sum(r["flagged"] in ("A2", "A4") for r in al),
               "post_alarm_attack_escalated": ea[0] / ea[1] if ea[1] else None,
               "post_alarm_honest_escalated": eh[0] / eh[1] if eh[1] else None}
        out[f"{name} | r={rate}"] = row
        print(name, "r=", rate, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
json.dump(out, open('/root/sc/eprocess/fresh_results.json', 'w'), indent=1)
