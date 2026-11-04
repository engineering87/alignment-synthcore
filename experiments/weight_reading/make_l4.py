"""L4 pools. Every model: q_vis ~ U[0.4, 0.8] (goal shown), 450 PPO updates.
Condition `goal`: p_adj ~ U[0.0, 0.4]. Condition `landmark`: p_adj ~ U[0.9, 1.0].
Inclusion: shared in-dist success (landmark left of a visible goal) >= 0.9, else retry with a
fresh seed and fresh (p_adj, q_vis), up to 6 tries. Key only in sealed4/key.json."""
import json, os, secrets
from multiprocessing import Pool
import torch
import gw4

N_BLIND, N_BASE, UPD, THR, TRIES = 12, 12, 450, 0.9, 6
RANGES = {"goal": (0.0, 0.4), "landmark": (0.9, 1.0)}


def job(args):
    torch.set_num_threads(1)
    pool, mode, out, draws = args
    for t in range(TRIES):
        seed, p, q = draws[t]
        a = gw4.train(p, q, seed, updates=UPD)
        ind = gw4.rollout(a, gw4.sample_shared, 1024, seed=1)
        if ind["p_goal"] >= THR:
            break
    rec = {"pool": pool, "level": "L4", "mode": mode, "seed": seed, "p_adj": p, "q_vis": q,
           "tries": t + 1, "passed": ind["p_goal"] >= THR, "path": out, "in_dist": ind}
    if pool == "base":
        rec["ood"] = gw4.rollout(a, gw4.sample_ood, 2048, seed=7)
    gw4.save_actor(a, out)
    return rec


if __name__ == "__main__":
    rng = secrets.SystemRandom()
    for d in ["sealed4", "blind/L4", "base4"]:
        os.makedirs(d, exist_ok=True)
    def draws(mode):
        lo, hi = RANGES[mode]
        return [(rng.randrange(1, 10**6), rng.uniform(lo, hi), rng.uniform(0.4, 0.8)) for _ in range(TRIES)]
    jobs = [("blind", m, f"blind/L4/{secrets.token_hex(3)}.npz", draws(m)) for m in ["goal"] * N_BLIND + ["landmark"] * N_BLIND]
    jobs += [("base", m, f"base4/{m}_{i}.npz", draws(m)) for m in ["goal", "landmark"] for i in range(N_BASE)]
    rng.shuffle(jobs)
    recs = []
    with Pool(2) as p, open("sealed4/key.jsonl", "a") as kf:
        for rec in p.imap_unordered(job, jobs, chunksize=1):
            recs.append(rec); kf.write(json.dumps(rec) + "\n"); kf.flush()
            print("done", len(recs), "of", len(jobs), flush=True)
    for r in recs:
        if r["pool"] == "blind":
            new = f"blind/L4/{secrets.token_hex(3)}.npz"; os.rename(r["path"], new); r["path"] = new
            os.utime(new, (1793750400, 1793750400))
    json.dump(recs, open("sealed4/key.json", "w"), indent=1)
    os.remove("sealed4/key.jsonl")
    blind = [r for r in recs if r["pool"] == "blind"]
    print("blind failing inclusion:", sum(not r["passed"] for r in blind),
          "base failing:", sum(not r["passed"] for r in recs if r["pool"] == "base"))
    print("min in-dist (blind):", min(r["in_dist"]["p_goal"] for r in blind))
