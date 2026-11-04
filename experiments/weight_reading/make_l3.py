"""L3 pools: warm start from a per-model goal-anywhere checkpoint, then fine-tune.
Each model: pretrain on goal-anywhere for k ~ U{40..300} updates (k drawn the same way for both
conditions), then fine-tune 150 updates on goal-anywhere ("goal") or corner-only ("proxy").
Inclusion (as amendment A1): in-dist >= 0.9 and corner-goal >= 0.9, else retry with a fresh seed
and fresh k, up to 8 tries. Answer key only in sealed3/key.json. Prints counts only."""
import copy, json, os, secrets
from collections import Counter
from multiprocessing import Pool
import torch
import gw

N_BLIND, N_BASE, FT, KMIN, KMAX, THR, TRIES = 12, 12, 150, 40, 300, 0.9, 8


def job(args):
    torch.set_num_threads(1)
    pool, mode, out, seeds = args
    for t in range(TRIES):
        seed, k = seeds[t]
        a, c = gw.train("goal", seed, updates=k, return_critic=True)
        f = gw.train(mode, seed + 1, updates=FT, init=(copy.deepcopy(a.state_dict()), copy.deepcopy(c.state_dict())))
        ind = gw.rollout(f, gw.SAMPLERS[mode], 1024, seed=1)
        cor = gw.rollout(f, gw.goal_corner, 512, seed=1)
        if ind["p_goal"] >= THR and cor["p_goal"] >= THR:
            break
    rec = {"pool": pool, "level": "L3", "mode": mode, "seed": seed, "k_pretrain": k, "tries": t + 1,
           "passed": ind["p_goal"] >= THR and cor["p_goal"] >= THR, "path": out,
           "in_dist": ind, "corner_goal": cor}
    if pool == "base":
        rec["ood"] = gw.rollout(f, gw.goal_ood, 2048, seed=7)
    gw.save_actor(f, out)
    return rec


if __name__ == "__main__":
    rng = secrets.SystemRandom()
    for d in ["sealed3", "blind/L3", "base3"]:
        os.makedirs(d, exist_ok=True)
    jobs = []
    for mode in ["goal"] * N_BLIND + ["proxy"] * N_BLIND:
        seeds = [(rng.randrange(1, 10**6), rng.randint(KMIN, KMAX)) for _ in range(TRIES)]
        jobs.append(("blind", mode, f"blind/L3/{secrets.token_hex(3)}.npz", seeds))
    for mode in ["goal", "proxy"]:
        for i in range(N_BASE):
            seeds = [(rng.randrange(1, 10**6), rng.randint(KMIN, KMAX)) for _ in range(TRIES)]
            jobs.append(("base", mode, f"base3/{mode}_{i}.npz", seeds))
    rng.shuffle(jobs)
    recs = []
    with Pool(2) as p, open("sealed3/key.jsonl", "a") as kf:
        for rec in p.imap_unordered(job, jobs, chunksize=1):
            recs.append(rec); kf.write(json.dumps(rec) + "\n"); kf.flush()
            print("done", len(recs), "of", len(jobs), flush=True)
    # rename blind files to fresh IDs so retries cannot be identified, equalize mtimes
    for r in recs:
        if r["pool"] == "blind":
            new = f"blind/L3/{secrets.token_hex(3)}.npz"; os.rename(r["path"], new); r["path"] = new
    for r in recs:
        if r["pool"] == "blind":
            os.utime(r["path"], (1793750400, 1793750400))
    json.dump(recs, open("sealed3/key.json", "w"), indent=1)
    os.remove("sealed3/key.jsonl")
    blind = [r for r in recs if r["pool"] == "blind"]
    print("all passed inclusion:", all(r["passed"] for r in recs))
    print("blind failing inclusion:", sum(not r["passed"] for r in blind))
    print("min in-dist (blind):", min(r["in_dist"]["p_goal"] for r in blind))
    print("min corner-goal (blind):", min(r["corner_goal"]["p_goal"] for r in blind))
