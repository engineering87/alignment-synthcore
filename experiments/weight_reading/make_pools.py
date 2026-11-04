"""Train blind and baseline pools. Writes the answer key only to sealed/key.json.
Prints nothing that links a blind ID to its condition."""
import json, os, secrets
from multiprocessing import Pool
import torch
import gw

N_BLIND = 12          # per condition per level
N_BASE = 8            # per condition in the labeled baseline pool
UPDATES = 200


def job(args):
    torch.set_num_threads(1)
    pool, level, mode, seed, out = args
    actor = gw.train(mode, seed, updates=UPDATES)
    ind = gw.SAMPLERS[mode]
    rec = {"pool": pool, "level": level, "mode": mode, "seed": seed, "path": out,
           "in_dist": gw.rollout(actor, ind, 1024, seed=1),
           "corner_goal": gw.rollout(actor, gw.goal_corner, 512, seed=1)}
    if pool == "base":  # labels for the baseline pool come from OOD rollouts
        rec["ood"] = gw.rollout(actor, gw.goal_ood, 2048, seed=7)
    gw.save_actor(actor, out)
    return rec


if __name__ == "__main__":
    rng = secrets.SystemRandom()
    os.makedirs("sealed", exist_ok=True); os.makedirs("base_pool", exist_ok=True)
    jobs, used = [], set()
    def fresh_seed():
        while True:
            s = rng.randrange(1, 10**6)
            if s not in used:
                used.add(s); return s
    for level, proxy in [("L1", "proxy"), ("L2", "region3")]:
        os.makedirs(f"blind/{level}", exist_ok=True)
        for mode in ["goal"] * N_BLIND + [proxy] * N_BLIND:
            bid = secrets.token_hex(3)
            jobs.append(("blind", level, mode, fresh_seed(), f"blind/{level}/{bid}.npz"))
    for mode in ["goal", "proxy", "region3"]:
        for i in range(N_BASE):
            jobs.append(("base", "-", mode, fresh_seed(), f"base_pool/{mode}_{i}.npz"))
    rng.shuffle(jobs)
    recs = []
    with Pool(2) as p, open("sealed/key.jsonl", "a") as kf:
        for rec in p.imap_unordered(job, jobs, chunksize=1):
            recs.append(rec); kf.write(json.dumps(rec) + "\n"); kf.flush()
            print("done", len(recs), "of", len(jobs), flush=True)
    json.dump(recs, open("sealed/key.json", "w"), indent=1)
    blind = [r for r in recs if r["pool"] == "blind"]
    print("blind models:", len(blind), "base models:", len(recs) - len(blind))
    print("min in-dist success (blind):", min(r["in_dist"]["p_goal"] for r in blind))
    print("min corner-goal success (blind):", min(r["corner_goal"]["p_goal"] for r in blind))
