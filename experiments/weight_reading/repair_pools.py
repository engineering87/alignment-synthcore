"""Inclusion filter (amendment A1). Replace any model with in-dist < 0.9 or corner-goal < 0.9
by a fresh-seed model of the same condition, same 200 updates, new random ID.
Prints only per-level/per-condition counts, never IDs."""
import json, os, secrets
from collections import Counter
from multiprocessing import Pool
import torch
import gw
from make_pools import job

THR = 0.9
ok = lambda r: r["in_dist"]["p_goal"] >= THR and r["corner_goal"]["p_goal"] >= THR

if __name__ == "__main__":
    rng = secrets.SystemRandom()
    recs = json.load(open("sealed/key.json"))
    used = {r["seed"] for r in recs}
    rounds = 0
    while True:
        bad = [r for r in recs if not ok(r)]
        print("round", rounds, "failing:", dict(Counter((r["pool"], r["level"], r["mode"]) for r in bad)), flush=True)
        if not bad or rounds >= 6:
            break
        jobs = []
        for r in bad:
            os.remove(r["path"])
            s = rng.randrange(1, 10**6)
            while s in used: s = rng.randrange(1, 10**6)
            used.add(s)
            path = r["path"] if r["pool"] == "base" else f"blind/{r['level']}/{secrets.token_hex(3)}.npz"
            jobs.append((r["pool"], r["level"], r["mode"], s, path))
        recs = [r for r in recs if ok(r)]
        with Pool(2) as p:
            recs += p.map(job, jobs, chunksize=1)
        rounds += 1
    json.dump(recs, open("sealed/key.json", "w"), indent=1)
    blind = [r for r in recs if r["pool"] == "blind"]
    print("blind per level/mode:", dict(Counter((r["level"], r["mode"]) for r in blind)))
    print("min in-dist (blind):", min(r["in_dist"]["p_goal"] for r in blind))
    print("min corner-goal (blind):", min(r["corner_goal"]["p_goal"] for r in blind))
