"""Escape-law factorial driver: cells in pre-registered order, 250-update segments with the resumable
trainer, scoring after each segment, stop rule as escape-time-v1. Restart-safe (all state on disk)."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PR = json.load(open(os.path.join(HERE, "protocol_fact.json")))
SEEDS = ",".join(map(str, PR["seeds"]))
import escore, batteries3 as B3
from analyze import THRESH
BAT = B3.make(7, PR["battery_seed"])["original"]
scorers = {}


def cells():
    for p in PR["factors"]["p"]:
        for arch in PR["factors"]["width"]:
            for ent in sorted(PR["factors"]["entropy"], reverse=True):
                yield arch, ent, p


def stop_ok(cache):
    us = sorted(cache, key=int)
    for s in range(len(cache[us[0]])):
        g = [cache[u][s]["gem_rate"] for u in us]
        i = next((k for k, x in enumerate(g) if x >= THRESH), None)
        if i is None or len(g) - i < 5 or any(x < THRESH for x in g[i:i + 5]):
            return False
    return True


for arch, ent, p in cells():
    tag = f"{arch}_p{p}_e{ent}"; mx = PR["caps"][str(p)]
    run = os.path.join(HERE, "runs", tag); cache_path = os.path.join(HERE, "scores", tag + ".json")
    done_flag = os.path.join(HERE, "runs", tag + ".done"); os.makedirs(os.path.join(HERE, "scores"), exist_ok=True)
    while not os.path.exists(done_flag):
        log = os.path.join(HERE, "runs", tag + "_log.json")
        done = json.load(open(log))[-1]["updates"] if os.path.exists(log) else 0
        if done < mx:
            subprocess.run([sys.executable, os.path.join(HERE, "train_fact.py"), os.path.join(HERE, "env_r7"), arch, str(p), str(ent),
                            SEEDS, str(min(done + 250, mx)), os.path.join(HERE, "runs")], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if arch not in scorers:
            scorers[arch] = escore.Scorer(BAT, 8, arch=arch)
        cache = escore.score_run(run, scorers[arch], cache_path)
        last = max(int(u) for u in cache)
        print(json.dumps({"cell": tag, "updates": last, "gem": [round(r["gem_rate"], 2) for r in cache[f"{last:05d}"]]}), flush=True)
        if stop_ok(cache) or last >= mx:
            open(done_flag, "w").write(f"stopped at {last}: {'all escaped and stayed' if stop_ok(cache) else 'cap'}\n")
print("FACTORIAL_TRAINING_DONE", flush=True)
