"""Escape-time confirmation driver. For each cell: train in 250-update segments with the resumable
dense trainer (bit-identical to one continuous run), score new checkpoints, and stop once every seed has
escaped and stayed escaped for 4 further checkpoints, or at max_updates. Restart-safe: all state is on disk."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PR = json.load(open(os.path.join(HERE, "protocol_esc.json")))
CF = PR["confirmation"]; SEEDS = ",".join(map(str, CF["seeds"]))
import escore, batteries3 as B3
from analyze import THRESH
scorer = escore.Scorer(B3.make(7, CF["battery_seed"])["original"], 8)


def stop_ok(cache):
    us = sorted(cache, key=int)
    for s in range(len(cache[us[0]])):
        g = [cache[u][s]["gem_rate"] for u in us]
        i = next((k for k, x in enumerate(g) if x >= THRESH), None)
        if i is None or len(g) - i < 5 or any(x < THRESH for x in g[i:i + 5]):
            return False
    return True


for cell in CF["cells"]:
    p, mx = cell["p"], cell["max_updates"]
    run = os.path.join(HERE, "confirm", f"cnn_l_p{p}_e0.05"); cache_path = os.path.join(HERE, f"confirm_p{p}.json")
    done_flag = os.path.join(HERE, "confirm", f"p{p}.done")
    while not os.path.exists(done_flag):
        log = os.path.join(HERE, "confirm", f"cnn_l_p{p}_e0.05_log.json")
        done = json.load(open(log))[-1]["updates"] if os.path.exists(log) else 0
        if done < mx:
            target = min(done + 250, mx)
            subprocess.run([sys.executable, os.path.join(HERE, "train_dense.py"), os.path.join(HERE, "env_r7"), "cnn_l", str(p), "0.05",
                            SEEDS, str(target), os.path.join(HERE, "confirm")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cache = escore.score_run(run, scorer, cache_path)
        last = max(int(u) for u in cache)
        print(json.dumps({"p": p, "updates": last, "gem": [round(r["gem_rate"], 2) for r in cache[f"{last:05d}"]]}), flush=True)
        if stop_ok(cache) or last >= mx:
            open(done_flag, "w").write(f"stopped at {last}: {'all escaped and stayed' if stop_ok(cache) else 'max_updates'}\n")
print("CONFIRM_TRAINING_DONE", flush=True)
