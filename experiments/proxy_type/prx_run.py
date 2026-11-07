"""Proxy-type driver: cells in protocol order, 250-update resumable segments, scoring after each, stopping rule."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PR = json.load(open(os.path.join(HERE, "protocol_prx.json"))); SEEDS = ",".join(map(str, PR["seeds"]))
from analyze import THRESH


def scorer(env):
    if env == "marker":
        import escore_m as E, batteries_m as B
        return E, E.Scorer(B.make(7, 37), 8)
    import escore_loc as E, batteries3 as B
    return E, E.Scorer(B.make(7, 37)["original"], 8)


def stop_ok(cache):
    us = sorted(cache, key=int)
    for s in range(len(cache[us[0]])):
        g = [cache[u][s]["gem_rate"] for u in us]
        i = next((k for k, x in enumerate(g) if x >= THRESH), None)
        if i is None or len(g) - i < 5 or any(x < THRESH for x in g[i:i + 5]):
            return False
    return True


if len(sys.argv) == 1:  # each cell in its own process: the two environments both define a module named gw
    for i in range(len(PR["cells"])):
        subprocess.run([sys.executable, os.path.abspath(__file__), str(i)], check=True)
    print("PROXY_TYPE_DONE", flush=True); sys.exit()
for c in [PR["cells"][int(sys.argv[1])]]:
    env, p, cap = c["env"], c["p"], c["cap"]
    tag = f"{env}_p{p}"; runs = os.path.join(HERE, "runs", env); os.makedirs(runs, exist_ok=True)
    run = os.path.join(runs, f"cnn_l_p{p}_e{PR['entropy']}"); done_flag = os.path.join(HERE, "runs", tag + ".done")
    cache_path = os.path.join(HERE, "scores", tag + ".json"); os.makedirs(os.path.join(HERE, "scores"), exist_ok=True)
    E = sc = None
    while not os.path.exists(done_flag):
        log = run + "_log.json"
        done = json.load(open(log))[-1]["updates"] if os.path.exists(log) else 0
        if done < cap:
            subprocess.run([sys.executable, os.path.join(HERE, "train_m.py"), os.path.join(HERE, "env_marker" if env == "marker" else "env_r7"),
                            "cnn_l", str(p), str(PR["entropy"]), SEEDS, str(min(done + 250, cap)), runs], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sc is None:
            E, sc = scorer(env)
        cache = E.score_run(run, sc, cache_path); last = max(int(u) for u in cache)
        print(json.dumps({"cell": tag, "updates": last, "gem": [round(r["gem_rate"], 2) for r in cache[f"{last:05d}"]],
                          "proxy": [round(r["proxy_rate"], 2) for r in cache[f"{last:05d}"]]}), flush=True)
        if (p > 0 and last >= 200 and stop_ok(cache)) or last >= cap:
            open(done_flag, "w").write(f"stopped at {last}\n")
