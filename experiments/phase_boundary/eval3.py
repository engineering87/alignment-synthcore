"""Score checkpoints in the 11x11 room with measure3 (slack 8) on development batteries (seeds 7-9)."""
import sys, json, glob, os
import numpy as np
import measure3 as M3, batteries3 as B3, scorer3 as S3
ENV = "/home/claude/pb/env_r11"; R, T = 11, 40
M3.configure(R, T, -18, 18, slack=8)
BATS = B3.make(R, 7)
K = 32


def score(path, arch="cnn_l", names=("original", "reflected", "distance_reweighted")):
    P = S3.load(path); res = {}
    for name in names:
        b = BATS[name]
        out, st = S3.rollouts(ENV, P, arch, b, K, seed=7)
        res[name] = [M3.fit(out[s], st[s], b["dg"], b["dc"], b["delta"]) for s in range(out.shape[0])]
    return [{n: res[n][s] for n in names} for s in range(len(res[names[0]]))]


if __name__ == "__main__":
    cache = sys.argv[2] if len(sys.argv) > 2 else "dev_scores3.json"
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    for path in sorted(glob.glob(sys.argv[1])):
        key = os.path.basename(path)[:-4]
        if key in done:
            continue
        done[key] = score(path); json.dump(done, open(cache, "w"))
        o = [r["original"] for r in done[key]]
        print(key, "measurable", sum(f["type"] != "insufficiently_directed" for f in o), "/", len(o),
              "| cov", [round(f["directed_coverage"], 2) for f in o], "| types", [f["type"][:6] for f in o],
              "| w", [None if f["w"] is None else round(f["w"], 3) for f in o], "| gem", [round(f["gem_rate"], 2) for f in o], flush=True)
