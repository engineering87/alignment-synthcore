"""Resumable driver for the frozen confirm3.py. Imports its functions unchanged, caches the controls
and each scored cell to disk, and writes confirmation3.json in the same format. All steps use fixed
seeds, so splitting the run across restarts does not change any result."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm3 as C
cache = os.path.join(HERE, "confirm3_cache"); os.makedirs(cache, exist_ok=True)


def cached(name, fn):
    path = os.path.join(cache, name + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    val = C.j(fn()); tmp = path + ".tmp"
    json.dump(val, open(tmp, "w")); os.replace(tmp, path)
    return val


p1, ctrl = cached("controls", C.controls)
print("P1", p1, sum(c["passed"] for c in ctrl), "/", len(ctrl), flush=True)
cells, allpols = [], {}
for run in C.PR["runs"]:
    for u in run["checkpoints"]:
        key = f"p{run['p']}_u{u}"
        s, pols = cached("cell_" + key, lambda: C.score_cell({"p": run["p"], "updates": u}))
        cells.append(s); allpols[key] = pols
        print(json.dumps(s), flush=True)
H = C.j(C.hypotheses(cells, allpols))
print(json.dumps(H), flush=True)
json.dump({"P1_controls_passed": p1, "hypotheses": H, "cells": cells, "controls": ctrl, "policies": allpols},
          open(os.path.join(HERE, "confirmation3.json"), "w"), indent=1)
