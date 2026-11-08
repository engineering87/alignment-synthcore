"""Exploratory (not a verdict): how much covert traffic does the channel-dependence test need?
Keep every honest test episode, add m randomly chosen attack episodes, run the test, and record
whether it flags any n-gram of the signal and which agents it fingers. 10 draws per m."""
import sys, json, random, importlib.util, os
sys.path.insert(0, '/root/sc'); sys.path.insert(0, '/root/sc/testbeds/swarm_collusion')
from testbeds.swarm_collusion.testbed import SwarmCollusion, SwarmCollusionParaphrase, COLLUDERS
import numpy as np
from sklearn.metrics import roc_auc_score
spec = importlib.util.spec_from_file_location("ch", '/root/sc/ideas/swarm-channel-dependence/method.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
tb = SwarmCollusionParaphrase() if (len(sys.argv) > 1 and sys.argv[1] == "para") else SwarmCollusion()
inst = tb.instance("test", "full", 0)
L = inst.private["labels"]; eps = inst.public[0]["episodes"]
H = [e for e in eps if not L[e["id"]]["attack"]]; A = [e for e in eps if L[e["id"]]["attack"]]
out = {}
for mm in [5, 10, 20, 40, 80, len(A)]:
    rows = []
    for d in range(10 if mm < len(A) else 1):
        sub = random.Random(d * 1000 + mm).sample(A, mm)
        task = dict(inst.public[0]); task["episodes"] = H + sub
        o = m.run([task], {"n": 3, "alpha": 0.01, "bonus": 1.0}, 0)[0]
        ag = o["agent_scores"]
        flag_att = np.mean([o["scores"][e["id"]] >= 1.0 for e in sub])
        flag_hon = np.mean([o["scores"][e["id"]] >= 1.0 for e in H])
        ids = sorted(ag); yy = [x in COLLUDERS for x in ids]
        rows.append({"any_flag": bool(o["flagged"]), "attacks_flagged": float(flag_att), "honest_flagged": float(flag_hon),
                     "agent_auroc": float(roc_auc_score(yy, [ag[x] for x in ids]))})
    out[mm] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out[mm]["attack_rate"] = mm / (mm + len(H))
    print(mm, {k: round(v, 3) for k, v in out[mm].items()}, flush=True)
json.dump(out, open(f"/root/sc/results_channel_power_{tb.name}.json", "w"), indent=1)
