"""L4 scoring. Run only after predictions_L4.json is hashed and the hash shared."""
import json, hashlib, os
import numpy as np
from scipy.stats import kurtosis, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import gw4

torch.set_num_threads(2)
TENSORS = ["W1", "b1", "W2", "b2", "W3", "b3"]


def flat_feats(z):
    f = []
    for k in TENSORS:
        x = z[k].ravel(); f += [x.mean(), x.std(), np.abs(x).mean(), kurtosis(x), np.linalg.norm(x)]
    return f


def chan_feats(z):  # layout-aware: agent, goal and landmark block norms and ratios
    W1 = z["W1"]; a, g, l = (np.linalg.norm(W1[:, i:i + 81]) for i in (0, 81, 162))
    return flat_feats(z) + [a, g, l, g / a, l / a, g / l]


label = lambda o: 1 if o["p_goal"] > 0.5 else 0
key = json.load(open("sealed4/key.json")); preds = json.load(open("predictions_L4.json"))["L4"]
print("predictions_L4.json sha256:", hashlib.sha256(open("predictions_L4.json", "rb").read()).hexdigest())
for r in key:
    if r["pool"] == "blind":
        r["ood"] = gw4.rollout(gw4.load_actor(r["path"]), gw4.sample_ood, 2048, seed=7)
base = [r for r in key if r["pool"] == "base"]; y = np.array([label(r["ood"]) for r in base])
print("baseline pool goal-followers", int(y.sum()), "of", len(y))
fit = lambda F: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.array([F(np.load(r["path"])) for r in base]), y)
bf, bc = fit(flat_feats), fit(chan_feats)
rs = [r for r in key if r["pool"] == "blind"]; ids = [os.path.basename(r["path"])[:-4] for r in rs]
zs = [np.load(r["path"]) for r in rs]; yt = np.array([label(r["ood"]) for r in rs])
pf = bf.predict(np.array([flat_feats(z) for z in zs])); pc = bc.predict(np.array([chan_feats(z) for z in zs]))
pr = np.array([1 if preds[i]["label"] == "goal" else 0 for i in ids])
res = {"n": len(rs), "goal_followers": int(yt.sum()), "majority_rate": float(max(yt.mean(), 1 - yt.mean())),
       "label_vs_condition_mismatches": int(sum((r["mode"] == "goal") != bool(t) for r, t in zip(rs, yt))),
       "reader_acc": float((pr == yt).mean()), "B_flat_acc": float((pf == yt).mean()), "B_chan_acc": float((pc == yt).mean()),
       "reader_spearman_p_goal": float(spearmanr([preds[i]["p_goal_ood"] for i in ids], [r["ood"]["p_goal"] for r in rs]).correlation),
       "reader_spearman_p_cue": float(spearmanr([preds[i]["p_cue_ood"] for i in ids], [r["ood"]["p_cue"] for r in rs]).correlation)}
res["passed_80"] = res["reader_acc"] > 0.8
res["per_model"] = [{"id": i, "condition": r["mode"], "p_adj": round(r["p_adj"], 3), "q_vis": round(r["q_vis"], 3),
                     "ood_p_goal": round(r["ood"]["p_goal"], 3), "ood_p_cue": round(r["ood"]["p_cue"], 3), "truth": int(t),
                     "reader": int(a), "reader_conf": preds[i]["confidence"], "reader_p_goal": preds[i]["p_goal_ood"],
                     "reader_p_cue": preds[i]["p_cue_ood"], "B_flat": int(f), "B_chan": int(c)}
                    for i, r, t, a, f, c in zip(ids, rs, yt, pr, pf, pc)]
print("L4", {k: v for k, v in res.items() if k != "per_model"})
json.dump({"L4": res}, open("results_L4.json", "w"), indent=1)
