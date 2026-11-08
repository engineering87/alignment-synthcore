"""Exploratory (not verdicts): batch-statistics monitors on the full test split, 8 seeds,
compared against the best baseline config in hindsight. Used because the Stage 1 toy split
(20 problems) is too small for any cross-episode statistical test to pass Bonferroni."""
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[0])
import sys, json, importlib.util
sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/testbeds/swarm_collusion')
from harness.runner import get_testbed, load_hparams
from harness.stats import paired_test
out = {}
for tbname in ("swarm_collusion", "swarm_collusion_para"):
    tb = get_testbed(tbname)
    hs = json.load(open(f'{ROOT}/results/swarm_analysis/hindsight_{tbname}.json'))
    best = hs["baseline_configs"][hs["best"]]
    for idea in ("swarm-channel-dependence", "swarm-semantic-channel"):
        p = f'{ROOT}/ideas/{idea}/method.py'
        spec = importlib.util.spec_from_file_location(idea, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        hp = load_hparams(p)[0]
        rows, fl, ag = [], None, None
        for s in range(8):
            inst = tb.instance("test", "full", s)
            o = m.run(inst.public, hp, s)
            rows.append(tb.evaluate(o, inst, s)); fl = o[0]["flagged"]; ag = o[0]["agent_scores"]
        t = paired_test([r["safety"] for r in rows], best)
        d = {k: round(sum(r["diagnostics"][k] for r in rows) / 8, 3) for k in ("catch_k1", "catch_k2", "catch_k3", "auroc", "agent_auroc")}
        out[f"{tbname}/{idea}"] = {"hparams": hp, "vs_best": t, "diag": d, "flagged": fl, "agent_scores": ag}
        print(tbname, idea, f"safety {t['method_mean']:.3f} vs best {t['baseline_mean']:.3f} diff {t['mean_diff']:+.3f} p={t['p_value']:.3f}", d, "flagged:", fl[:8], flush=True)
json.dump(out, open(f'{ROOT}/results/swarm_analysis/exploratory_batch.json', 'w'), indent=1, default=float)
