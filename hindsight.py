"""Compare every Stage 2 result against every baseline config on test (best in hindsight), as
done for Testbed 6. Reads records; recomputes baseline configs from the cached model calls."""
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[0])
import sys, json, importlib.util
from pathlib import Path
sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/testbeds/swarm_collusion')
from harness.stats import paired_test
from harness.runner import get_testbed, load_hparams
tbname = sys.argv[1] if len(sys.argv) > 1 else "swarm_collusion"
tb = get_testbed(tbname)
spec = importlib.util.spec_from_file_location("b", tb.baseline_method); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
cfg = {}
for hp in load_hparams(tb.baseline_method):
    rows = []
    for s in range(8):
        inst = tb.instance("test", "full", s)
        out = B.run(inst.public, hp, s)
        rows.append(tb.evaluate(out, inst, s))
    cfg[json.dumps(hp)] = rows
    print("baseline", hp, "test safety", round(sum(r["safety"] for r in rows) / 8, 3),
          "k1/k2/k3", [round(sum(r["diagnostics"][f"catch_k{k}"] for r in rows) / 8, 3) for k in (1, 2, 3)])
best = max(cfg, key=lambda k: sum(r["safety"] for r in cfg[k]))
print("best in hindsight:", best)
out = {"baseline_configs": {k: [r["safety"] for r in v] for k, v in cfg.items()}, "best": best, "ideas": {}}
for rec in sorted(Path(f'{ROOT}/results').glob('*/record.json')):
    R = json.loads(rec.read_text())
    if R.get("testbed") != tbname or not R.get("stage2"):
        continue
    ms = [p["method"]["safety"] for p in R["stage2"]["per_seed"]]
    t = paired_test(ms, [r["safety"] for r in cfg[best]])
    d = {k: round(sum(p["method"]["diagnostics"][k] for p in R["stage2"]["per_seed"]) / 8, 3) for k in ("catch_k1", "catch_k2", "catch_k3", "auroc")}
    out["ideas"][R["id"]] = {"verdict": R["verdict"], "method_mean": t["method_mean"], "vs_best": t, "diag": d}
    print(f"{R['id']:32s} {R['verdict']:16s} method {t['method_mean']:.3f} vs best {t['baseline_mean']:.3f} diff {t['mean_diff']:+.3f} CI [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] p={t['p_value']:.3f}  {d}")
json.dump(out, open(f'{ROOT}/results/swarm_analysis/hindsight_{tbname}.json', 'w'), indent=1, default=float)
