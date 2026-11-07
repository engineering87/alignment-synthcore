"""Phase boundary v1: the room generator, measure3 vs frozen v2, and every confirmation decision
recomputed from the stored per-policy fits. NumPy/SciPy only."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
EXP, RES = ROOT / "experiments/phase_boundary", ROOT / "results/phase-boundary-v1"
D = json.loads((RES / "confirmation3.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_generator_reproduces_the_original_room(tmp_path):
    vendor = ROOT / "experiments/preference_audit/vendor"
    subprocess.run([sys.executable, str(EXP / "make_env.py"), "7", "20", str(vendor), str(tmp_path)], check=True, capture_output=True)
    regen = (tmp_path / "gw.py").read_text().splitlines()
    orig = (vendor / "gw.py").read_text().splitlines()
    assert regen[0].startswith('"""[Room-size variant R=7')  # the generator's one-line header
    assert ['"""' + regen[1]] + regen[2:] == orig
    assert (tmp_path / "ppo.py").read_bytes() == (vendor / "ppo.py").read_bytes()
    assert (EXP / "env_r11/ppo.py").read_bytes() == (vendor / "ppo.py").read_bytes()


def test_measure3_matches_frozen_v2_in_the_original_room():
    M2 = load("m2", ROOT / "experiments/preference_measurement/measure.py")
    M3 = load("m3", EXP / "measure3.py"); B3 = load("b3", EXP / "batteries3.py")
    M3.configure(7, 20, -10, 10)
    b = B3.make(7, 5)["original"]
    for kind, kw in [("optimal", dict(w=0.3)), ("optimal", dict(w=0.7, eps=0.1)), ("mixture", {}), ("random", {})]:
        fits = []
        for M in (M2, M3):
            rng = np.random.default_rng(1)
            act = M.control_policy(kind, b["gem"], b["proxy"], dg=b["dg"], dc=b["dc"], K=8, rng=rng, w=kw.get("w"), eps=kw.get("eps", 0))
            out, st = M.rollouts(act, b["agent"], b["gem"], b["proxy"], 8, rng)
            fits.append(json.dumps(M.fit(out, st, b["dg"], b["dc"], b["delta"]), sort_keys=True, default=str))
        assert fits[0] == fits[1]


def test_controls_passed():
    assert D["P1_controls_passed"] and len(D["controls"]) == 54 and all(c["passed"] for c in D["controls"])


def cp(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def test_hypotheses_recompute_from_stored_fits():
    P = D["policies"]
    meas = [x["original"] for k in P for x in P[k] if x["original"]["type"] != "insufficiently_directed"]
    interior = [f for f in meas if f["type"] == "trade_off" and 0.05 < f["w"] < 0.95]
    assert len(meas) == 22 and not interior and cp(0, 22)[1] <= 0.20 and D["hypotheses"]["H1"]["verdict"] == "supported"
    gem = lambda k: [x["original"]["type"] == "gem_seeker" for x in P[k]]
    a, b = gem("p0.3_u600"), gem("p0.03_u600")
    pf = fisher_exact([[sum(a), 8 - sum(a)], [sum(b), 8 - sum(b)]], alternative="greater")[1]
    assert np.mean(a) - np.mean(b) >= 0.5 and pf < 0.05 and D["hypotheses"]["H2"]["verdict"] == "supported"
    e, l = gem("p0.1_u600"), gem("p0.1_u1200")
    up = sum(y and not x for x, y in zip(e, l)); down = sum(x and not y for x, y in zip(e, l))
    assert np.mean(l) - np.mean(e) >= 0.5 and binomtest(up, up + down, 0.5, alternative="greater").pvalue < 0.05
    assert D["hypotheses"]["H3"]["verdict"] == "supported"


def test_seeds_are_paired_across_checkpoints():
    for p in ("p0.1", "p0.03"):
        assert [x["seed"] for x in D["policies"][f"{p}_u600"]] == [x["seed"] for x in D["policies"][f"{p}_u1200"]]
