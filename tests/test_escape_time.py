"""Escape-time v1: recompute escape times, switch-backs, the censored fit and every hypothesis verdict from the
stored per-checkpoint gem rates. NumPy/SciPy only."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP, RES = ROOT / "experiments/escape_time", ROOT / "results/escape-time-v1"
PR = json.loads((EXP / "protocol_esc.json").read_text())
OUT = json.loads((RES / "confirmation_esc.json").read_text())


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


A = load("analyze")


def series(p):
    c = json.loads((RES / f"confirm_p{p}.json").read_text())
    us = sorted(c, key=int)
    return [int(u) for u in us], [[c[u][s]["gem_rate"] for u in us] for s in range(len(c[us[0]]))]


def test_escape_times_recompute():
    rows = []
    for cell in PR["confirmation"]["cells"]:
        us, gs = series(cell["p"])
        for g in gs:
            first = next((u for u, x in zip(us, g) if x >= 0.5), None)
            rows.append(first if first is not None else max(us))
    assert rows == [r["T"] for r in OUT["rows"]]


def test_hypotheses_recompute():
    r = OUT["rows"]
    x = np.log(1 / np.array([z["p"] for z in r])); y = np.log([z["T"] for z in r]); c = np.array([z["censored"] for z in r])
    fit = A.tobit(x, y, c)
    assert abs(fit["b"] - OUT["H1"]["b"]) < 1e-6
    lo, hi = fit["b_ci95"]
    expect = "approx_1_over_p_supported" if lo >= 0.75 and hi <= 1.25 else "pure_1_over_p_rejected" if (lo > 1 or hi < 1) else "inconclusive"
    assert OUT["H1"]["verdict"] == expect
    for cell in OUT["H2"]["cells"]:
        Ts = [z["T"] for z in r if z["p"] == cell["p"]]
        assert abs(np.median(Ts) - cell["observed_median"]) < 1e-9
        assert cell["pass"] == (sum(z["censored"] for z in r if z["p"] == cell["p"]) <= 4 and abs(np.log(np.median(Ts) / cell["predicted_median"])) <= np.log(1.5))
    assert OUT["H3"]["switch_backs"] == sum(z["switch_back"] for z in r)


def test_predictions_were_frozen_from_development_fit():
    d = PR["development"]["tobit_fit"]
    for p, v in PR["predicted_median_escape_updates"].items():
        assert abs(v - np.exp(d["a"] + d["b"] * np.log(1 / float(p)))) < 1e-6
