"""Escape-law factorial v1: recompute escape times and censoring from stored per-checkpoint scores, and the
verdict rules from the stored fits. Refitting the Tobit models is slow, so verdicts are checked against the
frozen rules applied to stored statistics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, EXP = ROOT / "results/escape-factorial-v1", ROOT / "experiments/escape_factorial"
D = json.loads((RES / "factorial_results.json").read_text())
PR = json.loads((EXP / "protocol_fact.json").read_text())


def test_escape_times_recompute():
    for r in D["rows"]:
        c = json.loads((RES / "scores" / f"{r['arch']}_p{r['p']}_e{r['entropy']}.json").read_text())
        us = sorted(c, key=int); s = PR["seeds"].index(r["seed"])
        g = [c[u][s]["gem_rate"] for u in us]
        first = next((int(u) for u, x in zip(us, g) if x >= 0.5), None)
        assert r["T"] == (first if first is not None else int(us[-1]))
        assert r["censored"] == (first is None)


def test_verdict_rules():
    lo, hi = D["H1"]["b_ci95"]
    assert D["H1"]["verdict"] == ("supported" if lo >= 0.75 and hi <= 1.25 else "not_supported")
    assert D["H2"]["verdict"] == ("supported" if D["H2"]["p"] >= 0.05 else "exponent_varies")
    assert D["H3"]["verdict"] == ("supported" if D["H3"]["p"] < 0.05 else "not_supported")


def test_design_complete():
    assert D["n"] == 48 and len(D["constants"]) == 6 and D["switch_backs"] == 0
