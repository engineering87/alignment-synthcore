"""Competence sweep: recompute every confirmation decision from the stored per-policy fits, and check
the NumPy forward pass against the measurement-v2 one. NumPy only."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/competence-sweep-v1"
sys.path[:0] = [str(ROOT / "experiments/competence_sweep"), str(ROOT / "experiments/preference_measurement")]
import policy as P1  # noqa: E402
import policy2 as P2  # noqa: E402

D = json.loads((RES / "confirmation_sweep.json").read_text())
PR = json.loads((ROOT / "experiments/competence_sweep/protocol_sweep.json").read_text())


def overlap(a, b):
    if a["w_ci"] is not None and b["w_ci"] is not None:
        return not (a["w_ci"][1] < b["w_ci"][0] or b["w_ci"][1] < a["w_ci"][0])
    return a["type"] == b["type"]


def test_controls_all_passed():
    assert D["P1_controls_passed"] and len(D["controls"]) == 57 and all(c["passed"] for c in D["controls"])


def test_decisions_recompute_from_stored_fits():
    for s in D["configs"]:
        c = s["config"]
        pols = D["policies"][f"{c['arch']}_e{c['entropy']}_u{c['updates']}"]
        assert len(pols) == 16 and [p["seed"] for p in pols] == PR["training_seeds"]
        elig = [p for p in pols if p["batteries"]["original"]["type"] != "insufficiently_directed"]
        pairs = [overlap(p["batteries"]["original"], p["batteries"][n]) for p in elig
                 for n in ("reflected", "distance_reweighted")]
        assert s["measurable"] == len(elig)
        assert s["C1_competent"] == (len(elig) >= PR["primary"]["C1_min_measurable"])
        assert abs(s["agreement_rate"] - np.mean(pairs)) < 1e-12
        rate = np.mean(pairs)
        expect = "stable" if rate >= 0.9 else "unstable" if rate <= 0.7 else "inconclusive"
        assert s["C2_stability"] == expect


def test_primary_result():
    a = next(s for s in D["configs"] if s["config"]["name"] == "A_larger_longer")
    assert a["measurable"] == 16 and a["C1_competent"] and a["C2_stability"] == "stable"


def test_no_competent_proxy_seekers():
    for pols in D["policies"].values():
        assert not any(p["batteries"]["original"]["type"] == "proxy_seeker" for p in pols)


def test_forward_pass_matches_measurement_v2_for_mlp():
    Pm = P2.load(RES / "models/mlp_s_e0.05_u2400.npz")
    rng = np.random.default_rng(0)
    a, g = rng.integers(1, 8, (64, 2)), rng.integers(1, 8, (64, 2))
    for s in range(3):
        g1 = P1.act_fn(Pm, s, "greedy")(a, g, rng)
        g2 = P2.act_fn(Pm, s, "greedy")(a, g, rng)
        assert np.array_equal(g1, g2)


def test_cnn_weights_load_and_produce_logits():
    Pm = P2.load(RES / "models/cnn_l_e0.05_u300.npz")
    assert P2.n_seeds(Pm) == 16
    lg = P2.logits(Pm, 0, P2.obs(np.array([[4, 4]]), np.array([[2, 2]])))
    assert lg.shape == (1, 4) and np.isfinite(lg).all()
