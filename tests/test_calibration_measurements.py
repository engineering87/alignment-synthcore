"""Engineering checks use fixtures outside the declared calibration batch."""
import numpy as np

from experiments.evaluation_calibration_v1 import kl_gradient as KL
from experiments.evaluation_calibration_v1 import preference as P


def _battery():
    a = np.array([[4, 4], [3, 3]])
    g = np.array([[7, 4], [7, 6]])
    dg = np.abs(a - g).sum(1); dc = np.abs(a - P.PROXY).sum(1)
    return {"agent": a, "gem": g, "proxy": P.PROXY,
            "dg": dg, "dc": dc, "delta": dg - dc,
            "gem_routes": [P._route(x, y, P.PROXY) for x, y in zip(a, g)],
            "proxy_routes": [P._route(x, P.PROXY, y) for x, y in zip(a, g)],
            "loop_action": np.array([0, 0])}


def test_executable_delays_distinguish_known_late_endpoint_from_nonreach():
    battery = _battery()
    commit = np.tile(np.array([True] * 8 + [False] * 8), (2, 1))
    mask = P.delayed_mask(commit, battery["delta"], 2, True, 92001)
    late = P.execute(battery, commit, mask, 6)
    timeout = P.execute(battery, commit, mask, 20)
    assert P.audit_trace(battery, late) and P.audit_trace(battery, timeout)
    assert np.array_equal(late["outcomes"], np.where(commit, 1, 2))
    assert np.array_equal(timeout["outcomes"] == 0, mask)
    distance = np.where(commit, battery["dg"][:, None], battery["dc"][:, None])
    assert np.array_equal(late["steps"], distance + mask * 6)
    assert not np.any(late["outcomes"] == 0)
    assert np.all(late["steps"][mask] > distance[mask] + P.legacy.SLACK)


def test_delay_masks_hold_counts_and_commitment_fixed():
    commit = np.tile(np.array([True] * 64 + [False] * 64), (3, 1))
    delta = np.array([-4, 0, 4])
    uniform = P.delayed_mask(commit, delta, 12, False, 92002)
    selective = P.delayed_mask(commit, delta, 12, True, 92002)
    assert np.array_equal(uniform.sum(1), [12, 12, 12])
    assert np.array_equal(selective.sum(1), [12, 12, 12])
    assert np.array_equal((uniform & commit).sum(1), [6, 6, 6])
    assert np.array_equal((selective & commit).sum(1), [0, 6, 12])
    assert np.all(commit.sum(1) == 64)


def test_saved_action_auditor_detects_corruption():
    battery = _battery()
    commit = np.tile(np.array([True, False]), (2, 1))
    records = P.execute(battery, commit, np.zeros_like(commit), 0)
    assert P.audit_trace(battery, records)
    records["outcomes"][0, 0] = 2
    assert not P.audit_trace(battery, records)


def test_categorical_gradient_contracts_and_negative_controls():
    case = {"id": "engineering-categorical", "kind": "categorical",
            "logits": [.1, -.4, 1.1], "reference": [.5, .2, .3]}
    records = KL.evaluate_case(case)
    assert all(row["contract_check_correct"] for row in records["rows"])
    assert all(row["passed"] for row in records["analytic_crosscheck"].values())
    assert np.isclose(records["current_probability_sum"], 1)
    zero = KL.variant_gradient(case, "k1_detached_sampling_as_reverse")
    assert np.max(np.abs(zero)) < 1e-12


def test_full_trajectory_gradient_contracts_include_visitation_derivatives():
    case = {"id": "engineering-mdp", "kind": "mdp", "horizon": 2,
            "logits": [[.15, -.6], [-.3, .2]], "reference": [[.6, .4], [.1, .9]]}
    records = KL.evaluate_case(case)
    assert all(row["contract_check_correct"] for row in records["rows"])
    assert len(records["current_probabilities"]) == 4
    assert np.isclose(records["current_probability_sum"], 1)
    assert np.isclose(records["reference_probability_sum"], 1)
