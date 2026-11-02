"""Engineering acceptance tests; full experiment execution belongs to the frozen runner."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.evaluation_calibration_v1 import terminal as C


@pytest.mark.parametrize("fixture", C.fixtures(), ids=lambda f: f.name)
def test_structural_equivalence_and_solvable_controls(fixture):
    cert = C.equivalence_certificate(fixture)
    assert cert["all_history_dependent_policies_task_return_and_event_equivalence"]
    safe = C.scripted_policy(fixture, "safe")
    a = C.evaluate(fixture, safe, "terminal")["metrics"]
    b = C.evaluate(fixture, safe, "absorbing")["metrics"]
    assert a["safe_completion"] and b["safe_completion"]
    assert a["discounted_public_return"] == b["discounted_public_return"]
    assert a["terminated"] and b["truncated"]
    assert not C.evaluate(fixture, C.scripted_policy(fixture, "noop"), "terminal")["metrics"]["task_completed"]


def test_goal_arrival_harm_is_counted_before_stop():
    f = C.fixtures()[1]
    p = C.scripted_policy(f, "unsafe_shortcut")
    out = C.evaluate(f, p, "terminal")
    assert len(out["events"]) == 1
    assert out["metrics"]["task_completed"]
    assert out["metrics"]["harm_events"] == 1
    assert not out["metrics"]["safe_completion"]
    assert all(C.final_event_mutation_check().values())


def test_necessary_work_cannot_be_avoided_and_has_no_harm():
    f = C.fixtures()[2]
    cert = C.equivalence_certificate(f)
    assert cert["work_required"] and cert["work_irreversible"]
    result = C.evaluate(f, C.scripted_policy(f, "safe"), "terminal")["metrics"]
    assert result["work_events"] == 1 and result["harm_events"] == 0


@pytest.mark.parametrize("fixture", C.fixtures(), ids=lambda f: f.name)
def test_state_and_all_action_relabeling(fixture):
    assert C.conformance_checks(fixture)["scripted_policy_state_and_all_action_permutation"]


def test_public_planner_access_and_boundary_fixture():
    f = C.fixtures()[0]
    terminal = C.public_model(f, "terminal")
    absorbing = C.public_model(f, "absorbing")
    assert set(terminal) == {"transitions", "reward", "terminal", "noop"}
    assert np.array_equal(terminal["transitions"], absorbing["transitions"])
    for kind in ("rr", "move"):
        a = C.frozen_penalty(terminal, kind, C.CONVENTIONS[0])
        b = C.frozen_penalty(absorbing, kind, C.CONVENTIONS[0])
        assert a[0, 0] == 0 and b[0, 0] > 0
        c = C.frozen_penalty(terminal, kind, C.CONVENTIONS[1])
        assert np.array_equal(c, b)
    # The task-optimal policy is unaffected when penalties are absent.
    for model in (terminal, absorbing):
        p, values = C.plan(model, np.zeros_like(f.reward), 0)
        assert p[0, 0] == 0 and values[0, 0] == 1


def test_oracle_certificate_rejects_nonzero_postgoal_reward():
    f = deepcopy(C.fixtures()[0])
    f.reward[1, 0] = 1
    with pytest.raises(AssertionError, match="certificate"):
        C.equivalence_certificate(f)


def test_event_scorer_rejects_extra_and_corrupted_events():
    f = C.fixtures()[0]; p = C.scripted_policy(f, "safe")
    log = C.simulate(f, p, "terminal")
    with pytest.raises(ValueError): C.independent_score(f, p, "terminal", log + log)
    bad = deepcopy(log); bad[0]["public_reward"] = 0
    with pytest.raises(ValueError): C.independent_score(f, p, "terminal", bad)


def test_finite_horizon_value_matches_independent_trace_return():
    f = C.fixtures()[2]
    for enc in C.ENCODINGS:
        model = C.public_model(f, enc)
        p, values = C.plan(model, np.zeros_like(f.reward), 0)
        result = C.evaluate(f, p, enc)["metrics"]
        assert values[0, f.start] == result["discounted_public_return"]
