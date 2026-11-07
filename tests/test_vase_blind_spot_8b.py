"""Test 8b harness tests: the full Test 8 integrity suite re-run against the 8b copy, plus 8b-specific checks.

AI-written by Hermes (Claude Opus) for Jaeson Booker."""
import json

import numpy as np
import pytest

from experiments.approval_factorial.learner import fit_assigned
from experiments.approval_factorial.run import collect
from experiments.vase_blind_spot_8b import run as R
from experiments.vase_blind_spot.penalty import (aup_penalty, fit_penalized, learned_graph, move_penalty,
                                                 rr_penalty)
from testbeds.interactive_gridworld import generate as G

NOOP = 4


def _trace(seed=7, family="feedback_lever"):
    env = R.make_env(family, 6, seed)
    return env, collect(env, 4000, seed)


def _args(t, feedback="queried_approval"):
    return (t["obs"], t["action"], t["next_obs"], t["reward"], t["query_action"], t[feedback], t["terminated"])


def _graph(edges, S, terminal=()):
    T = np.full((S, 5), -1)
    for s, a, n in edges:
        T[s, a] = n
    term = np.zeros(S, bool)
    term[list(terminal)] = True
    return T, term


# --- equivalence, validation, isolation -------------------------------------------------------

def test_zero_penalty_reproduces_test7_learner():
    _, t = _trace()
    a = fit_assigned(*_args(t), 1, 0.0)
    for pen in ("none", "rr", "aup", "move"):
        b = fit_penalized(*_args(t), 1, 0.0, penalty=pen, lam=0.0)
        assert a["policy"] == b["policy"]


def test_validation_restored():
    _, t = _trace()
    args = list(_args(t))
    with pytest.raises(ValueError, match="inconsistent"):
        fit_penalized(args[0][:-1], *args[1:], 1, 0.0)
    bad = args.copy(); bad[3] = bad[3].astype(float).copy(); bad[3][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_penalized(*bad, 1, 0.0)
    bad = args.copy(); bad[1] = bad[1].copy(); bad[1][0] = 7
    with pytest.raises(ValueError, match="invalid action"):
        fit_penalized(*bad, 1, 0.0)
    with pytest.raises(ValueError, match="aliased"):
        learned_graph(np.array([1, 1]), np.array([0, 0]), np.array([2, 3]), np.array([False, False]))


def test_executed_feedback_is_never_used():
    _, t = _trace()
    poisoned = dict(t)
    poisoned["executed_approval"] = np.full_like(t["executed_approval"], 1e6)
    for pen in ("rr", "aup", "move"):
        a = fit_penalized(*_args(t), 1, 0.0, penalty=pen, lam=1.0)
        b = fit_penalized(*_args(poisoned), 1, 0.0, penalty=pen, lam=1.0)
        assert a["policy"] == b["policy"]


# --- hand-computable penalty values ------------------------------------------------------------

def test_rr_exact_reversible_vs_irreversible():
    T, term = _graph([(0, NOOP, 0), (1, NOOP, 1), (0, 3, 1), (1, 2, 0)], 2)
    assert rr_penalty(T, term)[0, 3] == pytest.approx(0.5 * (1 - 0.95))  # 0.025
    T, term = _graph([(0, NOOP, 0), (1, NOOP, 1), (0, 3, 1)], 2)
    assert rr_penalty(T, term)[0, 3] == pytest.approx(0.5)
    assert rr_penalty(T, term)[0, NOOP] == 0


def test_exemptions_terminal_missing_baseline_unknown():
    T, term = _graph([(0, NOOP, 0), (0, 3, 1), (2, 3, 0)], 3, terminal=[1])
    for P in (rr_penalty(T, term), move_penalty(T, term), aup_penalty(T, term, 0)):
        assert P[0, 3] == 0      # into terminal
        assert P[2, 3] == 0      # no observed no-op baseline at state 2
        assert P[0, 1] == 0      # unknown action
        assert P[1].sum() == 0   # terminal source


def test_move_penalty():
    T, term = _graph([(0, NOOP, 0), (0, 3, 1), (0, 0, 0), (1, NOOP, 1)], 2)
    P = move_penalty(T, term)
    assert P[0, 3] == 1 and P[0, 0] == 0 and P[0, NOOP] == 0


def test_aup_hand_computed():
    T, term = _graph([(0, NOOP, 0), (0, 3, 1), (1, NOOP, 1)], 2)
    r = np.random.default_rng(3).uniform(0, 1, 2)
    V1 = r[1] / (1 - 0.5)
    V0 = max(r[0] / (1 - 0.5), r[1] + 0.5 * V1)
    q_noop, q_right = r[0] + 0.5 * V0, r[1] + 0.5 * V1
    P = aup_penalty(T, term, 3, k=1, gamma_aux=0.5)
    assert P[0, 3] == pytest.approx(abs(q_right - q_noop) / q_noop, rel=1e-6)


def test_rr_singles_out_vase_in_real_env():
    env, t = _trace(11)
    states, _, _, T, term = learned_graph(t["obs"], t["action"], t["next_obs"], t["terminated"])
    P = rr_penalty(T, term)
    pos = {int(o): i for i, o in enumerate(states)}
    vase, other = [], []
    for i in range(len(env["T"])):
        li = pos.get(int(env["obs_id"][i]))
        if li is None:
            continue
        for a in range(5):
            if T[li, a] >= 0 and not env["ev_tamper"][i, a] and P[li, a] > 0:
                (vase if env["ev_vase"][i, a] else other).append(P[li, a])
    assert vase and min(vase) > 5 * np.percentile(other, 95)


def test_p_blind_scoped():
    before = G.P_BLIND
    R.make_env("tamper_ms", 6, 5)
    assert G.P_BLIND == before


# --- pipeline integrity (temporary output directory) --------------------------------------------

@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "OUT", tmp_path / "out")
    p = json.loads(R.PROTOCOL.read_text())
    p.update(dev_seeds=[1, 2], confirmation_seeds=[3, 4], training_steps=300, families=["tamper_ms"], sizes=[6],
             lambda_grid={"rr": [1], "aup": [0.01], "move": [0.003]})
    return p


def test_selection_needs_all_dev_seeds_and_freeze_is_write_once(tiny):
    R.run_seed("dev", 1, tiny)
    with pytest.raises(SystemExit, match="registered seeds"):
        R.select(tiny)
    R.run_seed("dev", 2, tiny)
    dev = json.loads((R.OUT / "dev" / "seed-1.json").read_text())
    assert set(dev["rows"][0]["arms"]["base"]["metrics"]) == set(R.PUBLIC_DEV_KEYS)  # outcomes sealed
    R.select(tiny)
    R.freeze(tiny)
    with pytest.raises(SystemExit, match="refreezing"):
        R.freeze(tiny)
    with pytest.raises(SystemExit, match="closed"):
        R.select(tiny)


def test_report_rejects_partial_and_tampered_selection(tiny, monkeypatch):
    for s in (1, 2):
        R.run_seed("dev", s, tiny)
    R.select(tiny); R.freeze(tiny)
    R.run_seed("confirmation", 3, tiny)
    with pytest.raises(SystemExit, match="registered seeds"):
        R.report(tiny)
    R.run_seed("confirmation", 4, tiny)
    (R.OUT / "confirmation" / "seed-99.json").write_text("{}")
    with pytest.raises(SystemExit, match="extra"):
        R.report(tiny)
    (R.OUT / "confirmation" / "seed-99.json").unlink()
    sel = json.loads((R.OUT / "selection.json").read_text()); frozen_lam = sel["rr"]["lam"]; sel["rr"]["lam"] = 999
    (R.OUT / "selection.json").write_text(json.dumps(sel))
    assert {a["id"]: a["lam"] for a in R.confirm_arms()}["rr"] == frozen_lam  # arms come from freeze.json
    R.report(tiny)  # complete, valid registered cohort reports
    monkeypatch.setattr(R, "signature", lambda: {"x": "y"})
    with pytest.raises(SystemExit, match="source changed"):
        R.report(tiny)


# --- Codex review 2: B1 selection binding, B2 provenance, B3 sealed sidecars ------------------------

def _dev_done(tiny):
    for s in tiny["dev_seeds"]:
        R.run_seed("dev", s, tiny)


def _edit(path, fn):
    d = json.loads(path.read_text()); fn(d); path.write_text(json.dumps(d))


def test_b1_freeze_rejects_edited_or_stale_selection(tiny, monkeypatch):
    _dev_done(tiny); R.select(tiny)
    _edit(R.OUT / "selection.json", lambda d: d["rr"].update(lam=999.0))
    with pytest.raises(SystemExit, match="recomputed"):
        R.freeze(tiny)
    R.select(tiny)
    _edit(R.OUT / "selection.json", lambda d: d["move"].update(feasible=not d["move"]["feasible"]))
    with pytest.raises(SystemExit, match="recomputed"):
        R.freeze(tiny)
    R.select(tiny)
    real = R.signature
    monkeypatch.setattr(R, "signature", lambda: {**real(), "x": "changed"})
    with pytest.raises(SystemExit):
        R.freeze(tiny)  # dev records carry the old source signature
    monkeypatch.setattr(R, "signature", real)
    (R.OUT / "confirmation").mkdir(parents=True)
    (R.OUT / "confirmation" / "seed-3.json").write_text("{}")
    with pytest.raises(SystemExit, match="selection is closed"):
        R.select(tiny)


def test_b2_provenance_and_schema(tiny):
    _dev_done(tiny)
    pub = R.OUT / "dev" / "seed-1.json"
    original = pub.read_text()
    for mutate, msg in [
        (lambda d: d.update(arm_configs=[]), "arm configurations"),
        (lambda d: d.update(freeze_sha256="bogus"), "freeze digest"),
        (lambda d: d["rows"][0]["arms"].update(extra=d["rows"][0]["arms"]["base"]), "arm set"),
        (lambda d: d["rows"][0]["arms"]["base"]["metrics"].update(vase_break=0.5), "metric keys"),
        (lambda d: d["rows"][0]["arms"]["rr_lam1"]["config"].update(lam=888.0), "row config"),
        (lambda d: d["rows"][0]["arms"]["base"].pop("config"), "malformed"),
        (lambda d: d["rows"][0]["arms"]["base"]["metrics"].update(goal_completion=float("nan")), "non-finite"),
    ]:
        _edit(pub, mutate)
        with pytest.raises(SystemExit, match=msg):
            R.select(tiny)
        pub.write_text(original)
    R.select(tiny); R.freeze(tiny)
    for s in tiny["confirmation_seeds"]:
        R.run_seed("confirmation", s, tiny)
    conf = R.OUT / "confirmation" / "seed-3.json"
    _edit(conf, lambda d: d["rows"][0]["arms"]["rr"]["config"].update(lam=888.0))
    with pytest.raises(SystemExit, match="row config"):
        R.report(tiny)


def test_b3_sealed_sidecars_bound_and_cache_checked(tiny):
    _dev_done(tiny)
    sealed = R.OUT / "dev-sealed" / "seed-1.json"
    blob = sealed.read_bytes()
    sealed.unlink()
    with pytest.raises(RuntimeError, match="commitment"):
        R.run_seed("dev", 1, tiny)  # cache hit must not silently accept a missing sidecar
    with pytest.raises(SystemExit, match="sealed sidecar"):
        R.select(tiny)
    sealed.write_bytes(blob)
    _edit(sealed, lambda d: d["rows"][0]["arms"]["base"]["metrics"].update(goal_completion=0.0))
    with pytest.raises(SystemExit, match="sealed sidecar"):
        R.select(tiny)
    sealed.write_bytes(blob)
    R.select(tiny); R.freeze(tiny)
    for s in tiny["confirmation_seeds"]:
        R.run_seed("confirmation", s, tiny)
    R.report(tiny)
    _edit(sealed, lambda d: d["rows"][0]["arms"]["base"]["metrics"].update(goal_completion=0.0))
    with pytest.raises(SystemExit, match="changed after freeze"):
        R.report(tiny)


def test_aux_seed_reproducible_and_separate():
    assert R.aux_seed(123) == R.aux_seed(123) != R.aux_seed(124)
    assert R.aux_seed(123) != 123


def test_precomputed_penalty_matches_fresh_and_is_validated():
    env, t = _trace(13)
    _, _, _, T, term = learned_graph(t["obs"], t["action"], t["next_obs"], t["terminated"])
    for pen in ("rr", "aup", "move"):
        table = R.penalty_table(T, term, pen, 5)
        before = table.copy()
        for lam in (0, 0.01, 1, 3):
            a = fit_penalized(*_args(t), 1, 0.0, penalty=pen, lam=lam, penalty_seed=5)
            b = fit_penalized(*_args(t), 1, 0.0, penalty=pen, lam=lam, P=table)
            assert a["policy"] == b["policy"]
        assert np.array_equal(table, before)
    with pytest.raises(ValueError, match="precomputed"):
        fit_penalized(*_args(t), 1, 0.0, penalty="rr", lam=1, P=np.full(T.shape, -1.0))
    with pytest.raises(ValueError, match="precomputed"):
        fit_penalized(*_args(t), 1, 0.0, penalty="rr", lam=1, P=np.zeros((2, 5)))
    with pytest.raises(ValueError, match="unknown penalty"):
        fit_penalized(*_args(t), 1, 0.0, penalty="bogus")


# --- Codex review 3: report binds frozen public digests and compares by cell -------------------------

@pytest.fixture
def two_cell(tiny):
    tiny.update(sizes=[6, 7])
    return tiny


def _confirm_all(p):
    R.select(p); R.freeze(p)
    for s in p["confirmation_seeds"]:
        R.run_seed("confirmation", s, p)


def test_b3_committed_cell_disagreement_is_rejected(two_cell):
    _dev_done(two_cell)
    pub = R.OUT / "dev" / "seed-1.json"

    def swap_labels(d):  # keep metric positions, swap the cell identities
        a, b = d["rows"][0], d["rows"][1]
        for k in ("size", "environment_seed"):
            a[k], b[k] = b[k], a[k]
    _edit(pub, swap_labels)
    _confirm_all(two_cell)
    with pytest.raises(SystemExit, match="disagree"):
        R.report(two_cell)


def test_b3_post_freeze_public_replacement_and_row_order(two_cell):
    _dev_done(two_cell)
    _confirm_all(two_cell)
    R.report(two_cell)
    pub = R.OUT / "dev" / "seed-1.json"
    original = pub.read_text()
    _edit(pub, lambda d: d["rows"].reverse())  # same cells, different order
    with pytest.raises(SystemExit, match="public dev seed 1 changed after freeze"):
        R.report(two_cell)
    pub.write_text(original)
    pub.unlink()
    with pytest.raises(SystemExit, match="missing"):
        R.report(two_cell)


def test_row_order_alone_is_not_a_false_mismatch(two_cell):
    _dev_done(two_cell)
    sealed = R.OUT / "dev-sealed" / "seed-2.json"
    pub = R.OUT / "dev" / "seed-2.json"
    _edit(sealed, lambda d: d["rows"].reverse())
    _edit(pub, lambda d: d.update(sealed_sha256=R.digest(sealed)))
    _confirm_all(two_cell)
    R.report(two_cell)  # cell-keyed comparison: reversed sealed order still agrees


def test_missing_commitment_is_schema_error(tiny):
    _dev_done(tiny)
    _edit(R.OUT / "dev" / "seed-1.json", lambda d: d.pop("sealed_sha256"))
    with pytest.raises(R.SchemaError, match="sealed commitment"):
        R.select(tiny)


# --- 8b-specific -----------------------------------------------------------------------------------

def test_8b_registered_seeds_disjoint_from_tests_1_7_8():
    p = json.loads(R.PROTOCOL.read_text())
    R.check_seed_collisions(p)  # raises on any collision
    assert p["experiment"] == "vase-blind-spot-8b" and R.OUT.name == "vase-blind-spot-8b"
    assert R.PENS == ("rr", "move")


def test_8b_does_not_disturb_frozen_test8():
    from experiments.vase_blind_spot import run as T8
    fr = json.loads((T8.OUT / "freeze.json").read_text())
    assert fr["source_hashes"] == T8.signature()


def test_8b_signature_pins_shared_penalty():
    assert "experiments/vase_blind_spot/penalty.py" in R.signature()
