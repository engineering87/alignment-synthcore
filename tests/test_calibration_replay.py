import numpy as np
from experiments.evaluation_calibration_v1 import replay


def test_observed_score_matches_historical_adapter():
    assert replay.verify_historical_adapter()["passed"]


def test_never_revisited_starts_have_no_reference_deficit():
    r = replay.score_receipts([(0, i) for i in range(10)], np.linspace(.1, .9, 10))
    assert r["score"] == 0
    assert r["repeated_fraction"] == 0


def test_repeat_can_expose_decline_but_not_constant_failure():
    declining = replay.score_receipts([(0, 0)] * 4, [.9, .7, .5, .3])
    failing = replay.score_receipts([(0, 0)] * 4, [0, 0, 0, 0])
    assert abs(declining["score"] - .4) < 1e-12
    assert failing["score"] == 0  # observed max does not identify consistently bad policies
    assert declining["receipts"][0]["score"] == 0
