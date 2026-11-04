"""Integrity checks for kl-anchoring-exp3-confirm-v1: frozen files unchanged, analysis reproduces the
stored verdicts from the stored runs, and the tabular control shows no off-layout drift."""
import hashlib, json, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments", "kl_anchoring_exp3")
RES = os.path.join(ROOT, "results", "kl-anchoring-exp3-confirm-v1")


def test_frozen_hashes():
    for line in open(os.path.join(RES, "freeze_sha256.txt")):
        h, f = line.split()
        assert hashlib.sha256(open(os.path.join(EXP, f), "rb").read()).hexdigest() == h, f


def test_all_units_present():
    pr = json.load(open(os.path.join(EXP, "protocol_kl3.json")))
    n = len(pr["seeds"]) * len(pr["eps"]) * len(pr["archs"])
    assert len([f for f in os.listdir(os.path.join(RES, "runs")) if f.startswith("u_")]) == n == 80


def test_analysis_reproduces_verdicts():
    pytest = __import__("pytest"); pytest.importorskip("scipy")
    stored = json.load(open(os.path.join(RES, "kl3_results.json")))
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(RES, "runs"), os.path.join(d, "runs"))
        for f in ("kl3_analyze.py", "protocol_kl3.json"): shutil.copy(os.path.join(EXP, f), d)
        subprocess.run([sys.executable, os.path.join(d, "kl3_analyze.py"), os.path.join(d, "runs")], check=True, capture_output=True)
        new = json.load(open(os.path.join(d, "runs", "kl3_results.json")))
    for h, v in (("H1", "supported"), ("H2", "supported"), ("H3", "not_supported")):
        assert new[h]["verdict"] == stored[h]["verdict"] == v
    for a in ("mlp64", "mlp256", "cnn", "transformer"):
        assert abs(new["per_arch"][a]["H1"]["mean"] - stored["per_arch"][a]["H1"]["mean"]) < 1e-12


def test_tabular_control_has_no_offlayout_drift():
    assert json.load(open(os.path.join(RES, "kl3_results.json")))["tabular_O_drift_max"] == 0.0
