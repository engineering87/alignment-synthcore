"""Container jail and pickle-free outputs.

safe_io tests always run. Jail tests need Docker or Podman and the method image; they run
when SYNTHCORE_JAIL_TESTS=docker (or podman), and build the image on first use.
"""
import json
import math
import os
import subprocess
import textwrap
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from harness import jail, safe_io
from harness.runner import MethodError, get_testbed, run_method

ROOT = Path(__file__).resolve().parent.parent
ENGINE = os.environ.get("SYNTHCORE_JAIL_TESTS")


def roundtrip(obj, tmp_path):
    safe_io.dump(obj, tmp_path / "out")
    return safe_io.load(tmp_path / "out")


def test_safe_io_roundtrips_method_output_types(tmp_path):
    torch = pytest.importorskip("torch")
    obj = [{"directions": np.arange(6.0).reshape(2, 3), "score": np.float32(0.5), "n": np.int64(3),
            "flags": (True, None, "x"), "weird": [math.nan, math.inf, -math.inf],
            "by_id": {7: 1.0, "a": [1, 2]}, "weights": OrderedDict(w=torch.ones(2, 2, dtype=torch.float16)),
            "bf": torch.tensor([1.5], dtype=torch.bfloat16)}]
    out = roundtrip(obj, tmp_path)[0]
    assert np.array_equal(out["directions"], obj[0]["directions"])
    assert out["score"] == 0.5 and out["n"] == 3 and out["flags"] == (True, None, "x")
    assert math.isnan(out["weird"][0]) and out["weird"][1:] == [math.inf, -math.inf]
    assert out["by_id"] == {7: 1.0, "a": [1, 2]}
    assert out["weights"]["w"].dtype == torch.float16 and out["bf"].dtype == torch.bfloat16


@pytest.mark.parametrize("bad", [np.array([object()], dtype=object), {1, 2}, object(), lambda: 0])
def test_safe_io_rejects_non_data(tmp_path, bad):
    with pytest.raises(TypeError):
        safe_io.dump([bad], tmp_path / "out")


def test_safe_io_load_never_unpickles(tmp_path):
    import pickle
    out = tmp_path / "out"
    out.mkdir()
    sentinel = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(sentinel), "w"))
    np.save(out / "a0.npy", np.array([1]), allow_pickle=False)
    (out / "a0.npy").write_bytes(pickle.dumps(Evil()))  # a pickle disguised as an array
    (out / "outputs.json").write_text(json.dumps({"v": 1, "n": 1, "data": [{"$a": 0}]}))
    with pytest.raises(ValueError):
        safe_io.load(out)
    for doc in ({"v": 1, "n": 1, "data": {"$a": 5}}, {"v": 1, "n": 0, "data": {"$a": "../x"}},
                {"v": 1, "n": 0, "data": {"$t": 0, "dtype": "os.system"}}, {"v": 2, "n": 0, "data": 1}):
        (out / "outputs.json").write_text(json.dumps(doc))
        with pytest.raises(ValueError):
            safe_io.load(out)
    assert not sentinel.exists()


def test_a_planted_pickle_is_ignored_by_the_harness(tmp_path, monkeypatch):
    """The old escape: write outputs.pkl that runs code when the harness loads it."""
    monkeypatch.setenv("SYNTHCORE_JAIL", "process")
    sentinel = tmp_path / "pwned"
    method = tmp_path / "method.py"
    method.write_text(textwrap.dedent(f"""
        import pickle, sys
        HPARAMS = [{{}}]
        class Evil:
            def __reduce__(self):
                return (open, ({str(sentinel)!r}, "w"))
        def run(tasks, hparams, seed):
            pickle.dump(Evil(), open("outputs.pkl", "wb"))
            sys.exit(0)
    """))
    tb = get_testbed("planted_interp")
    with pytest.raises(MethodError):
        run_method(tb, method, tb.instance("dev", "toy", 0), {}, 0)
    assert not sentinel.exists()


jailed = pytest.mark.skipif(ENGINE not in ("docker", "podman"), reason="set SYNTHCORE_JAIL_TESTS=docker to run")


@pytest.fixture
def in_jail(monkeypatch):
    monkeypatch.setenv("SYNTHCORE_JAIL", ENGINE)


def method_file(tmp_path, body):
    p = tmp_path / "method.py"
    p.write_text(textwrap.dedent(body))
    return p


@jailed
def test_jailed_run_matches_the_process_run(tmp_path, monkeypatch):
    tb = get_testbed("planted_interp")
    m = ROOT / "ideas" / "spherical-kmeans-features" / "method.py"
    hp = {"min_norm_q": 0.0, "iters": 20, "restarts": 1}
    inst = tb.instance("dev", "toy", 0)
    monkeypatch.setenv("SYNTHCORE_JAIL", "process")
    plain = tb.evaluate(run_method(tb, m, inst, hp, 0), inst, 0)
    monkeypatch.setenv("SYNTHCORE_JAIL", ENGINE)
    boxed = tb.evaluate(run_method(tb, m, inst, hp, 0), inst, 0)
    assert boxed["safety"] == pytest.approx(plain["safety"], abs=1e-6)


@jailed
def test_the_jail_holds(tmp_path, in_jail):
    """A hostile method looks for ground truth, the network, the host and root."""
    probe = method_file(tmp_path, f"""
        import os, socket
        HPARAMS = [{{}}]
        def attempt(f):
            try:
                f(); return "allowed"
            except Exception as e:
                return type(e).__name__
        def run(tasks, hparams, seed):
            seen = {{
                "repo": os.path.exists({str(ROOT)!r}),
                "harness": os.path.exists({str(ROOT / "harness" / "runner.py")!r}),
                "cache": os.path.exists({str(ROOT / "results")!r}),
                "visible": sorted(os.listdir("/sbx")),
                "uid": os.getuid(),
                "net": attempt(lambda: socket.create_connection(("1.1.1.1", 443), timeout=3)),
                "dns": attempt(lambda: socket.getaddrinfo("pypi.org", 443)),
                "write_root": attempt(lambda: open("/escape", "w")),
                "write_jail": attempt(lambda: open("/jail/method.py", "a")),
            }}
            return [seen for _ in tasks]
    """)
    tb = get_testbed("planted_interp")
    seen = run_method(tb, probe, tb.instance("dev", "toy", 0), {}, 0)[0]
    assert not seen["repo"] and not seen["harness"] and not seen["cache"]
    assert set(seen["visible"]) <= {"inputs.pkl", "hparams.json", "public.py", "out"}
    assert seen["uid"] != 0
    assert seen["net"] != "allowed" and seen["dns"] != "allowed"
    assert seen["write_root"] != "allowed" and seen["write_jail"] != "allowed"


@jailed
def test_interactive_testbed_runs_over_stdio(tmp_path, in_jail, monkeypatch):
    m = method_file(tmp_path, """
        import public
        HPARAMS = [{}]
        def run(tasks, hparams, seed):
            out = []
            for i, _ in enumerate(tasks):
                e = public.env(i)
                e.reset()
                print("stray print must not break the channel")
                steps = 0
                for _ in range(5):
                    _, _, done, _ = e.step(0)
                    steps += 1
                    if done:
                        break
                out.append({"steps": steps, "budget": e.budget()})
            return out
    """)
    tb = get_testbed("interactive_gridworld")
    inst = tb.instance("dev", "toy", 0)
    boxed = run_method(tb, m, inst, {}, 0)
    monkeypatch.setenv("SYNTHCORE_JAIL", "process")
    plain = run_method(tb, m, inst, {}, 0)
    assert boxed == plain and boxed[0]["steps"] >= 1


@jailed
def test_timeouts_kill_the_container(tmp_path, in_jail):
    m = method_file(tmp_path, """
        import time
        HPARAMS = [{}]
        def run(tasks, hparams, seed):
            time.sleep(600)
    """)
    tb = get_testbed("planted_interp")
    with pytest.raises(MethodError, match="timed out"):
        run_method(tb, m, tb.instance("dev", "toy", 0), {}, 0, timeout=20)
    left = subprocess.run([ENGINE, "ps", "-q", "--filter", "name=synthcore-method-"], capture_output=True, text=True)
    assert left.stdout.strip() == ""


@jailed
def test_unsupported_testbeds_refuse_instead_of_running_unjailed(tmp_path, in_jail):
    tb = get_testbed("control")
    with pytest.raises(jail.JailError, match="cannot run in the container jail"):
        run_method(tb, method_file(tmp_path, "HPARAMS=[{}]\ndef run(tasks, hparams, seed):\n    return []\n"),
                   type("I", (), {"public": [], "probe": None})(), {}, 0)
