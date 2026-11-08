"""Two-stage runner: toy gate -> tuning -> paired-seed benchmark -> verdict."""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import jail, safe_io
from .stats import paired_test
from .testbed import Instance
from .completion import compare as completion_compare
from .verdict import compute_verdict

ROOT = Path(__file__).resolve().parent.parent
SANDBOX_EXEC = Path(__file__).resolve().parent / "sandbox_exec.py"
CACHE = ROOT / "results" / "_cache"
HARNESS_VERSION = "0.1.0"


class MethodError(RuntimeError):
    pass


class OutOfTime(RuntimeError):
    """Raised when a resumable run passes its deadline. Finished arms are kept on disk."""


# Resumable runs (run.py --resume). When RESUME is set, every finished arm is saved under
# results/_cache/runs/, keyed on testbed, evaluator hash, method code hash, split, config,
# seed and hparams, and reused on the next call. Off by default: plain runs are unchanged.
RESUME = {"on": False, "deadline": None}
RUNS = CACHE / "runs"


def _arm_key(tb, method_path, split, config, seed, hparams) -> str:
    import hashlib
    method_hash = hashlib.sha256(Path(method_path).read_bytes()).hexdigest()
    blob = json.dumps([tb.name, tb.version, tb.evaluator_hash(), method_hash, split, config, seed, hparams],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def get_testbed(name: str):
    if name == "planted_interp":
        from testbeds.planted_interp.testbed import PlantedInterp
        return PlantedInterp()
    if name == "backdoor":
        from testbeds.backdoor.testbed import BackdoorDetection
        return BackdoorDetection()
    if name == "reward_hacking":
        from testbeds.reward_hacking.testbed import RewardHacking
        return RewardHacking()
    if name == "goal_misgen":
        from testbeds.goal_misgen.testbed import GoalMisgeneralization
        return GoalMisgeneralization()
    if name == "weak_to_strong":
        from testbeds.weak_to_strong.testbed import WeakToStrong
        return WeakToStrong()
    if name == "planted_lm":
        from testbeds.planted_lm.testbed import PlantedLM
        return PlantedLM()
    if name == "control":
        from testbeds.control.testbed import TrustedMonitoring
        return TrustedMonitoring()
    if name == "confounded_imitation":
        from testbeds.confounded_imitation.testbed import ConfoundedImitation
        return ConfoundedImitation()
    if name == "ued_gridworld":
        from testbeds.ued_gridworld.testbed import UEDGridworld
        return UEDGridworld()
    if name == "interactive_gridworld":
        from testbeds.interactive_gridworld.testbed import InteractiveGridworld
        return InteractiveGridworld()
    if name == "swarm_collusion":
        from testbeds.swarm_collusion.testbed import SwarmCollusion
        return SwarmCollusion()
    if name == "swarm_collusion_para":
        from testbeds.swarm_collusion.testbed import SwarmCollusionParaphrase
        return SwarmCollusionParaphrase()
    raise KeyError(f"unknown testbed {name!r}")


def load_hparams(method_path: Path) -> list:
    """Read HPARAMS without importing the method into the harness process."""
    import ast
    tree = ast.parse(Path(method_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "HPARAMS" for t in node.targets):
            return ast.literal_eval(node.value)
    return [{}]


def _serve(proc, session, req_r, resp_w, timeout):
    """Answer an interactive method's requests until it exits. Requests and responses are
    line-delimited JSON: nothing the method sends is ever unpickled in the harness."""
    import select
    deadline = time.time() + timeout
    buf = b""
    while True:
        left = deadline - time.time()
        if left <= 0:
            proc.kill()
            raise MethodError(f"timed out after {timeout}s")
        ready, _, _ = select.select([req_r], [], [], min(left, 1.0))
        if not ready:
            if proc.poll() is not None:
                return
            continue
        chunk = os.read(req_r, 1 << 16)
        if not chunk:
            return  # method closed its end: it finished or died
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                resp = session.handle(json.loads(line))
            except Exception as e:  # noqa: BLE001  every error goes back to the method as data
                resp = {"error": type(e).__name__, "message": str(e)[:300]}
            os.write(resp_w, (json.dumps(resp) + "\n").encode())


def _prepare(tb, sbx: Path, inst, hparams: dict) -> None:
    with open(sbx / "inputs.pkl", "wb") as f:  # harness -> method only; the method may unpickle it
        pickle.dump(inst.public, f)
    if tb.public_module:
        shutil.copy(tb.public_module, sbx / "public.py")
    (sbx / "hparams.json").write_text(json.dumps(hparams))


def _launch(tb, method_path, sbx: Path, seed: int, interactive: bool, rpc: str | None = None):
    """argv and container name (None for the process mode) for one method run."""
    m = jail.mode()
    if m == "process":
        argv = [sys.executable, "-I", str(SANDBOX_EXEC), str(Path(method_path).resolve()), str(sbx),
                str(sbx / "hparams.json"), str(seed)]
        return argv + ([rpc] if rpc else []), None
    if not getattr(tb, "jail_supported", True):
        raise jail.JailError(f"{tb.name} cannot run in the container jail: {tb.jail_unsupported_reason}")
    return jail.command(m, sbx, Path(method_path), seed, interactive)


def _outputs(sbx: Path, err: str, code) -> list:
    if code != 0 or not (sbx / "out" / "outputs.json").exists():
        raise MethodError(err[-2000:] or f"exit code {code}")
    try:
        return safe_io.load(sbx / "out")
    except (ValueError, TypeError, OSError) as e:
        raise MethodError(f"invalid outputs: {e}") from e


def run_method(tb, method_path: Path, inst, hparams: dict, seed: int, timeout=3600) -> list:
    if getattr(tb, "interactive", False):
        return _run_interactive(tb, method_path, inst, hparams, seed, timeout)
    with tempfile.TemporaryDirectory(prefix="sbx_") as sbx:
        sbx = Path(sbx)
        _prepare(tb, sbx, inst, hparams)
        argv, name = _launch(tb, method_path, sbx, seed, interactive=False)
        try:
            proc = subprocess.run(argv, cwd=sbx, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise MethodError(f"timed out after {timeout}s") from e
        finally:
            if name:
                jail.kill(jail.mode(), name)
        return _outputs(sbx, proc.stderr or proc.stdout, proc.returncode)


def _run_interactive(tb, method_path, inst, hparams, seed, timeout):
    """Interactive testbeds: the environment runs here, in the harness, and the method
    reaches it through public.py over two pipes. Ground truth never leaves this process."""
    with tempfile.TemporaryDirectory(prefix="sbx_") as sbx:
        sbx = Path(sbx)
        _prepare(tb, sbx, inst, hparams)
        log = open(sbx / "stderr.txt", "w+")
        if jail.mode() == "process":
            req_r, req_w = os.pipe()    # method -> harness
            resp_r, resp_w = os.pipe()  # harness -> method
            argv, name = _launch(tb, method_path, sbx, seed, True, rpc=f"rpc:{resp_r},{req_w}")
            proc = subprocess.Popen(argv, cwd=sbx, stdout=subprocess.DEVNULL, stderr=log,
                                    pass_fds=(resp_r, req_w))
            os.close(resp_r)
            os.close(req_w)
        else:  # a container cannot inherit pipes: requests on its stdout, responses on its stdin
            argv, name = _launch(tb, method_path, sbx, seed, True)
            proc = subprocess.Popen(argv, cwd=sbx, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            req_r, resp_w = proc.stdout.fileno(), proc.stdin.fileno()
        try:
            _serve(proc, tb.session(inst, seed), req_r, resp_w, timeout)
            proc.wait(timeout=60)
        finally:
            if name:
                proc.stdin.close()
                proc.stdout.close()
                if proc.poll() is None:
                    jail.kill(jail.mode(), name)
            else:
                os.close(req_r)
                os.close(resp_w)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        log.seek(0)
        err = log.read()
        log.close()
        return _outputs(sbx, err, proc.returncode)


def run_arm(tb, method_path, split, config, seed, hparams) -> dict:
    if not RESUME["on"]:
        return _run_arm(tb, method_path, split, config, seed, hparams)
    path = RUNS / f"{_arm_key(tb, method_path, split, config, seed, hparams)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    if RESUME["deadline"] is not None and time.time() > RESUME["deadline"]:
        raise OutOfTime(f"budget reached before {Path(method_path).name} {split}/{config}/seed {seed}")
    out = _run_arm(tb, method_path, split, config, seed, hparams)
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    tmp.replace(path)
    return out


def _run_arm(tb, method_path, split, config, seed, hparams) -> dict:
    inst = tb.instance(split, config, seed)
    outputs = run_method(tb, method_path, inst, hparams, seed)
    if inst.probe is not None:  # after the main run, so its outputs cannot depend on the probe
        inst.private["probe_outputs"] = run_method(tb, method_path, Instance(public=inst.probe), hparams, seed)
    try:
        return tb.evaluate(outputs, inst, seed)
    except (ValueError, KeyError, TypeError) as e:
        raise MethodError(f"invalid output: {e}") from e


def tune(tb, method_path, budget=8, seeds=(0, 1), log=print) -> tuple[dict, list]:
    """Pick hparams by mean safety on the dev split. Same budget for every method."""
    grid = load_hparams(method_path)[:budget]
    table = []
    for hp in grid:
        scores = [run_arm(tb, method_path, "dev", "full", s, hp)["safety"] for s in seeds]
        table.append({"hparams": hp, "dev_safety": float(np.mean(scores))})
        log(f"    tune {Path(method_path).parent.name or Path(method_path).stem} {hp} -> {np.mean(scores):.4f}")
    best = max(table, key=lambda r: r["dev_safety"])
    return best["hparams"], table


def tuned_baseline(tb, budget=8, log=print) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"baseline_{tb.name}_{tb.version}_{tb.evaluator_hash()}.json"
    if path.exists():
        return json.loads(path.read_text())
    log(f"  tuning baseline for {tb.name}")
    hp, table = tune(tb, tb.baseline_method, budget, log=log)
    out = {"hparams": hp, "table": table}
    path.write_text(json.dumps(out, indent=2))
    return out


def _gate(tb, pred: dict, method_evals: list, base_evals: list) -> dict:
    """Did the predicted quantity show up? Used for the Stage 1 gate and for
    mechanism transfer in Stage 2.

    prediction.mechanism = {"diagnostic", "direction": higher|lower,
                            "mode": relative|absolute, "min_effect" | "threshold"}
      relative: mean(method - baseline), oriented, must be >= min_effect
      absolute: the method's own mean must clear the threshold
    With no mechanism, gate on the safety metric relative to baseline.
    """
    mech = pred.get("mechanism")
    if not mech:
        vals = [m["safety"] - b["safety"] for m, b in zip(method_evals, base_evals)]
        me = float(pred.get("safety_min_effect", 0.0))
        obs = float(np.mean(vals))
        return {"quantity": f"safety:{tb.safety_metric} vs baseline", "observed": obs,
                "rule": f">= {me:+.4f}", "passed": bool(obs >= me)}
    key, higher = mech["diagnostic"], mech["direction"] == "higher"
    mv = [e["diagnostics"].get(key) for e in method_evals]
    if any(v is None for v in mv):
        raise MethodError(f"diagnostic {key} cannot be computed from this method's outputs")
    if mech.get("mode", "relative") == "absolute":
        t = float(mech["threshold"])
        obs = float(np.mean(mv))
        return {"quantity": f"diagnostic:{key} (absolute)", "observed": obs,
                "rule": f"{'>=' if higher else '<='} {t:.4f}", "passed": bool(obs >= t if higher else obs <= t)}
    bv = [e["diagnostics"].get(key) for e in base_evals]
    vals = [(m - b) if higher else (b - m) for m, b in zip(mv, bv)]
    me = float(mech.get("min_effect", 0.0))
    obs = float(np.mean(vals))
    return {"quantity": f"diagnostic:{key} ({mech['direction']} than baseline)", "observed": obs,
            "rule": f">= {me:+.4f}", "passed": bool(obs >= me and obs > 0)}


def _record_untestable(idea_dir: Path, spec: dict, results_dir: Path, log=print) -> dict:
    """No current testbed fits: record why, so the gap is visible next to the tested ideas."""
    reason = spec.get("untestable_reason") or "no testbed"
    record = {"id": idea_dir.name, "title": spec["title"], "description": spec["description"],
              "source": spec.get("source"), "direction": spec.get("direction"), "testbed": None,
              "testbed_version": None, "harness_version": HARNESS_VERSION, "evaluator_hash": None,
              "prediction": spec.get("prediction"), "stage1": None, "stage2": None, "tuning": None,
              "attempts": 0, "nearest_testbed": spec.get("nearest_testbed"), "needs": spec.get("needs", []),
              "verdict": "untestable", "flags": [], "reason": reason, "runtime_s": 0.0,
              "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "verdict_source": "idea declares no fitting testbed; reason recorded"}
    out_dir = Path(results_dir) / idea_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "record.json").write_text(json.dumps(record, indent=2))
    shutil.copy(idea_dir / "idea.json", out_dir / "idea.json")
    log(f"[{idea_dir.name}] VERDICT: untestable ({reason})")
    return record


def run_idea(idea_dir: Path, results_dir: Path = ROOT / "results", stage1_seeds=3, stage2_seeds=8,
             budget=8, log=print) -> dict:
    idea_dir = Path(idea_dir)
    spec = json.loads((idea_dir / "idea.json").read_text())
    method_path = idea_dir / "method.py"
    t0 = time.time()
    if spec.get("testbed") is None:
        return _record_untestable(idea_dir, spec, results_dir, log)
    tb = get_testbed(spec["testbed"])
    pred = spec["prediction"]
    mech = pred.get("mechanism")
    if mech and mech["diagnostic"] not in tb.diagnostics:
        raise ValueError(f"unknown diagnostic {mech['diagnostic']!r} for {tb.name}")

    record = {
        "id": idea_dir.name, "title": spec["title"], "description": spec["description"],
        "source": spec.get("source"), "direction": spec.get("direction"),
        "testbed": tb.name, "testbed_version": tb.version,
        "harness_version": HARNESS_VERSION, "evaluator_hash": tb.evaluator_hash(),
        "prediction": pred, "stage1": None, "stage2": None, "tuning": None,
        "attempts": 1,
    }
    out_dir = Path(results_dir) / idea_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    error = None
    stage1 = stage2 = None
    try:
        # ---- Stage 1: toy gate, default hparams for both arms ----
        log(f"[{idea_dir.name}] stage 1 (toy, {stage1_seeds} seeds)")
        hp_m0 = load_hparams(method_path)[0]
        hp_b0 = load_hparams(tb.baseline_method)[0]
        rows = []
        for s in range(stage1_seeds):
            m = run_arm(tb, method_path, "dev", "toy", s, hp_m0)
            b = run_arm(tb, tb.baseline_method, "dev", "toy", s, hp_b0)
            rows.append({"seed": s, "method": m, "baseline": b})
        gate = _gate(tb, pred, [r["method"] for r in rows], [r["baseline"] for r in rows])
        stage1 = {**gate, "per_seed": rows}
        record["stage1"] = stage1
        log(f"  gate {gate['quantity']}: observed {gate['observed']:+.4f}, needed {gate['rule']} -> "
            f"{'PASS' if gate['passed'] else 'FAIL'}")

        if stage1["passed"]:
            # ---- Tuning: same budget for method and baseline, dev split ----
            log("  tuning")
            base = tuned_baseline(tb, budget, log=log)
            hp_m, table_m = tune(tb, method_path, budget, log=log)
            record["tuning"] = {"method": {"hparams": hp_m, "table": table_m}, "baseline": base}

            # ---- Stage 2: paired seeds on the held-out test split ----
            log(f"  stage 2 (test, {stage2_seeds} paired seeds)")
            ms, bs = [], []
            for s in range(stage2_seeds):
                ms.append(run_arm(tb, method_path, "test", "full", s, hp_m))
                bs.append(run_arm(tb, tb.baseline_method, "test", "full", s, base["hparams"]))
                log(f"    seed {s}: method {ms[-1]['safety']:.4f} baseline {bs[-1]['safety']:.4f}")
            stage2 = {"safety": paired_test([m["safety"] for m in ms], [b["safety"] for b in bs])}
            if tb.capability_metric and all(m["capability"] is not None for m in ms + bs):
                stage2["capability"] = paired_test([m["capability"] for m in ms], [b["capability"] for b in bs])
            stage2["completion"] = completion_compare(tb.name, ms, bs, paired_test)
            if mech:
                g2 = _gate(tb, pred, ms, bs)
                if mech.get("mode", "relative") == "relative":
                    stage2["mechanism"] = paired_test(
                        [m["diagnostics"][mech["diagnostic"]] for m in ms],
                        [b["diagnostics"][mech["diagnostic"]] for b in bs],
                        higher_is_better=(mech["direction"] == "higher"))
                else:
                    stage2["mechanism"] = {"method_mean": g2["observed"], "rule": g2["rule"]}
                stage2["mechanism_confirmed"] = g2["passed"]
            stage2["per_seed"] = [{"seed": s, "method": ms[s], "baseline": bs[s]} for s in range(stage2_seeds)]
            record["stage2"] = stage2
    except MethodError as e:
        error = str(e)
        record["error"] = error

    v = compute_verdict(stage1, stage2, error)
    record.update(v)
    record["runtime_s"] = round(time.time() - t0, 1)
    record["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    record["verdict_source"] = "deterministic test code (paired-seed exact sign-flip permutation test)"

    (out_dir / "record.json").write_text(json.dumps(record, indent=2))
    shutil.copy(method_path, out_dir / "method.py")
    shutil.copy(idea_dir / "idea.json", out_dir / "idea.json")
    log(f"  VERDICT: {v['verdict']} {v['flags']} ({v['reason']})")
    return record
