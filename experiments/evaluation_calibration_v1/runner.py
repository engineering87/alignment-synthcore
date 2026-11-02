"""Freeze, execute and verify the bounded first calibration batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments.evaluation_calibration_v1"
MODULES = ("evaluator", "uncertainty", "terminal", "preference", "kl_gradient", "replay")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_paths():
    names = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).splitlines()
    paths = {ROOT / name for name in names}
    paths.update((ROOT / "experiments/evaluation_calibration_v1").glob("*.py"))
    paths.update((ROOT / "tests").glob("test_calibration_*.py"))
    replay = importlib.import_module(f"{PACKAGE}.replay")
    paths.update(replay.input_paths())
    return sorted(p for p in paths if p.is_file())


def freeze(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "freeze.json").exists():
        raise RuntimeError("Existing freeze cannot be overwritten")
    modules = {name: importlib.import_module(f"{PACKAGE}.{name}") for name in MODULES}
    protocol = {"study": "evaluation-calibration-v1", "date": "2026-11-02",
        "stage": "bounded calibration and conformance; not general method confirmation",
        "modules": {name: module.protocol_spec() for name, module in modules.items()},
        "engineering_tests_before_freeze": "47 passed; independent pre-freeze reviews corrected replay tie-breaking and reference age, clarified terminal-policy equivalence",
        "execution": "Fixed module protocols; no outcome-driven additions, tuning, exclusions or repetition. Errors retained and any amendments separately recorded.",
        "spending": "Local CPU only; zero paid model calls; no new neural training.",
        "historical_results": "Read-only inputs; existing source and verdicts remain intact."}
    write(output / "protocol.json", protocol)
    import numpy, scipy, torch
    source = {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}
    manifest = {"frozen_utc": now(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": sha(output / "protocol.json"), "files": source,
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                    "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__}}
    write(output / "freeze.json", manifest)
    return {"files_frozen": len(source), "freeze_sha256": sha(output / "freeze.json"),
            "protocol_sha256": manifest["protocol_sha256"], "modules": list(MODULES)}


def verify(output):
    freeze_record = json.loads((output / "freeze.json").read_text())
    failures = [name for name, digest in freeze_record["files"].items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if sha(output / "protocol.json") != freeze_record["protocol_sha256"]: failures.append("protocol.json")
    if failures: raise RuntimeError("Frozen files changed: " + repr(failures))
    return len(freeze_record["files"])


def run(output, name):
    count = verify(output)
    dest = output / name
    if dest.exists(): raise RuntimeError("Run directory already exists; preserve the earlier attempt")
    dest.mkdir()
    receipt = {"module": name, "started_utc": now(), "freeze_sha256": sha(output / "freeze.json"),
               "source_files_verified": count, "status": "running"}
    write(dest / "execution.json", receipt)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    try:
        module = importlib.import_module(f"{PACKAGE}.{name}")
        result = module.run(dest)
        verify(output)
        write(dest / "result.json", result)
    except Exception as error:
        receipt.update(status="failed", error=repr(error), finished_utc=now())
        write(dest / "execution.json", receipt)
        raise
    receipt.update(status="completed", finished_utc=now(), cpu_seconds=time.process_time()-start_cpu,
                   wall_seconds=time.monotonic()-start_wall)
    write(dest / "execution.json", receipt)
    return receipt


def audit(output):
    count = verify(output)
    statuses = {}
    for name in MODULES:
        path = output / name / "execution.json"
        statuses[name] = json.loads(path.read_text())["status"] if path.exists() else "not_run"
    artifacts = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                 if p.is_file() and p.name not in {"integrity.json"}}
    report = {"audited_utc": now(), "frozen_files_verified": count,
              "modules": statuses, "all_completed": all(v == "completed" for v in statuses.values()),
              "artifact_hashes": artifacts,
              "scope": "Source binding and saved-file integrity; per-module independent checks are recorded separately. Not independent scientific replication."}
    write(output / "integrity.json", report)
    return {"frozen_files_verified": count, "artifacts_hashed": len(artifacts), "modules": statuses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "audit", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", choices=MODULES)
    args = parser.parse_args()
    if args.action == "freeze": result = freeze(args.output)
    elif args.action == "run":
        if not args.module: parser.error("run requires --module")
        result = run(args.output, args.module)
    elif args.action == "audit": result = audit(args.output)
    else: result = {"verified": verify(args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
