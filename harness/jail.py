"""Container jail for method code.

SYNTHCORE_JAIL selects how methods run:
  process  (default) a separate Python process. Not a jail: the method can read any file
           the user can, including cached ground truth. Use only for code you wrote or read.
  docker | podman    a container with no network, a read-only root filesystem, no
           capabilities, an unprivileged user, and only the sandbox folder mounted.
           Use this for any method you have not reviewed.

The image is built from containers/method/Dockerfile and tagged by its hash, so two
runners with the same Synthcore commit run methods on the same image.
Limits: SYNTHCORE_JAIL_MEMORY (default 4g), SYNTHCORE_JAIL_CPUS (default 2).
A container shares the host kernel. On macOS, Docker Desktop adds a VM boundary.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCKERFILE = ROOT / "containers" / "method" / "Dockerfile"
MODES = ("process", "docker", "podman")
_built: set[str] = set()


class JailError(RuntimeError):
    """The jail itself failed or cannot run this testbed. Not a method failure."""


def mode() -> str:
    m = os.environ.get("SYNTHCORE_JAIL", "process").strip().lower()
    if m not in MODES:
        raise JailError(f"SYNTHCORE_JAIL must be one of {MODES}, not {m!r}")
    return m


def image() -> str:
    return "synthcore-method:" + hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()[:12]


def describe() -> dict:
    m = mode()
    return {"mode": m, "image": None if m == "process" else image()}


def ensure_image(engine: str) -> str:
    tag = image()
    if tag in _built:
        return tag
    found = subprocess.run([engine, "image", "inspect", tag], capture_output=True)
    if found.returncode != 0:
        print(f"[jail] building {tag} (once per Dockerfile version)", flush=True)
        built = subprocess.run([engine, "build", "-q", "-t", tag, str(DOCKERFILE.parent)],
                               capture_output=True, text=True)
        if built.returncode != 0:
            raise JailError(f"could not build {tag}: {built.stderr[-1500:]}")
    _built.add(tag)
    return tag


def command(engine: str, sbx: Path, method_path: Path, seed: int, interactive: bool) -> tuple[list[str], str]:
    """argv for one jailed method run, and the container name (for kill on timeout)."""
    tag = ensure_image(engine)
    name = f"synthcore-method-{uuid.uuid4().hex[:12]}"
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    if uid == 0:  # never run method code as root, even inside the container
        os.chown(sbx, 65534, 65534)
        user = "65534:65534"
    else:
        user = f"{uid}:{os.getgid()}"
    bind = lambda src, dst, ro: ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")]  # noqa: E731
    argv = [
        engine, "run", "--rm", "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "256",
        "--memory", os.environ.get("SYNTHCORE_JAIL_MEMORY", "4g"),
        "--cpus", os.environ.get("SYNTHCORE_JAIL_CPUS", "2"),
        "--user", user, "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m",
        "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
        *bind(sbx, "/sbx", False),
        *bind(Path(method_path).resolve(), "/jail/method.py", True),
        *bind(HERE / "sandbox_exec.py", "/jail/sandbox_exec.py", True),
        *bind(HERE / "safe_io.py", "/jail/safe_io.py", True),
        "-w", "/sbx",
    ]
    if interactive:
        argv.append("-i")
    argv += [tag, "python", "-I", "/jail/sandbox_exec.py", "/jail/method.py", "/sbx",
             "/sbx/hparams.json", str(seed)]
    if interactive:
        argv.append("rpc:stdio")
    return argv, name


def kill(engine: str, name: str) -> None:
    subprocess.run([engine, "kill", name], capture_output=True)
