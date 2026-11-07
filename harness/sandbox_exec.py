"""Runs one method in a separate process.

Invoked as: python -I sandbox_exec.py <method.py> <sandbox_dir> <hparams.json> <seed> [rpc:<rfd>,<wfd>|rpc:stdio]

With rpc, the testbed is interactive: public.py talks to the harness over two pipes, and
the environment runs in the harness process. rpc:stdio uses stdin and stdout (the only
channel into a container); the method's own prints are moved to stderr.

The sandbox dir holds only public inputs (inputs.pkl) and the testbed's public helper
module (public.py). Ground truth never leaves the parent process. Outputs go back as
JSON and .npy files (safe_io), never as a pickle, so the harness runs no method code.
Isolation depends on harness/jail.py: a separate process by default, or a container.
"""
import importlib.util
import json
import os
import pickle
import random
import sys
import traceback


def _safe_io():
    # Loaded by path: putting harness/ on sys.path would expose harness modules to the method.
    spec = importlib.util.spec_from_file_location(
        "synthcore_safe_io", os.path.join(os.path.dirname(os.path.abspath(__file__)), "safe_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    method_path, sandbox, hp_path, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, sandbox)
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(2)
    except ImportError:
        pass

    safe_io = _safe_io()
    if len(sys.argv) > 5 and sys.argv[5].startswith("rpc:"):
        if sys.argv[5] == "rpc:stdio":
            rfd, wfd = os.dup(0), os.dup(1)
            os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
            os.dup2(2, 1)  # stray prints must not corrupt the request stream
            sys.stdout = sys.stderr
        else:
            rfd, wfd = (int(x) for x in sys.argv[5][4:].split(","))
        import public
        public._connect(rfd, wfd)

    with open(f"{sandbox}/inputs.pkl", "rb") as f:
        tasks = pickle.load(f)
    with open(hp_path) as f:
        hparams = json.load(f)

    spec = importlib.util.spec_from_file_location("method_under_test", method_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        outputs = mod.run(tasks, hparams, seed)
        if not isinstance(outputs, list) or len(outputs) != len(tasks):
            raise ValueError(f"run() must return a list of {len(tasks)} outputs")
        safe_io.dump(outputs, os.path.join(sandbox, "out"))
    except Exception:
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
