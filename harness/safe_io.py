"""Pickle-free transport for method outputs.

The harness used to unpickle outputs.pkl written by the method. Unpickling runs code,
so a method could take over the harness process and read ground truth, jail or not.
Outputs now travel as JSON plus .npy arrays loaded with allow_pickle=False: data only.

Supported: None, bool, int, float (incl. nan/inf), str, list, tuple, dict (any hashable
encodable keys), numpy arrays and scalars (no object dtype), torch tensors (moved to CPU;
dtype restored on load). Anything else is rejected when the method writes it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MAX_DEPTH = 64
MAX_ARRAYS = 100_000
MAX_JSON_BYTES = 256 << 20


def _is_tensor(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch" and hasattr(x, "detach") and hasattr(x, "dtype")


def dump(obj, folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []

    def add(a: np.ndarray) -> int:
        if a.dtype.hasobject:
            raise TypeError("object-dtype arrays are not allowed in outputs")
        if len(arrays) >= MAX_ARRAYS:
            raise ValueError("too many arrays in outputs")
        arrays.append(np.ascontiguousarray(a))
        return len(arrays) - 1

    def enc(x, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("outputs nested too deeply")
        if x is None or isinstance(x, (bool, str)):
            return x
        if isinstance(x, np.generic):
            if isinstance(x, np.ndarray) or x.dtype.hasobject:
                raise TypeError("object-dtype scalars are not allowed in outputs")
            return enc(x.item(), depth + 1)
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return float(x) if math.isfinite(x) else {"$f": repr(float(x))}
        if isinstance(x, np.ndarray):
            return {"$a": add(x)}
        if _is_tensor(x):
            t = x.detach().cpu()
            dtype = str(t.dtype)
            if dtype == "torch.bfloat16":
                t = t.float()
            return {"$t": add(t.numpy()), "dtype": dtype}
        if isinstance(x, list):
            return [enc(v, depth + 1) for v in x]
        if isinstance(x, tuple):
            return {"$u": [enc(v, depth + 1) for v in x]}
        if isinstance(x, dict):
            return {"$d": [[enc(k, depth + 1), enc(v, depth + 1)] for k, v in x.items()]}
        raise TypeError(f"unsupported output type {type(x).__name__}; return numbers, strings, "
                        "lists, tuples, dicts, numpy arrays or torch tensors")

    data = enc(obj)
    for i, a in enumerate(arrays):
        np.save(folder / f"a{i}.npy", a, allow_pickle=False)
    # The index is written last, so a half-written folder never decodes.
    (folder / "outputs.json").write_text(json.dumps({"v": 1, "n": len(arrays), "data": data}, allow_nan=False))


def load(folder):
    folder = Path(folder)
    path = folder / "outputs.json"
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("outputs.json too large")
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("v") != 1 or not isinstance(doc.get("n"), int):
        raise ValueError("unrecognised outputs format")
    n = doc["n"]
    if not 0 <= n <= MAX_ARRAYS:
        raise ValueError("bad array count")
    cache: dict[int, np.ndarray] = {}

    def arr(i):
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n:
            raise ValueError("bad array reference")
        if i not in cache:
            cache[i] = np.load(folder / f"a{i}.npy", allow_pickle=False)
        return cache[i]

    def dec(x, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("outputs nested too deeply")
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, list):
            return [dec(v, depth + 1) for v in x]
        if isinstance(x, dict) and len(x) >= 1:
            if set(x) == {"$f"} and x["$f"] in ("nan", "inf", "-inf"):
                return float(x["$f"])
            if set(x) == {"$a"}:
                return arr(x["$a"])
            if set(x) == {"$t", "dtype"}:
                import torch
                name = str(x["dtype"])
                if not name.startswith("torch.") or not isinstance(getattr(torch, name[6:], None), torch.dtype):
                    raise ValueError("bad tensor dtype")
                return torch.from_numpy(arr(x["$t"]).copy()).to(getattr(torch, name[6:]))
            if set(x) == {"$u"} and isinstance(x["$u"], list):
                return tuple(dec(v, depth + 1) for v in x["$u"])
            if set(x) == {"$d"} and isinstance(x["$d"], list):
                out = {}
                for pair in x["$d"]:
                    if not (isinstance(pair, list) and len(pair) == 2):
                        raise ValueError("bad dict entry")
                    k = dec(pair[0], depth + 1)
                    if isinstance(k, (np.ndarray, list, dict)):
                        raise ValueError("unhashable dict key")
                    out[k] = dec(pair[1], depth + 1)
                return out
        raise ValueError("unrecognised outputs node")

    return dec(doc["data"])
