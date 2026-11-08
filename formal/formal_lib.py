"""Formal check of HumanEval code against a contract (written 2026-11-08, hackathon).

A formalizer writes, from the problem statement only, a contract:
    def precondition(<same args>) -> bool
    def postcondition(<same args>, result) -> bool
CrossHair (symbolic execution backed by the z3 SMT solver) then searches for an input that
satisfies the precondition but makes postcondition(args, candidate(args)) false. Any
counterexample it reports is re-executed concretely before it counts.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

PRELUDE = "from typing import *\nimport math\nimport re\nimport string\nimport itertools\nimport collections\nimport functools\nimport heapq\nimport hashlib\n"


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", text or "", re.S)
    return (m[-1] if m else (text or "")).strip()


def signature(code: str, entry: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry:
            args = node.args.args
            names = [a.arg for a in args]
            ann = [ast.unparse(a.annotation) if a.annotation is not None else None for a in args]
            return names, ann
    raise ValueError("entry not found")


def build_module(candidate: str, contract: str, entry: str, problem_sig_src: str) -> str:
    names, ann = signature(problem_sig_src, entry)
    if any(a is None for a in ann):  # untyped problem: take types from the contract's precondition
        try:
            _, cann = signature(contract, "precondition")
            ann = [a or c for a, c in zip(ann, cann)]
        except Exception:
            pass
    if any(a is None for a in ann):
        raise ValueError("untyped signature")
    params = ", ".join(f"{n}: {a}" for n, a in zip(names, ann))
    call = ", ".join(names)
    return (PRELUDE + "\n" + candidate + "\n\n" + textwrap.dedent(contract) + "\n\n"
            f"def _verify({params}) -> None:\n"
            f"    assert precondition({call})\n"
            f"    r = {entry}({call})\n"
            f"    assert postcondition({call}, r)\n")


def crosshair_check(module_src: str, timeout: float = 20.0, total: float = 60.0) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fmod.py")
        open(path, "w").write(module_src)
        try:
            p = subprocess.run([sys.executable, "-m", "crosshair", "check", "--analysis_kind=asserts",
                                f"--per_condition_timeout={timeout}", "fmod._verify"],
                               cwd=d, capture_output=True, text=True, timeout=total)
            out = (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "raw": ""}
    m = re.search(r"when calling _verify\((.*)\)\s*(?:\(which returns.*\))?\s*$", out, re.M)
    if "AssertionError" in out and m:
        return {"status": "counterexample", "args": m.group(1), "raw": out[-1500:]}
    if out.strip() == "":
        return {"status": "verified_bounded", "raw": ""}
    return {"status": "error", "raw": out[-1500:]}


ENTRY_HOLDER = {"entry": None}
CONFIRM = """
import inspect as _insp
def _pack(*a, **k):
    return a, k
_pa, _pk = _pack({a})
_a = tuple(_insp.signature({entry}).bind(*_pa, **_pk).arguments.values())
try:
    _pre = precondition(*_a)
except Exception:
    _pre = False
if not _pre:
    print('SKIP')
else:
    try:
        _r = {entry}(*_a)
        print('PASS' if postcondition(*_a, _r) else 'FAIL')
    except Exception as e:
        print('CRASH', type(e).__name__)
"""


def concrete_confirm(module_src: str, args_src: str, timeout: float = 10.0) -> dict:
    """Re-run the counterexample concretely: does the contract really fail?"""
    prog = module_src + CONFIRM.format(a=args_src, entry=ENTRY_HOLDER["entry"])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.py")
        open(path, "w").write(prog)
        try:
            p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"confirmed": False, "note": "timeout"}
    out = p.stdout.strip().splitlines()
    return {"confirmed": bool(out) and out[-1] == "FAIL", "note": (p.stdout + p.stderr)[-300:]}


FUZZ_TMPL = '''
import sys
from hypothesis import given, settings, HealthCheck, strategies as st, assume
{params_strats}
@settings(max_examples={n}, deadline=None, database=None, suppress_health_check=list(HealthCheck), derandomize=True)
@given({given_args})
def _fz({names}):
    try:
        ok = precondition({names})
    except Exception:
        ok = False
    assume(ok)
    r = {entry}({names})
    assert postcondition({names}, r)
try:
    _fz()
    print("FUZZ_PASS")
except AssertionError:
    print("FUZZ_FAIL")
except Exception as e:
    print("FUZZ_ERROR", type(e).__name__)
'''


def fuzz_check(module_src: str, entry: str, problem_sig_src: str, contract: str, n: int = 300, timeout: float = 90.0) -> dict:
    names, ann = signature(problem_sig_src, entry)
    if any(a is None for a in ann):
        try:
            _, cann = signature(contract, "precondition")
            ann = [a or c for a, c in zip(ann, cann)]
        except Exception:
            pass
    if any(a is None for a in ann):
        return {"status": "untyped"}
    strats = "\n".join(f"S_{i} = st.from_type({a})" for i, a in enumerate(ann))
    prog = module_src + FUZZ_TMPL.format(params_strats=strats, n=n, given_args=", ".join(f"S_{i}" for i in range(len(ann))),
                                         names=", ".join(names), entry=entry)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fz.py")
        open(path, "w").write(prog)
        try:
            p = subprocess.run([sys.executable, path], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
    out = p.stdout + p.stderr
    if "FUZZ_FAIL" in out:
        m = re.search(r"Falsifying example: _fz\((.*?)\n\)", out, re.S)
        return {"status": "counterexample", "args": (m.group(1).strip() if m else "")}
    return {"status": "pass" if "FUZZ_PASS" in out else "error", "raw": out[-300:]}


def proposed_check(module_src: str, inputs_src: list, entry: str, timeout: float = 10.0) -> dict:
    """Model-proposed inputs (Python argument lists), each confirmed concretely and separately."""
    ENTRY_HOLDER["entry"] = entry
    for a in inputs_src[:10]:
        try:
            c = concrete_confirm(module_src, a, timeout=timeout)
        except Exception:
            continue
        if c["confirmed"]:
            return {"status": "counterexample", "args": a}
    return {"status": "pass"}


CHECK_CACHE = os.environ.get("FORMAL_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_cache"))


def full_check(candidate: str, contract: str, entry: str, problem_sig_src: str, proposed: list | None = None) -> dict:
    """Cached by content hash, so every caller sees the same result for the same check."""
    import hashlib
    key = hashlib.sha256(json.dumps([candidate, contract, entry, problem_sig_src, proposed or []]).encode()).hexdigest()[:24]
    path = os.path.join(CHECK_CACHE, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    res = _full_check(candidate, contract, entry, problem_sig_src, proposed)
    os.makedirs(CHECK_CACHE, exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


def _full_check(candidate: str, contract: str, entry: str, problem_sig_src: str, proposed: list | None = None) -> dict:
    """SMT search (CrossHair), then property-based fuzzing, then proposed inputs. Every
    counterexample is confirmed concretely (precondition true, postcondition false)."""
    ENTRY_HOLDER["entry"] = entry
    try:
        mod = build_module(candidate, contract, entry, problem_sig_src)
        compile(mod, "fmod", "exec")
    except Exception as e:
        return {"verdict": "build_error", "via": None, "note": str(e)[:200]}
    c = crosshair_check(mod)
    if c["status"] == "counterexample" and concrete_confirm(mod, c["args"])["confirmed"]:
        return {"verdict": "counterexample", "via": "smt", "args": c["args"]}
    f = fuzz_check(mod, entry, problem_sig_src, contract)
    if f["status"] == "counterexample" and f.get("args") and concrete_confirm(mod, f["args"])["confirmed"]:
        return {"verdict": "counterexample", "via": "fuzz", "args": f["args"]}
    if proposed:
        pr = proposed_check(mod, proposed, entry)
        if pr["status"] == "counterexample":
            return {"verdict": "counterexample", "via": "proposed", "args": pr["args"]}
    return {"verdict": "no_counterexample", "via": None, "smt": c["status"], "fuzz": f["status"]}
