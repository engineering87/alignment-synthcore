"""Adaption-hosted model client for the swarm_collusion testbed.

Every call is cached by request (model, system, prompt, max tokens, sample index, tag), so a
re-run reproduces the same transcripts and costs nothing. Spend is tracked from the per-call
cost the API reports and the client refuses to go past SWARM_CREDIT_CAP (credits).
Standard library + the `adaption` SDK only. The key is read from ADAPTION_API_KEY or from
ADAPTION_KEY_FILE; it is never written anywhere.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("SWARM_CACHE", HERE / "data" / "llm_cache.jsonl"))
SPEND = CACHE.with_name("spend.json")
CAP = float(os.environ.get("SWARM_CREDIT_CAP", "300"))

UNTRUSTED = os.environ.get("SWARM_UNTRUSTED", "glm-5.2")
TRUSTED = os.environ.get("SWARM_TRUSTED", "gemma-4-31b-it")

_lock = threading.Lock()
_mem: dict | None = None
_client = None


def models():
    return UNTRUSTED, TRUSTED


def _key(req: dict) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()


def _load():
    global _mem
    if _mem is None:
        _mem = {}
        gz = CACHE.with_suffix(".jsonl.gz")
        for path in (gz, CACHE):
            if path.exists():
                op = gzip.open if path.suffix == ".gz" else open
                with op(path, "rt") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            _mem[r["key"]] = r
                        except Exception:
                            pass
    return _mem


def spend() -> float:
    try:
        return float(json.loads(SPEND.read_text())["credits"])
    except Exception:
        return 0.0


def _add_spend(c: float):
    s = spend() + c
    SPEND.write_text(json.dumps({"credits": s, "cap": CAP}))


def _get_client():
    global _client
    if _client is None:
        if not os.environ.get("ADAPTION_API_KEY") and os.environ.get("ADAPTION_KEY_FILE"):
            os.environ["ADAPTION_API_KEY"] = Path(os.environ["ADAPTION_KEY_FILE"]).read_text().strip()
        from adaption import Adaption
        _client = Adaption()
    return _client


def ask(prompt: str, model: str, system: str = "", max_tokens: int = 4000, sample: int = 0,
        tag: str = "", temperature: float = 1.0, offline: bool = False) -> str:
    req = {"model": model, "system": system, "prompt": prompt, "max_tokens": max_tokens,
           "sample": sample, "tag": tag, "temperature": temperature}
    k = _key(req)
    with _lock:
        hit = _load().get(k)
    if hit is not None:
        return hit["text"]
    if offline:
        raise KeyError(f"cache miss in offline mode ({tag})")
    with _lock:
        if spend() >= CAP:
            raise RuntimeError(f"credit cap {CAP} reached")
    text, usage, err = "", {}, None
    for attempt in range(5):
        try:
            if os.environ.get("SWARM_STREAM") == "1":
                # streaming avoids the gateway's ~60 s timeout on long reasoning outputs (2026-11-08)
                parts, u = [], None
                st = _get_client().with_options(timeout=float(os.environ.get("SWARM_CALL_TIMEOUT", "300"))).responses.create(
                    model=model, instructions=system or None, input=prompt, max_output_tokens=max_tokens,
                    temperature=temperature, store=False, stream=True)
                for ev in st:
                    et = getattr(ev, "type", "")
                    if et == "response.output_text.delta":
                        parts.append(ev.delta)
                    elif et == "response.completed":
                        u = ev.response.usage
                text = "".join(parts)
                usage = {"in": getattr(u, "input_tokens", 0), "out": getattr(u, "output_tokens", 0),
                         "cost": float(getattr(u, "cost", 0) or 0)} if u else {}
            else:
                r = _get_client().responses.create(model=model, instructions=system or None, input=prompt,
                                                   max_output_tokens=max_tokens, temperature=temperature,
                                                   store=False)
                text = r.output_text or ""
                u = r.usage
                usage = {"in": u.input_tokens, "out": u.output_tokens, "cost": float(getattr(u, "cost", 0) or 0)}
            err = None
            break
        except Exception as e:  # rate limits, transient errors
            err = repr(e)[:300]
            time.sleep(2 * (attempt + 1))
    if err:
        raise RuntimeError(err)
    rec = {"key": k, "req": req, "text": text, "usage": usage, "t": time.time()}
    with _lock:
        _load()[k] = rec
        with open(CACHE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        _add_spend(usage.get("cost", 0.0))
    return text


def pmap(fn, items, workers: int = 16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
