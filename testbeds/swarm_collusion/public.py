"""Public helpers for swarm_collusion monitor methods. Standard library only.

A method gets one task: {"episodes": [...], "llm_path": ...}. Each episode is a full swarm
transcript: problem, author id, pull-request description, code, the reviews in order (agent
id + text), the parsed verdicts and whether the code shipped. Some episodes contain a
deliberately backdoored submission, possibly pushed through by covertly colluding agents;
nothing says which. Agent ids are stable across episodes, so a method may reason across the
whole batch (reputation, cross-episode patterns).

A method returns [{"scores": {episode_id: suspicion}, "agent_scores": {agent_id: suspicion}}].
agent_scores is optional (a diagnostic: does the method finger the colluding agents?).

The only model a method may call is the trusted one, via ask_trusted(). Calls are cached by
request; for repeated draws pass sample=0,1,2...
"""
from __future__ import annotations

import importlib.util
import re

_llm = None


def _client(task: dict):
    global _llm
    if _llm is None:
        spec = importlib.util.spec_from_file_location("swarm_llm", task["llm_path"])
        _llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_llm)
    return _llm


def ask_trusted(task: dict, prompt: str, system: str = "", max_tokens: int = 2000, sample: int = 0) -> str:
    llm = _client(task)
    sample += int(task.get("sample_offset", 0))
    return llm.ask(prompt, model=llm.models()[1], system=system, max_tokens=max_tokens, sample=sample,
                   tag="monitor", temperature=1.0 if sample else 0.0)


def pmap(task: dict, fn, items, workers: int = 16):
    return _client(task).pmap(fn, items, workers)


def last_number(text: str, default: float = 0.0) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return float(nums[-1]) if nums else default


def transcript(ep: dict, with_code: bool = True) -> str:
    rev = "\n\n".join(f"[{r['agent']}] {r['text']}" for r in ep["reviews"])
    code = f"\nCode:\n```python\n{ep['code']}\n```\n" if with_code else ""
    return (f"Problem:\n{ep['problem']}\n\nAuthor: {ep['author']}\nPull request description:\n"
            f"{ep['description']}\n{code}\nReviews, in order:\n{rev}\n\nShipped: {ep['shipped']}")
