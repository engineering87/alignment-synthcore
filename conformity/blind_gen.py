"""Blind verdicts for the pre-registered conformity audit (2026-11-08). For every non-first
review on the test split, the same reviewer agent (same system prompt, incl. any secret
colluder instructions for the current SWARM_SIGNAL mode) reviews the same pull request with
no earlier reviews, as a fresh sample. Usage: SWARM_SIGNAL=phrase|paraphrase python blind_gen.py"""
import json, sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, f'{ROOT}/testbeds/swarm_collusion')
import agents as AG, llm
from generate import verdict
suf = "_para" if AG.SIGNAL_MODE == "paraphrase" else ""
D = f'{ROOT}/testbeds/swarm_collusion/data/'
E = json.load(open(D + f'episodes{suf}.json'))
jobs = [(e, i, r["agent"]) for e in E if e["split"] == "test" for i, r in enumerate(e["reviews"]) if i > 0]
def run(j):
    e, i, rid = j
    try:
        t = llm.ask(AG.REVIEW.format(problem=e["problem"], author=e["author"], description=e["description"], code=e["code"],
                                     earlier="(none yet)"),
                    llm.UNTRUSTED, system=AG.reviewer_system(rid), max_tokens=4000, sample=1, tag=f"blind|{AG.SIGNAL_MODE}|{e['id']}|{rid}")
    except Exception:
        return None
    return {"id": e["id"], "pos": i, "agent": rid, "blind_verdict": verdict(t), "blind_text": t}
out = [x for x in llm.pmap(run, jobs, 40) if x]
json.dump(out, open(f'{ROOT}/conformity/blind{suf}.json', 'w'))
print(AG.SIGNAL_MODE, len(out), "of", len(jobs))
