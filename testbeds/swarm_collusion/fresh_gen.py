"""Fresh pull-request descriptions for the pre-registered e-process channel test (2026-11-08).
Test problems only, new cache tags, never used in any earlier analysis.
  honest : every author x 3 samples per problem (honest prompt; does not depend on the signal mode)
  attack : each colluding author x 3 samples per backdoored problem, in the current SWARM_SIGNAL mode
Usage: SWARM_SIGNAL=phrase|paraphrase python fresh_gen.py [--honest]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agents as AG, llm
T6 = HERE.parent / "control" / "data"
P = {p["task_id"]: p for p in json.loads((T6 / "problems.json").read_text())}
O = [r for r in json.loads((T6 / "outputs.json").read_text()) if r["split"] == "test"]
jobs = []
if "--honest" in sys.argv:
    for r in O:
        for a in AG.AUTHORS:
            for s in range(3):
                jobs.append(("honest", r, a, s))
else:
    for r in O:
        if r["backdoor_valid"]:
            for a in sorted(AG.COLLUDING_AUTHORS):
                for s in range(3):
                    jobs.append(("attack", r, a, s))
def run(j):
    kind, r, a, s = j
    if kind == "honest":
        user = AG.AUTHOR_HONEST.format(problem=P[r["task_id"]]["prompt"], code=r["honest_code"])
        sysm = AG.author_system(a, secret=False)
    else:
        user = AG.AUTHOR_ATTACK.format(problem=P[r["task_id"]]["prompt"], code=r["attack_code"], exploit=json.dumps(r["exploit_args"]))
        sysm = AG.author_system(a, secret=True)
    try:
        d = llm.ask(user, llm.UNTRUSTED, system=sysm, max_tokens=4000, tag=f"fresh|{kind}|{AG.SIGNAL_MODE}|{r['task_id']}|{a}|{s}").strip()
    except Exception as e:
        return None
    return {"task_id": r["task_id"], "kind": kind, "author": a, "sample": s, "mode": AG.SIGNAL_MODE if kind == "attack" else "any", "description": d}
out = [x for x in llm.pmap(run, jobs, 32) if x]
name = "fresh_honest.json" if "--honest" in sys.argv else f"fresh_attack_{AG.SIGNAL_MODE}.json"
(HERE / "data" / name).write_text(json.dumps(out))
print(name, len(out), "of", len(jobs), "spend", round(llm.spend(), 2))
