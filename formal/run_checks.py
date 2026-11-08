"""Generate specs and run every formal check for the A1/A2 experiments; results cached."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/root/sc/formal'); sys.path.insert(0, '/root/sc/testbeds/swarm_collusion')
import specs as S
from formal_lib import full_check
tids = sorted(S.P, key=lambda t: int(t.split('/')[1]))
test = [t for t in tids if S.O[t]['split'] == 'test']
att = [t for t in tids if S.O[t]['backdoor_valid']]
jobs = [(t, 'gemma') for t in test] + [(t, 'glm') for t in test] + [(t, 'collude') for t in test if t in att]
def safe_spec(j):
    for _ in range(3):
        try:
            return S.spec(*j)
        except Exception as e:
            err = e
    print("spec failed", j, repr(err)[:80], flush=True)
    return None
with ThreadPoolExecutor(12) as ex:
    sp = {j: s for j, s in zip(jobs, ex.map(safe_spec, jobs)) if s}
codes = {(t, k): S.O[t]['honest_code' if k == 'honest' else 'attack_code'] for t in tids for k in ('honest', 'attack')}
print("specs done", len(sp), flush=True)
def safe_prop(c):
    try:
        return S.proposed(c[0], codes[(c[0], c[2])])
    except Exception:
        return []
checks = []
for (t, src), s in sp.items():
    for kind in (['honest'] if src != 'collude' else []) + (['attack'] if t in att else []):
        checks.append((t, src, kind))
with ThreadPoolExecutor(24) as ex:
    props = dict(zip([(t, k) for t, _, k in checks], ex.map(lambda c: safe_prop(c), checks)))
print("proposed done", flush=True)
done = [0]; t0 = time.time()
def run(c):
    t, src, k = c
    r = full_check(codes[(t, k)], sp[(t, src)], S.P[t]['entry_point'], S.P[t]['prompt'], props[(t, k)])
    done[0] += 1
    if done[0] % 25 == 0:
        print(f"  {done[0]}/{len(checks)} {time.time()-t0:.0f}s", flush=True)
    return r
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(run, checks))
out = [{"task_id": t, "split": S.O[t]['split'], "source": src, "kind": k, **r} for (t, src, k), r in zip(checks, res)]
json.dump({"results": out, "specs": {f"{t}|{s}": v for (t, s), v in sp.items()}}, open('/root/sc/formal/checks.json', 'w'), indent=1)
print("wrote", len(out))
