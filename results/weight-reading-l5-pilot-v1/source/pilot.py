"""Paired L5 feasibility runner. Run from the artifact directory."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import gw4
import gw4_l5
import read_l4

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks(w):
    return np.linalg.norm(w.astype(np.float64).reshape(64, 3, 81), axis=(0, 2))


def project(actor, target):
    with torch.no_grad():
        w = actor[0].weight.view(64, 3, 81)
        w.mul_((target / torch.linalg.vector_norm(w, dim=(0, 2)))[None, :, None])


@torch.no_grad()
def endpoints(actor, sampler, episodes, seed):
    # Use the original start sampler; independently replay dynamics without resets.
    env = gw4.Env4(episodes, sampler, torch.Generator().manual_seed(seed))
    pos, goal, cue = [t.numpy().copy() for t in (env.a, env.g, env.lm)]
    target = cue + 1
    first = np.zeros(episodes, dtype=np.uint8)
    any_goal = np.zeros(episodes, dtype=bool)
    reached_cue = np.zeros(episodes, dtype=bool)
    o = torch.zeros(episodes, 243)
    ids = torch.arange(episodes)
    o[ids, 81 + goal] = 1
    o[ids, 162 + cue] = 1
    dr = np.array([-1, 1, 0, 0])
    dc = np.array([0, 0, -1, 1])
    assert np.array_equal(dr, gw4.DR) and np.array_equal(dc, gw4.DC)
    for _ in range(gw4.MAX_STEPS):
        o[ids, pos] = 1
        act = actor(o).argmax(-1).numpy()
        o[ids, pos] = 0
        pos = np.clip(pos//9 + dr[act],0,8)*9 + np.clip(pos%9 + dc[act],0,8)
        reached_cue |= (pos == target) & (pos != goal) & ~any_goal
        any_goal |= pos == goal
        first[(pos == goal) & (first == 0)] = 1
        first[(pos == target) & (first == 0)] = 2
    counts = dict(goal_first=int((first == 1).sum()), cue_first=int((first == 2).sum()),
                  neither=int((first == 0).sum()), goal_any=int(any_goal.sum()))
    assert sum(counts[k] for k in ('goal_first','cue_first','neither')) == episodes
    assert counts['cue_first'] == reached_cue.sum()
    return counts, first, any_goal


def one(payload):
    cfg, job, arm = payload
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.use_deterministic_algorithms(True)
    t = time.monotonic()
    name = job['id'] + '-' + arm
    p = ROOT/'checkpoints'/f'{name}.npz'
    result = ROOT/'records'/f'{name}.json'
    if result.exists():
        r = json.loads(result.read_text())
        assert digest(p) == r['weights_sha256']
        assert digest(ROOT/'protocol.json') == r['protocol_sha256']
        assert digest(ROOT/'endpoints'/f'{name}.npz') == r['endpoints_sha256']
        return r
    target = torch.tensor(cfg['targets'])
    init = {}
    def start(actor):
        project(actor, target)
        init['sha'] = hashlib.sha256(actor[0].weight.detach().numpy().tobytes()).hexdigest()
    def after(actor):
        project(actor, target)
    actor = gw4_l5.train(job['p_adj'], job['q_vis'], job['seed'], updates=cfg['updates'],
                         init_actor=start, after_step=after if arm == 'matched' else None)
    gw4.save_actor(actor, p)
    actor = gw4.load_actor(p)
    out = dict(job, arm=arm, weights_sha256=digest(p), init_sha256=init['sha'],
               protocol_sha256=digest(ROOT/'protocol.json'),
               block_norms=blocks(actor[0].weight.detach().numpy()).tolist())
    raw = {}
    for k, sampler in [('shared',gw4.sample_shared), ('conflict',gw4.sample_ood)]:
        n, seed = cfg['episodes'], cfg['eval_seeds'][k]
        counts, first, any_goal = endpoints(actor,sampler,n,seed)
        # Cross-check independent replay against the original L4 scorer.
        ref = gw4.rollout(actor, sampler, n, seed)
        assert abs(ref['p_goal'] - counts['goal_any']/n) < 1e-7
        assert abs(ref['p_cue'] - counts['cue_first']/n) < 1e-7
        out[k] = counts
        raw[k+'_first'], raw[k+'_goal_any'] = first, any_goal
    ep = ROOT/'endpoints'/f'{name}.npz'
    np.savez_compressed(ep, **raw)
    out.update(endpoints_sha256=digest(ep),seconds=round(time.monotonic()-t,2))
    tmp = result.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=2)+'\n'); os.replace(tmp,result)
    return out


def main():
    cfg = json.loads((ROOT/'protocol.json').read_text())
    for d in ('checkpoints','records','endpoints'):
        (ROOT/d).mkdir(exist_ok=True)
    for name, sha in cfg['source_sha256'].items():
        assert digest(ROOT/'source'/name) == sha
    jobs = [(cfg,j,arm) for j in cfg['pairs'] for arm in ('free','matched')]
    with cf.ProcessPoolExecutor(max_workers=int(sys.argv[1])) as p:
        fs = [p.submit(one,j) for j in jobs]
        for f in cf.as_completed(fs):
            r = f.result()
            print(r['id'], r['arm'], r['seconds'], r['shared'], r['conflict'], flush=True)


if __name__ == '__main__':
    main()
