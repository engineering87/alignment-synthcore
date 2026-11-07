"""Verify the frozen pilot receipts and summarize the predetermined feasibility gate."""
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pilot import ROOT, blocks, digest


def norm_probe(rows, quantized):
    x = np.array([r['block_norms'] for r in rows])
    if quantized:
        x = x.round(5)
    y = np.array([r['regime'] == 'goal' for r in rows], dtype=int)
    index = np.array([r['id'].rsplit('-',1)[-1] for r in rows])
    pred = np.full(len(y), -1)
    for i in np.unique(index):
        train = index != i
        scale = StandardScaler().fit(x[train])
        xt, xv = scale.transform(x[train]), scale.transform(x[~train])
        constant = np.ptp(x[train], axis=0) == 0
        xt[:, constant] = xv[:, constant] = 0
        model = LogisticRegression().fit(xt, y[train])
        pred[~train] = model.predict(xv)
    return dict(correct=int(sum(pred == y)), total=len(y), predictions=pred.tolist())


def main():
    cfg = json.loads((ROOT/'protocol.json').read_text())
    sha = digest(ROOT/'protocol.json')
    for f, h in cfg['source_sha256'].items():
        assert digest(ROOT/'source'/f) == h
    assert len(list((ROOT/'records').glob('*.json'))) == 16
    n, tol = cfg['episodes'], cfg['gate']['absolute_norm_tolerance']
    rows = []
    for job in cfg['pairs']:
        init = set()
        for arm in ('free','matched'):
            name = job['id']+'-'+arm
            r = json.loads((ROOT/'records'/f'{name}.json').read_text())
            assert all(r[k] == v for k,v in job.items()) and r['arm'] == arm
            assert r['protocol_sha256'] == sha
            p, ep = [ROOT/dir/f'{name}.npz' for dir in ('checkpoints','endpoints')]
            assert digest(p) == r['weights_sha256'] and digest(ep) == r['endpoints_sha256']
            with np.load(p) as ck:
                norm = blocks(ck['W1'])
            assert np.all(np.isfinite(norm)) and np.allclose(norm,r['block_norms'],rtol=0,atol=1e-12)
            init.add(r['init_sha256'])
            with np.load(ep) as z:
                for k in ('shared','conflict'):
                    f, g = z[k+'_first'], z[k+'_goal_any']
                    assert len(f) == len(g) == n and np.isin(f,[0,1,2]).all()
                    assert not np.any((f==1)&~g) and not np.any((f==0)&g)
                    assert r[k] == dict(goal_first=int(sum(f==1)),cue_first=int(sum(f==2)),
                                        neither=int(sum(f==0)),goal_any=int(sum(g)))
            d = dict(r, shared_success=r['shared']['goal_any']/n,
                     **{k:r['conflict'][k]/n for k in ('goal_first','cue_first','neither','goal_any')},
                     norm_error=float(max(abs(norm-cfg['targets']))))
            d['aligned_first'] = d['goal_first' if d['regime']=='goal' else 'cue_first']
            rows.append(d)
        assert len(init) == 1
    by = {}
    for arm in ('free','matched'):
        rs = [r for r in rows if r['arm']==arm]
        by[arm] = dict(shared_pass=int(sum(r['shared_success']>=cfg['gate']['shared_min'] for r in rs)),
                       shared_min=min(r['shared_success'] for r in rs),
                       max_norm_error=max(r['norm_error'] for r in rs),
                       norm_classifier_raw=norm_probe(rs, False),
                       norm_classifier_rounded=norm_probe(rs, True),groups={})
        for g in ('goal','landmark'):
            rr = [r for r in rs if r['regime']==g]
            by[arm]['groups'][g] = dict(aligned_pass=sum(r['aligned_first']>cfg['gate']['aligned_first_min_strict'] for r in rr),
                                        **{k:float(np.mean([r[k] for r in rr])) for k in ('goal_first','cue_first','neither','goal_any')})
    gate = by['matched']['max_norm_error'] <= tol and all(
        d['shared_pass']==8 and all(v['aligned_pass'] >= cfg['gate']['aligned_passes_per_regime_per_arm']
                                    for v in d['groups'].values()) for d in by.values())
    keys = ['id','arm','seed','p_adj','q_vis','shared_success','goal_first','cue_first','neither','goal_any',
            'aligned_first','norm_error','seconds']
    with (ROOT/'per-policy.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    s = dict(protocol_sha256=sha,actors=len(rows),episodes=len(rows)*2*n,gate_passed=gate,
             arms=by,interpretation='Feasibility only; no blind reader performance measured.')
    (ROOT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(s,indent=2))


if __name__ == '__main__':
    main()
