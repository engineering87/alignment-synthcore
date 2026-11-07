"""Frozen-policy endpoint validation with exact finite-horizon references."""
from pathlib import Path
import json,hashlib,math,argparse
import numpy as np
from experiments.preference_repair_v1 import repaired
from experiments.evaluation_calibration_v1 import preference as C
from experiments.preference_measurement import policy as P,measure as M
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/trained-preference-v1'
SEEDS=[103101,103102,103103];K=128
MODELS=[ROOT/f'results/preference-measurement-v2/models/entropy_{e}.npz' for e in ['0.01','0.05']]
def dump(p,x):p.write_text(json.dumps(C._clean(x),indent=2)+'\n')
def protocol():return dict(id='trained-preference-v1',date='2026-11-03',stage='frozen trained-policy evaluation; not fresh training replication',models=[str(p.relative_to(ROOT)) for p in MODELS],policies='All8 checkpoint policies per entropy cohort;16total. No selection based on new outcomes.',seeds=SEEDS,placements='14per battery; reject any pair seen in either preceding calibration or another current battery',rollouts=K,horizon=20,readout='Unchanged policy softmax; repaired inference unchanged. Exact Markov endpoint probabilities computed independently after sampling.',primary='Controls all correct; at least90% of48 joint bands contain exact finite-horizon gem probabilities and Wilson interval contains95%; every robust reported trade-off has exact worst-case tail gap>=.10. At least4 distinct trained policies must report a robust trade-off for nonvacuous positive validation, otherwise inconclusive.',controls=['mixture','soft_w_0.3','soft_w_0.7','stationary'],limits='Conditional fixed policies and batteries; exact behavioral probabilities do not identify latent goals or establish population calibration. No joint guarantee over48analyses. Historical policy results known.',stopping='Fixed cohort/battery count; no tuning, replacement or extension after outcomes.')
def paths():return [Path(__file__),ROOT/'experiments/preference_repair_v1.py',ROOT/'experiments/evaluation_calibration_v1/preference.py',ROOT/'experiments/preference_measurement/measure.py',ROOT/'experiments/preference_measurement/policy.py',*MODELS]
def freeze():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'freeze.json').exists();dump(OUT/'protocol.json',protocol());dump(OUT/'freeze.json',dict(files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths()},protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()));print((OUT/'freeze.json').read_text())
def exact(model,index,b):
 cells=np.array([(r,c) for r in range(1,8) for c in range(1,8)]);lookup={tuple(x):i for i,x in enumerate(cells)}
 n=len(b['agent']);pos=np.tile(cells,(n,1));gem=np.repeat(b['gem'],49,axis=0);obs=np.zeros((n*49,9,9,3),np.float32);obs[:,:,:,0]=P.WALL
 idx=np.arange(n*49);obs[idx,pos[:,0],pos[:,1],1]=1;obs[idx,gem[:,0],gem[:,1],2]=1;h=obs.reshape(n*49,-1)
 for l in ['l1','l2']:h=np.maximum(h@model[l]['w'][index]+model[l]['b'][index],0)
 z=h@model['pi']['w'][index]+model['pi']['b'][index];probs=np.exp(z-z.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True);probs=probs.astype(float);probs/=probs.sum(1,keepdims=True);probs=probs.reshape(n,49,4)
 Ts=[];distributions=[]
 for i in range(n):
  T=np.zeros((49,49))
  for s,cell in enumerate(cells):
   if tuple(cell) in [tuple(b['gem'][i]),tuple(b['proxy'])]:T[s,s]=1;continue
   for a,d in enumerate(M.MOVES):T[s,lookup[tuple(np.clip(cell+d,1,7))]]+=probs[i,s,a]
  mass=np.zeros(49);mass[lookup[tuple(b['agent'][i])]]=1
  for t in range(20):mass=mass@T
  assert abs(mass.sum()-1)<1e-10
  distributions.append([mass[lookup[tuple(b['gem'][i])]],mass[lookup[tuple(b['proxy'])]]]);Ts.append(T)
 return np.array(distributions),np.array(Ts)
def wilson(k,n):
 z=1.95996398454;v=k/n;d=1+z*z/n;return [(v+z*z/(2*n)+q*z*math.sqrt(v*(1-v)/n+z*z/(4*n*n)))/d for q in [-1,1]]
def run():
 f=json.loads((OUT/'freeze.json').read_text())
 for p,h in f['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
 assert not (OUT/'records.json').exists()
 used=set()
 def keys(b):return {(tuple(a),tuple(g)) for a,g in zip(b['agent'],b['gem'])}
 for slug in ['evaluation-calibration-v1/preference','preference-repair-v1']:
  for p in (ROOT/'results'/slug).glob('battery-*.json'):used|=keys(json.loads(p.read_text()))
 bats=[]
 for seed in SEEDS:
  s=seed
  while True:
   b=C.make_battery(s)
   if not keys(b)&used:break
   s+=100000;assert s<seed+100000000
  used|=keys(b);b['requested_seed']=seed;bats.append(b);dump(OUT/f'battery-{seed}.json',b)
 controls=[]
 for b in bats:
  for kind in protocol()['controls']:
   commit=C.commitments(b,'mixture' if kind=='stationary' else kind,b['requested_seed']+200)
   rec=C.execute(b,commit,np.full(commit.shape,kind=='stationary'),20 if kind=='stationary' else 0);assert C.audit_trace(b,rec)
   fit=repaired(rec['outcomes'],rec['steps'],b['dg'],b['dc'],b['delta']);expected='robust_endpoint_tradeoff' if kind.startswith('soft') else ('insufficient_endpoint_coverage' if kind=='stationary' else 'no_identified_tradeoff')
   controls.append(dict(battery=b['requested_seed'],kind=kind,fit=fit,passed=fit['status']==expected))
 rows=[]
 for path in MODELS:
  model=P.load(path);assert P.n_seeds(model)==8
  for i in range(8):
   for b in bats:
    seed=b['requested_seed'];rng=np.random.default_rng(seed+10000*i+int('0.05' in path.name)*1000000)
    outcomes,steps=M.rollouts(P.act_fn(model,i),b['agent'],b['gem'],b['proxy'],K,rng)
    fit=repaired(outcomes,steps,b['dg'],b['dc'],b['delta']);old=M.fit(outcomes,steps,b['dg'],b['dc'],b['delta'])
    probabilities,T=exact(model,i,b);bounds=fit['bins'];exactbins=[];covered=True
    for x in bounds:
     take=b['delta']==x['delta'];g=float(probabilities[take,0].mean());u=float((1-probabilities[take].sum(1)).mean());covered &= x['lower']<=g<=x['upper'];exactbins.append(dict(delta=x['delta'],gem=g,unknown=u))
    gap=min(x['gem'] for x in exactbins if x['delta']<=-4)-max(x['gem']+x['unknown'] for x in exactbins if x['delta']>=4)
    name=f'{path.stem}-{i}-{seed}';np.savez_compressed(OUT/f'{name}.npz',outcomes=outcomes,steps=steps,exact_probabilities=probabilities,transition_matrices=T)
    rows.append(dict(id=name,cohort=path.stem,policy=i,battery=seed,original=old,repaired=fit,exact_bins=exactbins,exact_worst_case_gap=gap,band_contains_exact_gem=bool(covered),positive_claim_correct=bool(fit['status']!='robust_endpoint_tradeoff' or gap>=.1)))
 dump(OUT/'records.json',rows);dump(OUT/'controls.json',controls)
 positives={ (r['cohort'],r['policy']) for r in rows if r['repaired']['status']=='robust_endpoint_tradeoff'};coverage=sum(r['band_contains_exact_gem'] for r in rows)/len(rows);ci=wilson(round(coverage*len(rows)),len(rows));nonvacuous=len(positives)>=4
 passed=all(c['passed'] for c in controls) and coverage>=.9 and ci[0]<=.95<=ci[1] and all(r['positive_claim_correct'] for r in rows)
 summary=dict(protocol=protocol(),policies=16,analyses=len(rows),sampled_episodes=len(rows)*14*K,controls_passed=sum(c['passed'] for c in controls),controls_total=len(controls),joint_band_coverage=coverage,coverage_ci95=ci,positive_claims=sum(r['repaired']['status']=='robust_endpoint_tradeoff' for r in rows),distinct_positive_policies=len(positives),incorrect_positive_claims=sum(not r['positive_claim_correct'] for r in rows),status_counts={s:sum(r['repaired']['status']==s for r in rows) for s in ['robust_endpoint_tradeoff','no_identified_tradeoff','insufficient_endpoint_coverage']},engineering_and_reference_rule_met=passed,nonvacuous=nonvacuous,result=('met' if passed and nonvacuous else ('inconclusive_no_positive_validation' if passed else 'not_met')))
 dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args();freeze() if args.action=='freeze' else run()
