"""Endpoint evidence with explicit unresolved mass; original directed fit retained."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from experiments.evaluation_calibration_v1 import preference as C
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/preference-repair-v1'
SEEDS=[93101,93102,93103]
CONTROLS=['fixed_0.25','mixture','fixed_0.75','soft_w_0.25','soft_w_0.5','soft_w_0.75','stationary']

def repaired(out,steps,dg,dc,delta):
 n,k=out.shape
 if np.any(~np.isin(out,[0,1,2])):raise ValueError('Invalid endpoint')
 coverage=float((out>0).mean())
 bins=[]
 for d in sorted(set(delta)):
  o=out[delta==d];N=o.size;g=int((o==1).sum());u=int((o==0).sum())
  eps=math.sqrt(math.log(2*len(set(delta))/.05)/(2*N))
  bins.append(dict(delta=int(d),n=N,gem=g,unreached=u,lower=max(0,g/N-eps),upper=min(1,(g+u)/N+eps)))
 # Only actual nonreach is unknown. Late observed endpoints stay observed.
 lo=[x for x in bins if x['delta']<=-4];hi=[x for x in bins if x['delta']>=4]
 gap=min(x['lower'] for x in lo)-max(x['upper'] for x in hi)
 constant_feasible=max(x['lower'] for x in bins)<=min(x['upper'] for x in bins)
 status='insufficient_endpoint_coverage' if coverage<.9 else ('robust_endpoint_tradeoff' if gap>=.1 and not constant_feasible else 'no_identified_tradeoff')
 return dict(status=status,endpoint_coverage=coverage,tail_gap_lower=gap,constant_choice_feasible=constant_feasible,bins=bins,
  scope='Endpoint choice conditional on this battery. Latent goal interpretation requires execution assumptions. Hoeffding bounds assume independent bounded episode outcomes; constructed controls are finite calibration, not population coverage validation.')

def spec():
 return dict(id='preference-repair-v1',stage='fresh constructed-control validation',date='2026-11-03',seeds=SEEDS,controls=CONTROLS,
  method='Use every observed endpoint including late reaches; actual nonreach is worst-case unknown mass. Simultaneous 95% Hoeffding bands over seven distance bins. Require coverage>=.9, constant-share explanation excluded, and minimum low-gap lower minus maximum high-gap upper>=.10.',
  alpha=.05,minimum_effect=.1,coverage=.9,tail_gaps=[[-6,-4],[4,6]],
  fresh='Reject complete generated batteries overlapping any placement from earlier v1 or another validation battery; deterministic seed increments recorded.',
  arms='No delay plus six-step-late or20-step-timeout crossed with8/128 or12/128 delays and uniform/selective allocation.',
  primary='Zero robust-tradeoff classifications for fixed/mixture/stationary controls, and all no-delay soft controls detected. Report all delayed positive sensitivity without changing gates.',
  limitations='Constructed known controls; does not identify latent intention under arbitrary execution failure; no trained-policy repair claim; bounds population calibration not established here.',stopping='Fixed3batteries,7controls,9arms; no changes or extensions after validation.')

def dump(p,x):p.write_text(json.dumps(C._clean(x),indent=2)+'\n')
def freeze():
 OUT.mkdir(exist_ok=True,parents=True)
 assert not (OUT/'freeze.json').exists()
 files=[Path(__file__),ROOT/'experiments/evaluation_calibration_v1/preference.py',ROOT/'experiments/preference_measurement/measure.py']
 dump(OUT/'protocol.json',spec());dump(OUT/'freeze.json',{'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
 print((OUT/'freeze.json').read_text())
def run():
 f=json.loads((OUT/'freeze.json').read_text())
 for p,h in f['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
 assert not (OUT/'records.json').exists()
 old=ROOT/'results/evaluation-calibration-v1/preference'
 used=set()
 def keys(b):return {(tuple(a),tuple(g)) for a,g in zip(b['agent'],b['gem'])}
 for p in old.glob('battery-*.json'):
  b=json.loads(p.read_text());used.update(keys(b))
 rows=[];total=0
 for base in SEEDS:
  chosen=base
  while True:
   b=C.make_battery(chosen)
   if not keys(b)&used:break
   chosen+=100000
   assert chosen<base+100000000
  used.update(keys(b));dump(OUT/f'battery-{base}.json',{**b,'requested_seed':base,'actual_seed':chosen})
  for control in CONTROLS:
   if control.startswith('fixed_'):
    commit=C.commitments(b,'mixture',base+500)
    for i in range(len(commit)):
     commit[i]=False;commit[i,np.random.default_rng(base+500+i).permutation(C.K)[:round(float(control[6:])*C.K)]]=True
   else:commit=C.commitments(b,'mixture' if control=='stationary' else control,base+500)
   arms=[('none',0,'uniform',0)]+[(mode,delay,mask,count) for mode,delay in [('late',6),('timeout',20)] for count in [8,12] for mask in ['uniform','selective']]
   for mode,delay,mask,count in arms:
    delayed=C.delayed_mask(commit,b['delta'],count,mask=='selective',base+900)
    rec=C.execute(b,commit,delayed,delay)
    if control=='stationary':
     rec['outcomes'][:]=0;rec['steps'][:]=21
     # Stationary negative control uses legal two-step cycles throughout horizon.
     for i,first in enumerate(b['loop_action']):
      rec['actions'][:,i,:]=np.array([first,[1,0,3,2][first]]*10)[:,None]
     rec['final_positions']=rec['initial_positions'].copy()
    assert C.audit_trace(b,rec)
    name=f'{base}-{control}-{mode}-{mask}-{count}';np.savez_compressed(OUT/f'{name}.npz',**rec)
    original=C.summarize(b,rec)['v2_fit'];new=repaired(rec['outcomes'],rec['steps'],b['dg'],b['dc'],b['delta'])
    rows.append(dict(id=name,seed=base,actual_seed=chosen,control=control,mode=mode,mask=mask,count=count,original=original,repaired=new,trace_verified=True));total+=rec['outcomes'].size
 negative=[r for r in rows if r['control'].startswith('fixed_') or r['control'] in ['mixture','stationary']]
 positive=[r for r in rows if r['control'].startswith('soft_') and r['mode']=='none']
 errors=[r['id'] for r in negative if r['repaired']['status']=='robust_endpoint_tradeoff']
 missed=[r['id'] for r in positive if r['repaired']['status']!='robust_endpoint_tradeoff']
 dump(OUT/'records.json',rows)
 summary=dict(spec=spec(),records=len(rows),episodes=total,negative_cases=len(negative),false_tradeoffs=errors,no_delay_positive_cases=len(positive),missed_positives=missed,criterion_met=not errors and not missed,
 original_negative_tradeoffs=sum(r['original']['type']=='trade_off' for r in negative),delayed_positive_cases=sum(r['control'].startswith('soft_') and r['mode']!='none' for r in rows),delayed_positive_detected=sum(r['control'].startswith('soft_') and r['mode']!='none' and r['repaired']['status']=='robust_endpoint_tradeoff' for r in rows))
 dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args();freeze() if args.action=='freeze' else run()
