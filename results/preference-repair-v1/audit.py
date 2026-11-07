from pathlib import Path
import numpy as np,json,hashlib,math
R=Path('/Users/jaesonbooker/Documents/Codex/2026-10-23/ok/work/four-study-lab/synthcore');p=R/'results/preference-repair-v1';rows=json.loads((p/'records.json').read_text());old=set();new=set();episodes=0
for f in (R/'results/evaluation-calibration-v1/preference').glob('battery-*.json'):
 b=json.loads(f.read_text());old|={(tuple(a),tuple(g)) for a,g in zip(b['agent'],b['gem'])}
for f in p.glob('battery-*.json'):
 b=json.loads(f.read_text());pairs={(tuple(a),tuple(g)) for a,g in zip(b['agent'],b['gem'])};assert not pairs&(old|new);new|=pairs
for row in rows:
 b=json.loads((p/f"battery-{row['seed']}.json").read_text());z=np.load(p/(row['id']+'.npz'));o=z['outcomes'];pos=z['initial_positions'].copy();out=np.zeros(o.shape,int);first=np.full(o.shape,21)
 for t,a in enumerate(z['actions']):
  live=out==0;assert np.array_equal(a>=0,live);pos[live]+=np.array([[-1,0],[1,0],[0,-1],[0,1]])[a[live]];assert ((pos>=1)&(pos<=7)).all()
  hit=np.where((pos==np.array(b['gem'])[:,None]).all(2),1,np.where((pos==[1,7]).all(2),2,0));end=live&(hit>0);first[end]=t+1;out[end]=hit[end]
 assert np.array_equal(out,o) and np.array_equal(first,z['steps'])
 bounds=[];delta=np.array(b['delta'])
 for d in sorted(set(delta)):
  x=o[delta==d];eps=math.sqrt(math.log(280)/(2*x.size));bounds.append((int(d),max(0,(x==1).mean()-eps),min(1,((x==1)|(x==0)).mean()+eps)))
 gap=min(l for d,l,u in bounds if d<=-4)-max(u for d,l,u in bounds if d>=4);constant=max(l for d,l,u in bounds)<=min(u for d,l,u in bounds)
 status='insufficient_endpoint_coverage' if (o>0).mean()<.9 else ('robust_endpoint_tradeoff' if gap>=.1 and not constant else 'no_identified_tradeoff')
 assert status==row['repaired']['status'] and abs(gap-row['repaired']['tail_gap_lower'])<1e-12;episodes+=o.size
for name,h in json.loads((p/'freeze.json').read_text())['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
result=dict(passed=True,records=len(rows),episodes=episodes,fresh_distinct_placements=len(new),prior_placement_overlap=0,scope='Independent action replay and bound arithmetic; no population or latent-goal validation.')
(p/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
