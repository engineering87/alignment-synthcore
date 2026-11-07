from pathlib import Path
import json,hashlib,math
import numpy as np
R=Path('/Users/jaesonbooker/Documents/Codex/2026-10-23/ok/work/four-study-lab/synthcore');D=R/'results/trained-preference-v1';rows=json.loads((D/'records.json').read_text());moves=np.array([[-1,0],[1,0],[0,-1],[0,1]]);checks=0;episodes=0;old=set();fresh=set();maxerr=0
for slug in ['evaluation-calibration-v1/preference','preference-repair-v1']:
 for p in (R/'results'/slug).glob('battery-*.json'):
  b=json.loads(p.read_text());old|={(tuple(a),tuple(g)) for a,g in zip(b['agent'],b['gem'])}
for p in D.glob('battery-*.json'):
 b=json.loads(p.read_text());keys={(tuple(a),tuple(g)) for a,g in zip(b['agent'],b['gem'])};assert not keys&(old|fresh);fresh|=keys
for row in rows:
 b=json.loads((D/f"battery-{row['battery']}.json").read_text());z=np.load(D/(row['id']+'.npz'));w=np.load(R/f"results/preference-measurement-v2/models/{row['cohort']}.npz");i=row['policy'];n=14;k=128
 a=np.repeat(np.array(b['agent']),k,0);g=np.repeat(np.array(b['gem']),k,0);o=np.zeros(n*k,int);steps=np.full(n*k,21);rng=np.random.default_rng(row['battery']+10000*i+int('0.05' in row['cohort'])*1000000)
 wall=np.ones((9,9),np.float32);wall[1:8,1:8]=0;idx=np.arange(n*k)
 for t in range(20):
  x=np.zeros((n*k,9,9,3),np.float32);x[:,:,:,0]=wall;x[idx,a[:,0],a[:,1],1]=1;x[idx,g[:,0],g[:,1],2]=1;h=x.reshape(n*k,-1)
  for l in ['l1','l2']:h=np.maximum(h@w[l+'__w'][i]+w[l+'__b'][i],0)
  logits=h@w['pi__w'][i]+w['pi__b'][i];p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);acts=(p.cumsum(1)>rng.random((n*k,1))).argmax(1)
  live=o==0;a[live]=np.clip(a[live]+moves[acts[live]],1,7);hit=np.where((a==g).all(1),1,np.where((a==[1,7]).all(1),2,0));end=live&(hit>0);o[end]=hit[end];steps[end]=t+1
  if (o>0).all():break
 assert np.array_equal(o.reshape(n,k),z['outcomes']) and np.array_equal(steps.reshape(n,k),z['steps']);episodes+=n*k
 cells=[(r,c) for r in range(1,8) for c in range(1,8)];lookup={p:j for j,p in enumerate(cells)}
 for j,T in enumerate(z['transition_matrices']):
  assert np.allclose(T.sum(1),1,atol=1e-12)
  vg=np.zeros(49);vc=np.zeros(49);vg[lookup[tuple(b['gem'][j])]]=1;vc[lookup[(1,7)]]=1
  for t in range(20):vg=T@vg;vc=T@vc
  start=lookup[tuple(b['agent'][j])];e=max(abs(vg[start]-z['exact_probabilities'][j,0]),abs(vc[start]-z['exact_probabilities'][j,1]));maxerr=max(maxerr,e);assert e<1e-12
 for band in row['repaired']['bins']:
  take=np.array(b['delta'])==band['delta'];v=z['outcomes'][take];eps=math.sqrt(math.log(280)/(2*v.size));lo=max(0,(v==1).mean()-eps);hi=min(1,((v==1)|(v==0)).mean()+eps);assert abs(lo-band['lower'])<1e-12 and abs(hi-band['upper'])<1e-12
 checks+=1
for p,h in json.loads((D/'freeze.json').read_text())['files'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
summary=dict(passed=True,analyses=checks,sampled_episodes_reproduced=episodes,fresh_distinct_placements=len(fresh),exact_reference_max_forward_backward_error=maxerr,scope='Separate NumPy sampler and backward propagation. Saved transition construction is not independently rederived from network logits here; finite floating-point policy evaluation and conditional cohort limits remain.')
(D/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
