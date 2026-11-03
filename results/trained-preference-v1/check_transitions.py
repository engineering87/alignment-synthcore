from pathlib import Path
import numpy as np,json
R=Path('/Users/jaesonbooker/Documents/Codex/2026-10-23/ok/work/four-study-lab/synthcore');D=R/'results/trained-preference-v1';err=0;count=0
cells=np.array([(r,c) for r in range(1,8) for c in range(1,8)]);lookup={tuple(x):i for i,x in enumerate(cells)};moves=np.array([[-1,0],[1,0],[0,-1],[0,1]])
for row in json.loads((D/'records.json').read_text()):
 b=json.loads((D/f"battery-{row['battery']}.json").read_text());saved=np.load(D/(row['id']+'.npz'));w=np.load(R/f"results/preference-measurement-v2/models/{row['cohort']}.npz");i=row['policy']
 for j in range(14):
  obs=np.zeros((49,9,9,3),float);obs[:,:,:,0]=1;obs[:,1:8,1:8,0]=0;idx=np.arange(49);obs[idx,cells[:,0],cells[:,1],1]=1;g=b['gem'][j];obs[:,g[0],g[1],2]=1;h=obs.reshape(49,-1)
  for l in ['l1','l2']:h=np.maximum(h@w[l+'__w'][i].astype(float)+w[l+'__b'][i],0)
  logits=h@w['pi__w'][i].astype(float)+w['pi__b'][i];p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);T=np.zeros((49,49))
  for s,pos in enumerate(cells):
   if tuple(pos) in [tuple(g),(1,7)]:T[s,s]=1
   else:
    for a in range(4):T[s,lookup[tuple(np.clip(pos+moves[a],1,7))]]+=p[s,a]
  e=np.max(np.abs(T-saved['transition_matrices'][j]));err=max(err,float(e));assert e<1e-5;count+=1
x=dict(passed=True,transition_tables_checked=count,max_float64_vs_float32_probability_difference=err,tolerance=1e-5)
(D/'transition-audit.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
