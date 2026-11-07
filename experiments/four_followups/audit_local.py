"""Separate saved-record audit. No training, outcome selection or threshold changes."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.four_followups.common import verify,seed
from experiments.oversight_coverage.environment import environment
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def interval(v,alpha):
 a=np.asarray(v,float);h=t.ppf(1-alpha/2,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a));return [float(a.mean()-h),float(a.mean()+h)]
def distances(T,terminal):
 rows=[];cols=[]
 for s in range(len(T)):
  if terminal[s]:continue
  for a in range(5):
   if T[s,a]>=0:rows.append(s);cols.append(T[s,a])
 graph=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(T),len(T)));return shortest_path(graph,directed=True,unweighted=True)
def penalties(T,terminal):
 D=distances(T,terminal);reach=.95**D;P=np.zeros(T.shape)
 for s in range(len(T)):
  if terminal[s] or T[s,4]<0:continue
  for a,n in enumerate(T[s]):
   if n<0 or terminal[n]:continue
   P[s,a]=np.maximum(0,reach[T[s,4]]-reach[n]).mean()
 return P,D
def qtable(T,R,terminal,gamma):
 V=np.zeros(len(T))
 for _ in range(2000):
  Q=R+gamma*V[T];Q[terminal]=0;new=Q.max(1);new[terminal]=0
  if np.max(abs(new-V))<1e-12:return Q
  V=new
 raise AssertionError('Independent Bellman solver failed')
def trajectory(T,terminal,R,harm,actions):
 s=0;ret=0.;event=False;path=[0]
 for _ in range(30):
  if terminal[s]:break
  a=actions[s];ret+=R[s,a];event|=bool(harm[s,a]);s=T[s,a];path.append(int(s))
 return {'goal':int(terminal[s]),'harm':int(event),'benign_return':float(ret),'path':path}
def check_freeze(out):
 f=verify(out,True);r=read(out/'registration-receipt.json');assert sha(out/'freeze.json')==r['source_freeze_sha256'];return len(f['source_hashes'])
def rr():
 out=ROOT/'results/rr-coverage-v1';source_count=check_freeze(out);expected={f'seed-{i}.json' for i in range(701,733)};assert {p.name for p in (out/'confirmation').glob('*.json')}==expected
 graph_count=0;trajectory_count=0;false=[];gain=[];harmdiff=[];baseharm=[];fullharm=[];partialharm=[]
 for name in sorted(expected):
  r=read(out/'confirmation'/name);fvalues=[];group={}
  for row in r['rows']:
   file=out/row['tables'];assert sha(file)==row['tables_sha256']
   with np.load(file,allow_pickle=False) as z:data={k:z[k] for k in z.files}
   T=data['T'];term=data['terminal'];R=data['R_env'];H=data['harm'];Q=qtable(T,R,term,.9);assert np.max(abs(data['objective']-(R+Q)))<1e-8
   Pf,Df=penalties(T,term);assert np.max(abs(Pf-data['full_penalty']))<1e-12
   for retain in [1.,.5,.25]:
    key=str(retain);M=data['graph_'+key];P,D=penalties(M,term);assert np.max(abs(P-data['penalty_'+key]))<1e-12;assert np.array_equal(M[:,4],T[:,4]);assert np.all((M==-1)|(M==T));block=row['blocks'][key];actions=np.argmax(data['objective']-.1*P,axis=1);assert actions.tolist()==block['actions'];metrics=trajectory(T,term,R,H,actions)
    for k in metrics:assert metrics[k]==block['metrics'][k] if k!='benign_return' else abs(metrics[k]-block['metrics'][k])<1e-10
    candidate=(M>=0)&(~term[:,None])&(~term[T]);eligible=np.zeros(T.shape,bool);bad=np.zeros(T.shape,bool)
    for s,a in np.argwhere(candidate):
     if np.isfinite(Df[T[s,a],T[s,4]]):eligible[s,a]=True;bad[s,a]=not np.isfinite(D[M[s,a],M[s,4]])
    fraction=float(bad.sum()/eligible.sum());assert abs(fraction-block['false_unreachable_fraction'])<1e-12;graph_count+=1;trajectory_count+=1
   if row['kind']=='reversible':gain.append(row['blocks']['1.0']['metrics']['goal']-row['blocks']['0.5']['metrics']['goal'])
   else:
    baseharm.append(row['base']['harm']);fullharm.append(row['blocks']['1.0']['metrics']['harm']);partialharm.append(row['blocks']['0.5']['metrics']['harm']);harmdiff.append(fullharm[-1]-partialharm[-1])
   fvalues.append(row['blocks']['0.5']['false_unreachable_fraction'])
  false.append(float(np.mean(fvalues)))
 summary=read(out/'summary.json')
 for data,key in [(false,'false_fraction'),(gain,'completion_gain'),(harmdiff,'harm_increase')]:assert np.allclose(interval(data,.05),summary[key]['interval'],atol=1e-12)
 result={'audit_passed':True,'source_files_verified':source_count,'groups':32,'graphs_independently_recomputed':graph_count,'policies_and_trajectories_checked':trajectory_count,'independent_shortest_paths':'SciPy sparse directed unweighted paths, separate from study BFS','hazard_rates':{'base':float(np.mean(baseharm)),'rr_full':float(np.mean(fullharm)),'rr_half':float(np.mean(partialharm))},'meaning':'Saved-record implementation audit, not independent researcher reproduction; frozen unsupported verdict unchanged.'};(out/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('RR AUDIT',result,flush=True)
def policy_score(e,policy):
 pi=np.full(e['T'].shape,.2);known=np.asarray(e['terminal']).copy()
 for s,id in enumerate(e['obs_id']):
  if str(int(id)) in policy:pi[s]=0;pi[s,int(policy[str(int(id))])]=1;known[s]=True
 d=np.zeros(len(pi));d[0]=1;good=d.copy();harm=ret=true=0.
 for _ in range(30):
  w=d[:,None]*pi;harm+=float((w*e['ev_vase']).sum());ret+=float((w*e['R_env']).sum());true+=float((w*e['R_true']).sum());d=np.bincount(e['T'].ravel(),weights=w.ravel(),minlength=len(pi));good=np.bincount(e['T'].ravel(),weights=(good[:,None]*pi*known[:,None]).ravel(),minlength=len(pi))
 return {'goal_rate':float(d[e['terminal']].sum()),'harm_rate':harm,'environment_return':ret,'hidden_return':true,'unknown_state_probability':float(1-good.sum())}
def mask():
 out=ROOT/'results/changing-mask-factorial-v1';count=check_freeze(out);expected={f'seed-{i}.json' for i in range(211,243)};assert {p.name for p in (out/'confirmation').glob('*.json')}==expected;trace_count=policies=0;groups=[]
 for filename in sorted(expected):
  record=read(out/'confirmation'/filename);assert len(record['rows'])==6
  for row in record['rows']:
   e=environment(row['environment_seed'],row['coverage'],row['regime']);path=out/row['trace'];assert sha(path)==row['trace_sha256']
   with np.load(path,allow_pickle=False) as z:tr={k:z[k] for k in z.files}
   assert len(tr['obs'])==20000;inv={int(v):s for s,v in enumerate(e['obs_id'])};obs=np.array([inv[int(v)] for v in tr['obs']]);nxt=np.array([inv[int(v)] for v in tr['next_obs']]);assert np.array_equal(nxt,e['T'][obs,tr['action']]);assert np.array_equal(tr['terminated'],e['terminal'][nxt]);assert np.allclose(tr['reward'],e['R_env'][obs,tr['action']],atol=1e-12)
   visible=e['masks'][:,e['cell_index'][e['vase']]].astype(int);noise=tr['original']-e['approvals'][tr['mask_index'],obs,tr['query_action']];A=[]
   for key in ['R_env','R_true']:
    Q=qtable(e['T'],e[key],e['terminal'],.95);A.append(Q-Q.max(1)[:,None])
   expected_matched=np.stack(A)[visible[tr['mask_index']],obs,tr['query_action']]+noise;assert np.max(abs(expected_matched-tr['matched']))<1e-8
   ids=np.unique(np.r_[tr['obs'],tr['next_obs']]);s=np.searchsorted(ids,tr['obs']);ns=np.searchsorted(ids,tr['next_obs']);S=len(ids);ac=np.bincount(s*5+tr['query_action'],minlength=S*5).reshape(S,5);tc=np.bincount(s*5+tr['action'],minlength=S*5).reshape(S,5);means=np.bincount(s*5+tr['query_action'],weights=tr['original'],minlength=S*5).reshape(S,5)/np.maximum(ac,1);sp=np.maximum(.1,np.ptp(means,axis=1));scaled=tr['original']/sp[s];known=(ac>0)&(tc>0);T=np.full((S,5),-1,int);T[s,tr['action']]=ns;terminal=np.zeros(S,bool);terminal[ns[tr['terminated']]]=True;rw=np.bincount(s*5+tr['action'],weights=tr['reward'],minlength=S*5).reshape(S,5)/np.maximum(tc,1)
   for name,arm in row['arms'].items():
    gamma=0 if name.endswith('myopic') else .95;labels=scaled if name.startswith('span') else tr['matched'] if name.startswith('matched') else np.zeros(20000) if name.startswith('reward') else tr['original'];av=np.bincount(s*5+tr['query_action'],weights=labels,minlength=S*5).reshape(S,5)/np.maximum(ac,1);R=rw+av;V=np.zeros(S)
    for _ in range(1500):
     Q=np.where(known,R+gamma*V[np.maximum(T,0)],-1e6);Q[terminal]=0;new=Q.max(1);new[~known.any(1)]=0;new[terminal]=0
     if np.max(abs(new-V))<1e-12:break
     V=new
    expected_ids={str(int(ids[j])) for j in range(S) if known[j].any() and not terminal[j]};assert set(arm['policy'])==expected_ids
    for j in range(S):
     if str(int(ids[j])) in arm['policy']:assert Q[j].max()-Q[j,int(arm['policy'][str(int(ids[j]))])]<1e-8
    metrics=policy_score(e,arm['policy'])
    for k,value in metrics.items():assert abs(value-arm['metrics'][k])<1e-9,(record['seed_group'],name,k)
    policies+=1
   trace_count+=1
  cells=[r for r in record['rows'] if r['regime']=='resampled'];groups.append({n:{k:float(np.mean([r['arms'][n]['metrics'][k] for r in cells])) for k in ['goal_rate','harm_rate']} for n in cells[0]['arms']})
 summary=read(out/'summary.json');pairs={'horizon_effect':[g['original_myopic']['goal_rate']-g['original_full']['goal_rate'] for g in groups],'normalization_effect':[g['span_full']['goal_rate']-g['original_full']['goal_rate'] for g in groups],'completion_floor':[g['span_full']['goal_rate'] for g in groups],'harm_reduction':[g['reward_only_full']['harm_rate']-g['span_full']['harm_rate'] for g in groups],'completion_cost':[g['span_full']['goal_rate']-g['reward_only_full']['goal_rate'] for g in groups]}
 for name,data in pairs.items():assert np.allclose(interval(data,.05/3),summary[name]['interval'],atol=1e-12)
 result={'audit_passed':True,'source_files_verified':count,'groups':32,'traces_checked':trace_count,'transitions_checked':trace_count*20000,'policies_checked':policies,'checks':['private graph transitions and reward','query-action label assignment','same-noise .9/.95 approval streams','observed-only span normalization','learned-table Bellman greediness within1e-8','independent distribution propagation and primary intervals'],'meaning':'Saved-record audit, not independent scientific reproduction. Original Test4 online methods are not rerun; this is the declared passive-trace adapter.'};(out/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('MASK AUDIT',result,flush=True)
if __name__=='__main__':rr();mask()
