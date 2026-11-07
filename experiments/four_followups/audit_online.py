"""Independent geometry/observation replay for immutable PPO snapshots."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.four_followups.common import seed,verify
from testbeds.ued_gridworld import generate as G
from testbeds.ued_gridworld.public import load_policy
OUT=ROOT/'results/online-start-replay-v1';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def geometry(L,starts):
 x={'walls':L['walls'].astype(int).tolist(),'goal':L['cheese'],'lamp':L.get('lamp'),'starts':starts};return hashlib.sha256(json.dumps(x,sort_keys=True,default=lambda o:list(o) if isinstance(o,tuple) else o.item()).encode()).hexdigest()
def evaluate(net,episodes,perm):
 pos=[tuple(p) for L,p in episodes];done=np.zeros(len(episodes),bool);success=np.zeros(len(episodes),bool);cue=np.zeros(len(episodes),bool)
 moves=[(-1,0),(1,0),(0,-1),(0,1)]
 # Assert the independently encoded action convention matches the frozen contract.
 assert list(G.MOVES)==moves
 with torch.no_grad():
  for _ in range(19):
   idx=np.flatnonzero(~done)
   if not len(idx):break
   batch=[]
   for j in idx:
    L=episodes[j][0];o=np.zeros((4,7,7),np.float32);o[G.CH_WALL]=L['walls'];o[G.CH_CHEESE][tuple(L['cheese'])]=1;o[G.CH_LAMP][tuple(L['lamp'])]=1;o[G.CH_AGENT][pos[j]]=1;batch.append(o[perm])
   actions=net(torch.from_numpy(np.stack(batch))).argmax(1).numpy()
   for j,a in zip(idx,actions):
    L=episodes[j][0];dr,dc=moves[int(a)];q=(pos[j][0]+dr,pos[j][1]+dc)
    if not L['walls'][q]:pos[j]=q
    if pos[j] in L['zone'] and pos[j]!=tuple(L['cheese']):cue[j]=True
    if pos[j]==tuple(L['cheese']):done[j]=success[j]=True
 return float(success.mean()),float(cue.mean())
def check(path):
 row=read(path);info=row['snapshot'];assert sha(OUT/info['model'])==info['model_sha256'];assert info['ledger']['oracle_queries']==0 and info['ledger']['training_cpu_s']>=120
 assert info['ledger']['steps']==info['ledger']['learning_steps']+info['ledger']['scoring_steps'];assert info['ledger']['steps']<=2000000
 with np.load(OUT/info['model'],allow_pickle=False) as f:
  assert all(np.isfinite(f[n]).all() for n in f.files);p={'arch':info['arch'],'state_dict':{n:f[f'p{k}'].copy() for k,n in enumerate(info['names'])}}
 net=load_policy(p,4,7);s=row['seed_group'];family=row['family'];perm=np.asarray(row['perm']);assert row['training_seed']==seed('online-start-replay-v1','confirmation',s,f'train/{family}')
 rng=np.random.default_rng(seed('online-start-replay-v1','confirmation',s,f'eval/{family}'));counts=0
 for name,correlated in [('goal',False),('iid',True)]:
  episodes=[];hashes=[]
  for _ in range(64):
   L=G.default_level(family,rng) if correlated else G.make_level(family,G.ROOM,rng,False);starts=G._starts(L,3,rng);episodes += [(L,p) for p in starts];hashes.append(geometry(L,starts))
  assert hashes==row['evaluation_geometry_hashes'][name];ok,cue=evaluate(net,episodes,perm);assert abs(ok-row['metrics'][name])<1e-12;assert abs(cue-row['metrics'][name+'_cue'])<1e-12;counts+=len(episodes)
 return {'id':row['id'],'record_sha256':sha(path),'model_sha256':info['model_sha256'],'episodes_independently_replayed':counts,'training_cpu_s':info['ledger']['training_cpu_s'],'steps':info['ledger']['steps']}
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);verify(OUT,True);cached={};target=OUT/'incremental-integrity-audit.json'
 if target.exists():cached={x['id']:x for x in read(target)['records']}
 expected={f'{s}-{f}-{a}' for s in range(401,409) for f in ['corner','lamp'] for a in ['maxlatest','startmatched','random','dr']}
 for path in sorted((OUT/'confirmation').glob('*.json')):
  assert path.stem in expected
  if path.stem not in cached or cached[path.stem].get('record_sha256')!=sha(path):cached[path.stem]=check(path);print('AUDITED',path.stem,flush=True)
 assert set(cached)<=expected
 result={'audit_passed_for_saved_fits':True,'complete':set(cached)==expected,'fits_checked':len(cached),'expected_fits':64,'episodes_independently_replayed':sum(x['episodes_independently_replayed'] for x in cached.values()),'method':'Independent feature-tensor construction and physical movement; frozen generator and network loader reused. Not independent scientific reproduction.','records':list(cached.values())};target.write_text(json.dumps(result,indent=2)+'\n')
 if result['complete']:
  from scipy.stats import t
  receipt=read(OUT/'registration-receipt.json');assert sha(OUT/'freeze.json')==receipt['source_freeze_sha256']
  rows=[read(OUT/'confirmation'/(id+'.json')) for id in expected];summary=read(OUT/'summary.json');means={a:{k:[float(np.mean([r['metrics'][k] for r in rows if r['arm']==a and r['seed_group']==i])) for i in range(401,409)] for k in ['goal','iid']} for a in ['startmatched','maxlatest','random','dr']}
  for a in means:
   for k in means[a]:assert np.allclose(means[a][k],summary['arm_seed_means'][a][k],atol=1e-12)
  contrasts={'gain':np.array(means['startmatched']['goal'])-means['random']['goal'],'iid_floor':means['startmatched']['iid'],'iid_cost':np.array(means['startmatched']['iid'])-means['dr']['iid']}
  for name,values in contrasts.items():
   x=np.array(values);half=t.ppf(.975,7)*x.std(ddof=1)/np.sqrt(8);assert np.allclose([x.mean()-half,x.mean()+half],summary[name]['interval'],atol=1e-12)
  result['source_freeze_and_registration_verified']=True;result['primary_seed_means_and_intervals_recomputed']=True;(OUT/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
