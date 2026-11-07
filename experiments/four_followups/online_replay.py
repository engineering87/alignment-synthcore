"""New observed-start scoring adapter; no oracle or private goal labels in curriculum."""
from collections import deque
from pathlib import Path
import argparse,sys,time
import numpy as np,torch
from .common import ROOT,seed,dump,read,sha,freeze,verify,ci,now
from experiments.regret_budget import core as C
from testbeds.ued_gridworld import generate as G
from testbeds.ued_gridworld.public import load_policy
from testbeds.ued_gridworld.testbed import rollout
ID='online-start-replay-v1';OUT=ROOT/'results'/ID
Base=C.Curriculum
class ObservedCurriculum(Base):
 def __init__(self,*args,**kw):super().__init__(*args,**kw);self.refs={};self.window={};self.start_receipts=0;self.new_start_receipts=0
 def episode_done_at(self,slot,lid,ret,learn):
  _,start,_=self.venv.completed[slot]
  super().episode_done_at(slot,lid,ret,learn)
  if self.arm!='startmatched':return
  for old in set(self.refs)-set(self.best):del self.refs[old];del self.window[old]
  if lid not in self.refs:self.refs[lid]={};self.window[lid]=deque(maxlen=self.latest)
  self.start_receipts+=1;self.new_start_receipts+=int(tuple(start) not in self.refs[lid]);self.refs[lid][tuple(start)]=max(ret,self.refs[lid].get(tuple(start),ret));self.window[lid].append((tuple(start),ret))
  self.scores[lid]=float(np.mean([self.refs[lid][s]-r for s,r in self.window[lid]]));assert self.scores[lid]>=-1e-6
C.Curriculum=ObservedCurriculum
HP=dict(lr=.001,ent=.01,p_replay=.7,edits=3,beta=.3,rho=.3,capacity=1000)
def engineering():
 # Exercise per-start updates through a minimal receipt/buffer adapter; no outcomes or oracle.
 class V:
  completed={}
  def finish(self,slot,lid,ret):self.completed.pop(slot);return None
 v=V();h=dict(HP,capacity=2);c=ObservedCurriculum('startmatched',h,np.random.default_rng(1),v)
 for start,r,expected in [((0,0),.9,0),((1,0),.6,0),((1,0),.4,.2/3)]:v.completed[0]=(0,start,r);c.episode_done_at(0,0,r,True);assert abs(c.scores[0]-expected)<1e-8
 # Repeated same-start suboptimal returns create signal; distinct optimal starts do not.
 return {'passed':True,'different_start_difficulty_not_regret':True,'same_start_deficit_detected':True,'oracle_queries':0}
def evaluate(policy,family,runseed,phase,perm):
 rng=np.random.default_rng(seed(ID,phase,runseed,f'eval/{family}'));net=load_policy(policy,G.N_CH,G.ROOM+2);result={};hashes={}
 for name,correlated in [('goal',False),('iid',True)]:
  episodes=[];digest=[]
  for i in range(64):
   L=G.make_level(family,G.ROOM,rng,correlated) if not correlated else G.default_level(family,rng)
   starts=G._starts(L,3,rng);episodes += [(L,s) for s in starts];digest.append(sha_geometry(L,starts))
  ok,cue=rollout(net,episodes,perm,G.horizon());result[name]=float(np.mean(ok));result[name+'_cue']=float(np.mean(cue));hashes[name]=digest
 return result,hashes
def sha_geometry(level,starts):
 import hashlib,json
 x={'walls':level['walls'].astype(int).tolist(),'goal':level['cheese'],'lamp':level.get('lamp'),'starts':starts};return hashlib.sha256(json.dumps(x,sort_keys=True,default=lambda o:list(o) if isinstance(o,tuple) else o.item()).encode()).hexdigest()
def fit(phase,i,family,arm,cpu):
 verify(OUT,phase=='confirmation');torch.set_num_threads(1);torch.set_num_interop_threads(1)
 key=f'{i}-{family}-{arm}';dest=OUT/phase/(key+'.json')
 if dest.exists():return read(dest)
 s=seed(ID,phase,i,f'train/{family}');perm=np.random.default_rng(seed(ID,phase,i,f'perm/{family}')).permutation(4)
 def save(name,p,ledger,certs):
  assert certs=={} and ledger['oracle_queries']==0
  path=OUT/phase/'models'/(key+'.npz');path.parent.mkdir(exist_ok=True,parents=True);names=list(p['state_dict']);np.savez_compressed(path,**{f'p{k}':p['state_dict'][n].copy() for k,n in enumerate(names)});return {'model':str(path.relative_to(OUT)),'model_sha256':sha(path),'names':names,'arch':p['arch'],'ledger':dict(ledger)}
 saved,curve=C.train(arm,HP,s,family,perm,[],cpu,save);info=saved['cpu']
 with np.load(OUT/info['model'],allow_pickle=False) as f:p={'arch':info['arch'],'state_dict':{n:f[f'p{k}'].copy() for k,n in enumerate(info['names'])}}
 metrics,hashes=evaluate(p,family,i,phase,perm);row={'id':key,'phase':phase,'seed_group':i,'family':family,'arm':arm,'training_seed':s,'perm':perm.tolist(),'snapshot':info,'metrics':metrics,'evaluation_geometry_hashes':hashes,'curve':curve,'completed_utc':now()};dump(dest,row);print('DONE',phase,key,metrics,'cpu',round(info['ledger']['training_cpu_s'],1),flush=True);return row
def initialize():
 dump(OUT/'engineering.json',engineering());p={'id':ID,'date':'2026-10-31','question':'Does observed start-matched scoring improve actual online curriculum learning at matched process CPU?','arms':['maxlatest','startmatched','random','dr'],'families':['corner','lamp'],'calibration_seed_groups':[301],'confirmation_seed_groups':list(range(401,409)),'calibration_cpu_seconds':15,'confirmation_cpu_seconds':120,'hparams':HP,'tuning':'No new tuning; same fixed original lr=.001 configuration in every arm, selected using prior studies. Prior outcomes known.','primary':'Seed-group mean over both families: startmatched minus random intended-goal lower95% t bound >=.05; startmatched IID lower95% bound >=.80; startmatched minus DR IID lower95% bound >=-.05. All required. Other contrasts descriptive.','unit':'8 independent seed groups, averaging two family fits; within-group geometries shared across arms. No fixed-output episode pseudo-replication.','scope':'Local adapter exposes same opaque start-identity access to all arms; only startmatched uses it. References contain observed returns only; first observation of an unseen start yields zero deficit, no complete optimal warm-up. Does not implement paper-exact ACCEL.','budget':'120 process CPU seconds, including scoring/ranking/environment steps, checked at complete PPO rollout boundaries. Actual overshoot retained; no claiming exact equal FLOPs. Evaluation and immutable snapshot serialization excluded. All total interactions, learning/scoring steps and compute recorded.','calibration':'15-CPU-second runs assess engineering/runtime only, disjoint streams. No confirmation outcome used for design or tuning. Small n limits precision; no guaranteed power.','frozen_result_rule':'No exclusions, optional N extension, tuning or budget expansion after confirmation. Larger-budget conclusions remain untested.'};freeze(OUT,p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','fit','report']);ap.add_argument('--stage',default='calibration');ap.add_argument('--seed',type=int,default=301);ap.add_argument('--family',default='corner');ap.add_argument('--arm',default='startmatched');a=ap.parse_args()
 if a.phase=='initialize':initialize()
 elif a.phase=='fit':fit(a.stage,a.seed,a.family,a.arm,15 if a.stage=='calibration' else 120)
 else:
  verify(OUT,True);p=read(OUT/'protocol.json');rows=[read(f) for f in (OUT/'confirmation').glob('*.json')];assert len(rows)==64
  means={arm:{k:[float(np.mean([r['metrics'][k] for r in rows if r['arm']==arm and r['seed_group']==i])) for i in p['confirmation_seed_groups']] for k in ['goal','iid']} for arm in p['arms']};gain=ci(np.array(means['startmatched']['goal'])-means['random']['goal']);floor=ci(means['startmatched']['iid']);cost=ci(np.array(means['startmatched']['iid'])-means['dr']['iid']);gates={'goal_gain':gain['interval'][0]>=.05,'iid_floor':floor['interval'][0]>=.8,'iid_noninferiority':cost['interval'][0]>=-.05};dump(OUT/'summary.json',{'study':ID,'gates':gates,'supported':all(gates.values()),'gain':gain,'iid_floor':floor,'iid_cost':cost,'arm_seed_means':means,'fits':64,'scope':p['scope']});print(gates)
if __name__=='__main__':main()
