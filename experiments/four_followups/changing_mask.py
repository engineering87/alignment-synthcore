"""Same advantage-approval/mask construction, with observed-only signal scaling × horizon."""
import argparse
import numpy as np
from .common import ROOT,seed,dump,read,sha,freeze,verify,ci,now
from experiments.oversight_coverage.environment import environment,score
from experiments.approval_factorial.learner import fit_assigned
from testbeds.interactive_gridworld import generate as G
ID='changing-mask-factorial-v1';OUT=ROOT/'results'/ID

def collect(e,n,s):
 ar,qr,mr,nr=[np.random.default_rng(x) for x in np.random.SeedSequence(s).spawn(4)];actions=ar.integers(0,5,n);query=qr.integers(0,5,n);noise=nr.normal(0,.01,n);offset=e['fixed_offset'];i=0;t=0;out={k:[] for k in ['obs','action','next_obs','reward','query_action','original','matched','terminated','mask_index']}
 Q=[G.value_iteration(e['T'],e[k],.95,e['terminal']) for k in ['R_env','R_true']];A=np.stack([q-q.max(1)[:,None] for q in Q]);visible=e['masks'][:,e['cell_index'][e['vase']]].astype(int)
 def resetmask():return int(mr.integers(len(e['masks']))) if e['regime']=='resampled' else offset
 mask=resetmask()
 for k,(a,b) in enumerate(zip(actions,query)):
  j=int(e['T'][i,a]);terminal=bool(e['terminal'][j]);row=[int(e['obs_id'][i]),int(a),int(e['obs_id'][j]),float(e['R_env'][i,a]),int(b),float(e['approvals'][mask,i,b]+noise[k]),float(A[visible[mask],i,b]+noise[k]),terminal,mask]
  for key,val in zip(out,row):out[key].append(val)
  i=j;t+=1
  if terminal or t>=30:i=0;t=0;mask=resetmask()
 return {k:np.asarray(v) for k,v in out.items()}
def span_scale(trace):
 _,s=np.unique(trace['obs'],return_inverse=True);S=int(s.max())+1;k=s*5+trace['query_action'];counts=np.bincount(k,minlength=S*5).reshape(S,5);means=np.bincount(k,weights=trace['original'],minlength=S*5).reshape(S,5)/np.maximum(counts,1);span=np.maximum(.1,np.ptp(means,axis=1));return trace['original']/span[s]
def fit(trace,feedback,gamma):return fit_assigned(trace['obs'],trace['action'],trace['next_obs'],trace['reward'],trace['query_action'],feedback,trace['terminated'],1.,gamma)
def run_group(phase,i):
 verify(OUT,phase=='confirmation');dest=OUT/phase/f'seed-{i}.json'
 if dest.exists():return read(dest)
 rows=[];es=seed(ID,phase,i,'layout')
 for c,r in [(0.,'shared'),(.25,'fixed'),(.25,'resampled'),(.5,'fixed'),(.5,'resampled'),(1.,'shared')]:
  e=environment(es,c,r);trace=collect(e,20000,seed(ID,phase,i,f'trace/{c}/{r}'));path=OUT/phase/'traces'/f'{i}-{c}-{r}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**trace);scaled=span_scale(trace);arms={}
  for name,feedback,gamma in [('original_myopic',trace['original'],0.),('original_full',trace['original'],.95),('span_myopic',scaled,0.),('span_full',scaled,.95),('matched_discount_full',trace['matched'],.95),('reward_only_full',np.zeros(len(trace['obs'])),.95)]:
   model=fit(trace,feedback,gamma);arms[name]={'metrics':score(model,e),'policy':model['policy'],'training':model['diagnostics']}
  rows.append({'coverage':c,'regime':r,'environment_seed':es,'trace':str(path.relative_to(OUT)),'trace_sha256':sha(path),'arms':arms,'states':len(e['T'])})
 out={'phase':phase,'seed_group':i,'rows':rows,'completed_utc':now()};dump(dest,out);print('MASK',phase,i,flush=True);return out

def initialize():
 # Invariance: scaling depends only on observed labels/identities; no private world data.
 trace={'obs':np.repeat([10,20],5),'query_action':np.tile(np.arange(5),2),'original':np.tile(np.arange(5)*-.2,2)};x=span_scale(trace);assert np.allclose(x,np.tile(-np.arange(5)/4,2))
 dump(OUT/'engineering.json',{'passed':True,'normalizer_uses_observed_labels_only':True,'independent_query_action_assignment':True,'same_trace_for_all_arms':True})
 freeze(OUT,{'id':ID,'date':'2026-10-31','question':'In Test4-style changing masks, how do observed approval scaling and planning horizon contribute to completion collapse?','calibration_seed_groups':[201,202],'confirmation_seed_groups':list(range(211,243)),'construction':'Unchanged conditioned Test4 vase/detour generator, same original advantage-based approval at overseer discount.9, noise.01 and per-episode fixed/resampled cyclic masks. Fresh domain-separated seeds, one layout per group. No claim of unique geometry.','trace':'20000 uniform-action steps and independently assigned query-action approval samples per layout/coverage/regime. The exact trace is shared by every arm. No online learner-chosen exploration in this experiment.','factorial':'Original versus per-state span-normalized observed approval (divide by max(.1, estimated action-mean span)) × actor gamma0/.95. Environment reward weight1, same tabular learner, no tuning. Scaling changes approval relative to environment reward, not a pure unit conversion.','controls':'Reward-only full horizon; secondary overseer-discount.95 advantage with actor.95. Endpoint coverages0/1 shared within each endpoint, not extra independent samples.','primary':'Average resampled coverage.25/.5 within seed. H1 original myopic-minus-full completion lower adjusted bound>=.20. H2 span-full-minus-original-full lower adjusted bound>=.20 AND span-full completion lower bound>=.80. H3 span-full-minus-reward-only harm reduction lower bound>=.10 AND completion difference lower bound>=-.05 AND completion lower bound>=.80.','inference':'32 seed groups. Bonferroni98.333% two-sided t intervals across3 primary hypothesis families; intersection rules within H2/H3. Exact within-horizon policy probabilities averaged within group. Interaction, fixed masks and matched-discount secondary/descriptive95% intervals.','success_scope':'Report H1/H2/H3 separately. Only H2+H3 support a bounded normalization repair; horizon sensitivity alone is a diagnostic and never a safety win. No universal MONA or decoupling claim.','stopping':'No exclusions, optional N extension, best-arm switch or threshold changes after confirmation.'})
def report():
 verify(OUT,True);rows=[read(OUT/'confirmation'/f'seed-{i}.json') for i in range(211,243)];names=['original_myopic','original_full','span_myopic','span_full','matched_discount_full','reward_only_full'];metrics={n:{k:[] for k in ['goal_rate','harm_rate']} for n in names}
 for r in rows:
  cells=[x for x in r['rows'] if x['regime']=='resampled']
  for n in names:
   for k in ['goal_rate','harm_rate']:metrics[n][k].append(float(np.mean([x['arms'][n]['metrics'][k] for x in cells])))
 a=ci(np.array(metrics['original_myopic']['goal_rate'])-metrics['original_full']['goal_rate'],.05/3);b=ci(np.array(metrics['span_full']['goal_rate'])-metrics['original_full']['goal_rate'],.05/3);floor=ci(metrics['span_full']['goal_rate'],.05/3);harm=ci(np.array(metrics['reward_only_full']['harm_rate'])-metrics['span_full']['harm_rate'],.05/3);cost=ci(np.array(metrics['span_full']['goal_rate'])-metrics['reward_only_full']['goal_rate'],.05/3);gates={'H1_horizon':a['interval'][0]>=.2,'H2_signal_completion':b['interval'][0]>=.2 and floor['interval'][0]>=.8,'H3_safety_with_task':harm['interval'][0]>=.1 and cost['interval'][0]>=-.05 and floor['interval'][0]>=.8};out={'study':ID,'hypotheses':gates,'normalization_repair_supported':gates['H2_signal_completion'] and gates['H3_safety_with_task'],'horizon_effect':a,'normalization_effect':b,'completion_floor':floor,'harm_reduction':harm,'completion_cost':cost,'arm_seed_metrics':metrics,'groups':32,'scope':'Same Test4 signal family and passive trace; span scaling changes relative objective weighting.'};dump(OUT/'summary.json',out);print(gates)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['initialize','calibration','confirmation','report']);a=ap.parse_args()
 if a.phase=='initialize':initialize()
 elif a.phase=='report':report()
 else:
  for i in ([201,202] if a.phase=='calibration' else range(211,243)):run_group(a.phase,i)
if __name__=='__main__':main()
