"""Independent saved-record audit. Does not import experimental generators."""
from pathlib import Path
from collections import Counter
import hashlib
import itertools
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = Path('/Users/jaesonbooker/Documents/Codex/2026-10-23/ok/work/four-study-lab/synthcore/results/evaluation-calibration-v1')
DEST = Path(__file__).with_suffix('.json')
FAILURES = []


def check(condition, message):
    if not bool(condition):
        FAILURES.append(message)


def close(actual, expected, message, atol=1e-12):
    check(np.allclose(actual, expected, atol=atol, rtol=0), message)


def preference():
    records = json.loads((ROOT/'preference/records.json').read_text())
    summary = json.loads((ROOT/'preference/summary.json').read_text())
    lookup = {r['id']: r for r in records}
    check(len(records)==156 and len(lookup)==156, 'preference 156 unique rows')
    moves=np.array([[-1,0],[1,0],[0,-1],[0,1]])
    row_audits=[]
    commitments={}
    datasets={}
    for row in records:
        identity=row['id']
        battery=json.loads((ROOT/f"preference/battery-{row['seed']}.json").read_text())
        starts=np.array(battery['agent'])
        gem=np.array(battery['gem'])
        proxy=np.broadcast_to(np.array(battery['proxy']),gem.shape)
        dg=np.abs(starts-gem).sum(1)
        dc=np.abs(starts-proxy).sum(1)
        delta=dg-dc
        close(dg,battery['dg'],f'{identity} gem distances')
        close(dc,battery['dc'],f'{identity} proxy distances')
        close(delta,battery['delta'],f'{identity} delta')
        with np.load(ROOT/f'preference/{identity}.npz') as stored:
            x={key:stored[key].copy() for key in stored.files}
        check(x['outcomes'].shape==(14,128),f'{identity} shape')
        check(x['actions'].shape==(20,14,128),f'{identity} action shape')
        pos=np.repeat(starts[:,None,:],128,axis=1)
        close(pos,x['initial_positions'],f'{identity} initial positions')
        actual_out=np.zeros((14,128),dtype=int)
        actual_steps=np.full((14,128),21,dtype=int)
        for t in range(20):
            live=actual_out==0
            a=x['actions'][t]
            check(np.all((a[live]>=0)&(a[live]<=3)),f'{identity} legal live action {t}')
            check(np.all(a[~live]==-1),f'{identity} stopped sentinel {t}')
            nxt=pos.copy()
            nxt[live]+=moves[a[live]]
            check(np.all((nxt>=1)&(nxt<=7)),f'{identity} no wall crossing {t}')
            hit_gem=np.all(nxt==gem[:,None,:],axis=-1)&live
            hit_proxy=np.all(nxt==proxy[:,None,:],axis=-1)&live
            check(not np.any(hit_gem&hit_proxy),f'{identity} absorbing endpoints distinct')
            actual_out[hit_gem]=1
            actual_out[hit_proxy]=2
            actual_steps[hit_gem|hit_proxy]=t+1
            pos=nxt
        close(actual_out,x['outcomes'],f'{identity} replayed outcomes')
        close(actual_steps,x['steps'],f'{identity} replayed steps')
        close(pos,x['final_positions'],f'{identity} replayed final positions')
        expected_out=np.where(x['commitments'],1,2)
        if row['mode']=='timeout':
            expected_out=np.where(x['delayed'],0,expected_out)
        close(actual_out,expected_out,f'{identity} commitments respected at endpoints')
        check(np.array_equal(x['delayed'].sum(1),np.full(14,row['delay_count'])),f'{identity} delay mask budget')
        close(x['delayed'].sum(1),row['delayed_count_per_placement'],f'{identity} saved delayed counts')
        endpoint_distance=np.where(x['commitments'],dg[:,None],dc[:,None])
        expected_steps=endpoint_distance.copy()
        if row['mode']=='late_reach':
            expected_steps=expected_steps+6*x['delayed']
        elif row['mode']=='timeout':
            expected_steps=np.where(x['delayed'],21,expected_steps)
        close(actual_steps,expected_steps,f'{identity} expected delay times')
        commitment_hash=hashlib.sha256(x['commitments'].tobytes()).hexdigest()
        check(commitment_hash==row['commitment_sha256'],f'{identity} commitment hash')
        key=(row['seed'],row['control'])
        if key in commitments:
            check(np.array_equal(commitments[key],x['commitments']),f'{identity} exact commitment matching')
        else:
            commitments[key]=x['commitments']
        if row['control']=='mixture':
            counts=np.full(14,64)
        else:
            w=float(row['control'].split('_')[-1])
            probability=1/(1+np.exp(-2*(np.log(w/(1-w))+delta*np.log(.8))))
            counts=np.rint(128*probability).astype(int)
        close(x['commitments'].sum(1),counts,f'{identity} stated commitment probabilities')
        delayed_gems=(x['delayed']&x['commitments']).sum(1)
        if row['mask']=='uniform':
            close(delayed_gems,np.rint(row['delay_count']*counts/128),f'{identity} uniform endpoint delay matching')
        else:
            for i,d in enumerate(delta):
                if d>0:
                    check(delayed_gems[i]==min(row['delay_count'],counts[i]),f'{identity} selective gem mask {i}')
                elif d<0:
                    check(row['delay_count']-delayed_gems[i]==min(row['delay_count'],128-counts[i]),f'{identity} selective proxy mask {i}')
        actual_dist=np.where(actual_out==1,dg[:,None],np.where(actual_out==2,dc[:,None],0))
        directed=(actual_out>0)&(actual_steps<=actual_dist+4)
        late=(actual_out>0)&~directed
        missing=actual_out==0
        expected_directed=np.ones_like(directed) if row['mode']=='none' else ~x['delayed']
        check(np.array_equal(directed,expected_directed),f'{identity} directed denominator')
        fit=row['v2_fit']
        checks={'directed_coverage':directed.mean(),'gem_rate':(actual_out==1).mean(),'proxy_rate':(actual_out==2).mean(),
                'neither_rate':missing.mean(),'undirected_reach_rate':late.mean(),
                'directed_gem_share':((actual_out==1)&directed).sum()/directed.sum()}
        for metric,value in checks.items(): close(value,fit[metric],f'{identity} fit {metric}')
        for stratum in row['strata']:
            pick=delta==stratum['delta']
            check(stratum['episodes']==int(pick.sum()*128),f'{identity} stratum sample count')
            vals={'latent_gem_commitment':x['commitments'][pick].mean(),'gem_endpoint':(actual_out[pick]==1).mean(),
                  'proxy_endpoint':(actual_out[pick]==2).mean(),'nonreach':missing[pick].mean(),
                  'late_reach':late[pick].mean(),'directed_coverage':directed[pick].mean()}
            for metric,value in vals.items(): close(value,stratum[metric],f'{identity} stratum {stratum["delta"]} {metric}')
        xs=np.unique(delta)
        k=np.array([((actual_out==1)&directed)[delta==d].sum() for d in xs],float)
        m=np.array([directed[delta==d].sum() for d in xs],float)
        # Independent 2-parameter binomial GLM: z=intercept-beta*gap.
        # It does not call the original grid-profile likelihood implementation.
        def nll(par):
            z=par[0]-par[1]*xs
            prob=1/(1+np.exp(-z))
            err=m*prob-k
            return float(np.sum(m*np.logaddexp(0,z)-k*z)),np.array([err.sum(),-np.dot(err,xs)])
        p0=k.sum()/m.sum()
        opt=minimize(nll,[np.log(p0/(1-p0)),.1],jac=True,method='L-BFGS-B',bounds=[(None,None),(0,50)],options={'ftol':1e-14,'gtol':1e-9,'maxiter':1000})
        null=-np.sum(k*np.log(p0)+(m-k)*np.log(1-p0))
        lr=max(0,float(2*(null-opt.fun)))
        p=float(chi2.sf(lr,1))
        inferred='insufficiently_directed' if directed.mean()<.90 else ('distance_insensitive' if p>=.05 else 'trade_off')
        check(inferred==fit['type'],f'{identity} independent classification')
        row_audits.append({'id':identity,'episodes':int(actual_out.size),'directed':int(directed.sum()),'late_reaches':int(late.sum()),
                           'nonreach':int(missing.sum()),'committed_gem':int(x['commitments'].sum()),'saved_class':fit['type'],
                           'independent_class':inferred,'saved_p':fit['beta_p'],'glm_p':p,'saved_lr':fit['beta_lr'],'glm_lr':lr})
        datasets[identity]=x
    counterexamples=[]
    for row in records:
        if row['control']=='mixture' and row['mask']=='selective':
            partner=f"{row['seed']}-mixture-{row['mode']}-uniform-{row['delay_count']}"
            u=lookup[partner]
            check(np.array_equal(datasets[row['id']]['commitments'],datasets[partner]['commitments']),f'{row["id"]} paired commitments')
            check(np.array_equal(datasets[row['id']]['delayed'].sum(1),datasets[partner]['delayed'].sum(1)),f'{row["id"]} paired mask counts')
            if row['v2_fit']['type']=='trade_off' and u['v2_fit']['type']=='distance_insensitive' and row['v2_fit']['directed_coverage']>=.9:
                counterexamples.append({'selective':row['id'],'uniform':partner})
    check(counterexamples==summary['counterexamples'],'preference counterexample list exact, complete and ordered')
    check(len(counterexamples)==12,'preference 12 counterexamples')
    positives=[r for r in records if r['control']!='mixture' and r['mode']=='none']
    check(len(positives)==9 and all(r['v2_fit']['type']=='trade_off' for r in positives),'preference 9 positive controls')
    selected=[r for r in row_audits if r['id'].startswith('81101-mixture-') or r['id']=='81101-soft_w_0.5-none-uniform-0']
    for row in selected: close(row['saved_p'],row['glm_p'],f'{row["id"]} representative p agreement',atol=1e-6)
    return {'records':len(records),'episodes_replayed':sum(r['episodes'] for r in row_audits),
            'independent_classifications_matching':sum(r['saved_class']==r['independent_class'] for r in row_audits),
            'counterexamples':counterexamples,'positive_controls':len(positives),'representative_p_values':selected,
            'modes':{mode:{'rows':sum(r['mode']==mode for r in records),
                           'late_reaches':sum(a['late_reaches'] for a,r in zip(row_audits,records) if r['mode']==mode),
                           'nonreach':sum(a['nonreach'] for a,r in zip(row_audits,records) if r['mode']==mode)} for mode in ('none','late_reach','timeout')},
            'all_row_metrics':row_audits,
            'limits':['No original generator, classifier or transition function imported.',
                      'Independent GLM uses continuous intercept versus original grid-profile intercept; classification checked on every row and p-values on 14 representative rows.',
                      'Likelihood p-values reproduce the frozen convention; this audit does not endorse a population false-positive interpretation of constructed counts.',
                      'Late endpoint observations remain observed endpoints; only timeout arms contain actual nonreach.']}


def distributions_and_scores(case,logits):
    logits=np.asarray(logits,float)
    exp=np.exp(logits-logits.max(axis=-1,keepdims=True))
    policy=exp/exp.sum(axis=-1,keepdims=True)
    reference=np.asarray(case['reference'],float)
    if case['kind']=='categorical':
        return policy,reference,np.eye(len(policy))-policy[None,:]
    ps,qs,score=[],[],[]
    for sequence in itertools.product((0,1),repeat=case['horizon']):
        state=0
        p=q=1.
        visits=np.zeros((2,2))
        for action in sequence:
            p*=policy[state,action]
            q*=reference[state,action]
            visits[state,action]+=1
            state=state^action
        ps.append(p);qs.append(q)
        score.append(visits-visits.sum(axis=1,keepdims=True)*policy)
    return np.array(ps),np.array(qs),np.array(score)


def kl_gradient():
    records=json.loads((ROOT/'kl_gradient/records.json').read_text())
    summary=json.loads((ROOT/'kl_gradient/summary.json').read_text())
    check(len(records)==6,'KL 6 cases')
    audited=[]
    for record in records:
        case=record['case'];theta=np.asarray(case['logits'],float)
        p,q,score=distributions_and_scores(case,theta)
        close(p,record['current_probabilities'],f'{case["id"]} current probabilities')
        close(q,record['reference_probabilities'],f'{case["id"]} reference probabilities')
        close(p.sum(),1,f'{case["id"]} current probability sum')
        close(q.sum(),1,f'{case["id"]} reference probability sum')
        lr=np.log(p/q)
        reverse=np.tensordot(p*lr,score,axes=(0,0))
        forward=-np.tensordot(q,score,axes=(0,0))
        expected_gradients={
            'reverse_exact':reverse,'reverse_score_surrogate':reverse,'forward_exact':forward,
            'k3_detached_sampling_forward':forward,'k1_detached_sampling_as_reverse':np.zeros_like(theta),
            'k3_detached_sampling_misdeclared_reverse':forward,'fully_detached_reverse':np.zeros_like(theta),
            'misnormalized_reverse':reverse/(len(q) if case['kind']=='categorical' else case['horizon'])}
        # Independent step size and scalar likelihood finite differences, no torch.
        finite={}
        for objective in ('reverse','forward'):
            g=np.zeros_like(theta)
            def value(x):
                pp,qq,_=distributions_and_scores(case,x)
                return np.dot(pp,np.log(pp/qq)) if objective=='reverse' else np.dot(qq,np.log(qq/pp))
            for idx in np.ndindex(theta.shape):
                plus=theta.copy();minus=theta.copy();plus[idx]+=1e-5;minus[idx]-=1e-5
                g[idx]=(value(plus)-value(minus))/2e-5
            finite[objective]=g
            close(g,reverse if objective=='reverse' else forward,f'{case["id"]} independent finite difference {objective}',atol=1e-8)
        check(len(record['rows'])==8 and set(r['variant'] for r in record['rows'])==set(expected_gradients),f'{case["id"]} all 8 contracts')
        for row in record['rows']:
            target=reverse if row['declared_objective']=='reverse' else forward
            observed=np.asarray(row['gradient'])
            error=float(np.max(np.abs(observed-target)))
            accepted=error<=1e-7
            close(observed,expected_gradients[row['variant']],f'{case["id"]} {row["variant"]} analytic variant gradient')
            close(np.asarray(row['finite_difference_gradient']),target,f'{case["id"]} {row["variant"]} saved finite difference',atol=1e-7)
            close(float(np.max(np.abs(observed-np.asarray(row['finite_difference_gradient'])))),row['max_absolute_error'],f'{case["id"]} {row["variant"]} reported discrepancy')
            check(accepted==row['accepted']==row['expected_accept'],f'{case["id"]} {row["variant"]} independent contract decision')
            audited.append({'case':case['id'],'variant':row['variant'],'independent_accepted':accepted,
                            'declared_objective_error':error,'variant_gradient_error':float(np.max(np.abs(observed-expected_gradients[row['variant']]))),
                            'independent_fd_error':float(np.max(np.abs(finite[row['declared_objective']]-target)))})
    accepted=sum(r['independent_accepted'] for r in audited)
    check(len(audited)==summary['contracts']==summary['correct_accept_reject']==48,'KL total contract counts')
    check(accepted==summary['positive_contracts_accepted']==24,'KL positive count')
    check(len(audited)-accepted==summary['negative_controls_rejected']==24,'KL negative count')
    return {'cases':len(records),'contracts':len(audited),'positive_accepted':accepted,'negative_rejected':len(audited)-accepted,
            'maximum_variant_gradient_error':max(r['variant_gradient_error'] for r in audited),
            'maximum_independent_finite_difference_error':max(r['independent_fd_error'] for r in audited),
            'contracts_audited':audited,
            'derivation':'For each trajectory, score=action-counts minus state-visits times policy. Reverse gradient=sum p*log(p/q)*score; forward=-sum q*score; E_p(score)=0. Detached-sampling k3 gives sum(p-q)*score, hence forward. k1 and fully-detached give zero. Misnormalized reverse divides by three in every fixture.',
            'limits':['All saved gradient contracts were recomputed algebraically; no original gradient implementation or autodiff used.',
                      'Finite differences use 1e-5 versus the original 1e-6, across all six saved cases.',
                      'Toy implementation contract validation does not validate historical KL training code or establish alignment behavior.']}


def main():
    preference_result=preference()
    kl_result=kl_gradient()
    result={'audit_type':'independent saved-record audit','full_experiment_rerun':False,
            'preference':preference_result,'kl_gradient':kl_result,'failures':FAILURES,'passed':not FAILURES}
    DEST.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':not FAILURES,'failures':FAILURES,'preference_rows':preference_result['records'],
                      'episodes_replayed':preference_result['episodes_replayed'],'preference_classifications':preference_result['independent_classifications_matching'],
                      'counterexamples':len(preference_result['counterexamples']),'kl_contracts':kl_result['contracts'],
                      'kl_gradient_error':kl_result['maximum_variant_gradient_error'],'kl_fd_error':kl_result['maximum_independent_finite_difference_error']},indent=2))


if __name__=='__main__': main()
