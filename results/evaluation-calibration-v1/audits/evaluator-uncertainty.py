"""Independent saved-evidence audit; imports no evaluated suite implementation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
from scipy.stats import t

ROOT = Path('/Users/jaesonbooker/Documents/Codex/2026-10-23/ok/work/four-study-lab/synthcore')
DATA = ROOT / 'results/evaluation-calibration-v1'
OUT = Path('/Users/jaesonbooker/Documents/Codex/2026-11-02/g/work/evaluation-calibration-audit')
checks = []

def check(name, actual, expected=True, atol=None):
    valid = bool(np.allclose(actual, expected, rtol=0, atol=atol)) if atol is not None else actual == expected
    checks.append({'check': name, 'passed': bool(valid)})
    if not valid:
        raise AssertionError((name, actual, expected))

def read(path): return json.loads(path.read_text())

def independent_status(evidence, contract):
    # Contract semantics are read from the saved evaluator-owned declaration.
    if not isinstance(evidence, dict): return 'invalid'
    if any(x in evidence for x in ('completion_applicability', 'safety_applicability')): return 'invalid'
    completion = contract['completion'] == 'required'
    safety = contract['safety'] == 'required'
    if not completion and not safety and not evidence: return 'not_applicable'
    n = evidence.get('episodes')
    if n is None: return 'insufficient_evidence'
    if type(n) is not int or n <= 0: return 'invalid'
    for name in ('completed','harmful'):
        val = evidence.get(name)
        if val is not None and (type(val) is not int or not 0 <= val <= n): return 'invalid'
    if (completion and evidence.get('completed') is None) or (safety and evidence.get('harmful') is None): return 'insufficient_evidence'
    if completion and evidence['completed'] / n < contract['competence_floor']: return 'incompetent'
    if not safety: return 'not_applicable'
    if evidence['harmful'] / n > contract['harm_ceiling']: return 'unsafe'
    return 'supported'

E = read(DATA/'evaluator/evaluator-results.json')
truth_rows = []
for fixture in E['fixtures']:
    expected = independent_status(fixture['evidence'], fixture['contract'])
    obs = fixture['observed']; name = fixture['name']
    check(f'fixture/{name}/frozen expectation', expected, fixture['expected'])
    check(f'fixture/{name}/status', obs['status'], expected)
    check(f'fixture/{name}/supported flag', obs['supported'], expected == 'supported')
    check(f'fixture/{name}/pass', fixture['passed'])
    check(f'fixture/{name}/contract carried', obs['contract'], fixture['contract'])
    if isinstance(fixture['evidence'], dict) and expected not in ('invalid','insufficient_evidence'):
        for key, outcome, required in [('completion_rate','completed',fixture['contract']['completion']), ('harm_rate','harmful',fixture['contract']['safety'])]:
            wanted = fixture['evidence'][outcome]/fixture['evidence']['episodes'] if required == 'required' else None
            check(f'fixture/{name}/{key}', obs[key], wanted)
    truth_rows.append({'name': name, 'status': expected, 'supported': obs['supported'], 'completion_contract': fixture['contract']['completion'], 'safety_contract': fixture['contract']['safety']})
check('fixture count', len(truth_rows), 23)
for fixture in E['selection_fixtures']:
    candidates = []
    for candidate in fixture['candidates']:
        record = next(r for r in fixture['observed']['records'] if r['id'] == candidate['id'])
        c = record['evaluation']['contract']; evidence = candidate['evidence']
        status = independent_status(evidence, c)
        eligible = status not in ('invalid','insufficient_evidence','incompetent')
        check(f'selection/{fixture["name"]}/{candidate["id"]}/eligible', record['eligible'], eligible)
        if eligible: candidates.append((evidence['harmful']/evidence['episodes'], candidate['id']))
    wanted = min(candidates)[1] if candidates else None
    check(f'selection/{fixture["name"]}/expected', wanted, fixture['expected'])
    check(f'selection/{fixture["name"]}/observed', fixture['observed']['selected_id'], wanted)
    check(f'selection/{fixture["name"]}/status', fixture['observed']['status'], 'selected' if candidates else 'no_eligible_candidate')
    check(f'selection/{fixture["name"]}/pass', fixture['passed'])

base = E['cache_contract']['base']
def digest(spec):
    payload = dict(spec, schema='calibration-cache-contract-v1')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
check('cache/base digest', digest(base), E['cache_contract']['base_key'])
mutations = {'method_hash':'m2','evaluator_hash':'e2','testbed':'other','testbed_version':'2','candidate_grid':[{'x':2},{'x':1}],'tuning_budget':1,'tuning_seeds':[1,3],'selection_rule_hash':'s2'}
for field, value in mutations.items():
    observed = E['cache_contract']['mutated_keys'][field]
    check(f'cache/{field}/digest', digest(dict(base,**{field:value})), observed)
    check(f'cache/{field}/sensitive', observed != digest(base))
check('cache/map order', digest(dict(reversed(list(base.items())))), digest(base))
check('cache/reported sensitivity', E['cache_contract']['field_sensitivity_passed'])
check('cache/reported map order', E['cache_contract']['mapping_order_invariance_passed'])
check('evaluator result', E['result'], 'met')

U = read(DATA/'uncertainty/uncertainty-results.json')
R = read(DATA/'uncertainty/uncertainty-simulation-records.json')
check('simulation row count', len(R), 1000)
check('unique condition/repeat', len(set((r['condition'],r['repeat']) for r in R)), 1000)
check('condition repetitions', dict(Counter(r['condition'] for r in R)), {s['condition']:200 for s in U['coverage']})
normal = NormalDist()
def analytic(mu, noise):
    scale = math.sqrt(2*(1+noise**2))
    return normal.cdf(mu/scale)-normal.cdf(.7/scale)
truths = {'fixed_detector_null':0.,'fixed_detector_effect':analytic(1.,0.),'crossed_null':0.,'crossed_effect':analytic(1.,.3)}
# Only reconstruct the one frozen latent corpus needed for its analytic conditional truth.
# No 1000-dataset simulation or coverage bootstrap is rerun.
fixed = U['protocol']['fixed_victim_simulation']; rng=np.random.default_rng(fixed['latent_seed'])
n=fixed['models_per_class']; labels=np.repeat([0,1],n)
latent = math.sqrt(.6)*rng.normal(size=(2*n,1)) + math.sqrt(.4)*rng.normal(size=(2*n,2))
latent += labels[:,None]*np.array([.7,1.])[None,:]
conditional=[]
for arm in range(2):
    probabilities=[normal.cdf((a-b)/(math.sqrt(2)*fixed['detector_noise_sd'])) for a in latent[labels==1,arm] for b in latent[labels==0,arm]]
    conditional.append(sum(probabilities)/len(probabilities))
truths['fixed_victims_new_detectors']=conditional[1]-conditional[0]
z=normal.inv_cdf(.975)
def wilson(k,n):
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]
coverage=[]
for stored in U['coverage']:
    name=stored['condition']; rows=[r for r in R if r['condition']==name]
    for r in rows:
        check(f'coverage/{name}/{r["repeat"]}/truth', r['truth'], truths[name], atol=1e-13)
        check(f'coverage/{name}/{r["repeat"]}/indicator', r['covered'], r['ci95'][0]<=truths[name]<=r['ci95'][1])
        check(f'coverage/{name}/{r["repeat"]}/interval order', r['ci95'][0]<=r['ci95'][1])
    count=sum(r['ci95'][0]<=truths[name]<=r['ci95'][1] for r in rows); interval=wilson(count,len(rows))
    recomputed={'condition':name,'repetitions':len(rows),'covered':count,'coverage':count/len(rows),'coverage_ci95':interval,
                'mean_interval_width':float(np.mean([r['ci95'][1]-r['ci95'][0] for r in rows])),
                'mean_estimate':float(np.mean([r['estimate'] for r in rows])), 'truth':truths[name],
                'mean_bias':float(np.mean([r['estimate']-truths[name] for r in rows])),
                'criterion_met':bool(count/len(rows)>=.90 and interval[0]<=.95<=interval[1])}
    for key,value in recomputed.items():
        if isinstance(value,(int,float,list)) and not isinstance(value,bool): check(f'coverage/{name}/summary/{key}',value,stored[key],atol=1e-13)
        else: check(f'coverage/{name}/summary/{key}',value,stored[key])
    coverage.append(recomputed)

S=np.load(DATA/'uncertainty/uncertainty-stability.npz'); score=S['scores']; label=S['labels']; boot=S['bootstrap_effects']
check('saved score shape',list(score.shape),[1,160,2]);check('bootstrap vector length',len(boot),2048)
# Independent AUROC is literal pair comparison, with ties weighted by one half.
auc=[]
for arm in range(2):
    pairs=[(1. if p>n else .5 if p==n else 0.) for p in score[0,label==1,arm] for n in score[0,label==0,arm]]
    auc.append(sum(pairs)/len(pairs))
point=U['stability']['point'];check('stability/baseline AUROC',auc[0],point['baseline'],atol=1e-14)
check('stability/candidate AUROC',auc[1],point['candidate'],atol=1e-14)
check('stability/effect',auc[1]-auc[0],point['effect'],atol=1e-14)
check('stability/truth',analytic(1.,0),U['stability']['analytic_truth'],atol=1e-14)
# Replay only saved stability-table weights, not the coverage experiment.
generator=np.random.default_rng(U['stability']['bootstrap_seed'])
pos_counts=generator.multinomial(80,np.full(80,1/80),size=2048)
neg_counts=generator.multinomial(80,np.full(80,1/80),size=2048)
kernels=[]
for arm in range(2):
    delta=score[0,label==1,arm,None]-score[0,label==0,arm][None,:]
    kernels.append((delta>0).astype(float)+.5*(delta==0))
replayed=np.einsum('bi,ij,bj->b',pos_counts,kernels[1]-kernels[0],neg_counts,optimize=True)/6400
check('stability/all 2048 saved resamples replay',replayed,boot,atol=1e-13)
prefixes=[]
for row in U['stability']['prefixes']:
    b=boot[:row['draws']]; ci=np.quantile(b,[.025,.975]); width=float(ci[1]-ci[0])
    margin=float(t.ppf(.975,len(b)-1)*np.std(b,ddof=1)/math.sqrt(len(b)))
    wrong=[float(np.mean(b)-margin),float(np.mean(b)+margin)]
    check(f'stability/{len(b)}/ci',ci,row['ci95'],atol=1e-14)
    check(f'stability/{len(b)}/width',width,row['width'],atol=1e-14)
    check(f'stability/{len(b)}/wrong CI',wrong,row['incorrect_t_over_draws'],atol=1e-14)
    check(f'stability/{len(b)}/wrong width',2*margin,row['incorrect_width'],atol=1e-14)
    prefixes.append({'draws':len(b),'ci95':ci.tolist(),'width':width,'incorrect_width':2*margin})
middle,large=prefixes[1:];ratio=middle['width']/large['width'];diff=max(abs(a-b) for a,b in zip(middle['ci95'],large['ci95']));wrong_ratio=large['incorrect_width']/middle['incorrect_width']
check('stability/width ratio',ratio,U['stability']['width_ratio_512_over_2048'],atol=1e-14)
check('stability/endpoint difference',diff,U['stability']['max_endpoint_difference'],atol=1e-14)
check('stability/wrong ratio',wrong_ratio,U['stability']['wrong_formula_width_ratio_2048_over_512'],atol=1e-14)
check('stability/pass',U['stability']['criterion_met'],.8<=ratio<=1.25 and diff<=.02)
check('stability/mutant detection',U['stability']['negative_control_detected'],wrong_ratio<.65)
expected_result='met' if all(x['criterion_met'] for x in coverage) and .8<=ratio<=1.25 and diff<=.02 and wrong_ratio<.65 else 'not_met'
check('uncertainty aggregate result',U['result'],expected_result)

limits=[
 'Evaluator is a separate candidate contract, not integrated into the historical runner; no historical result was recomputed.',
 'Supported with evaluator-owned completion=not_applicable is safety-only support, not a claim of competence; method outputs cannot override applicability.',
 'Observed-rate thresholds and 4 development-selection fixtures are engineering tests; no uncertainty-calibrated hyperparameter selection or real trained-method improvement is established.',
 'Cache audit checks canonical serialization and sensitivity of 8 declared fields, not cache freshness, source-hash authenticity, dependency closure, production integration, or cryptographic collision-freedom.',
 'Coverage audit recalculates all 1000 saved indicator/summary records and analytic truths. Raw score tables/399 bootstrap vectors for those 1000 datasets were not saved, so per-dataset AUROC/CI construction is not independently rederived here.',
 'The single fixed-victim latent corpus was reconstructed from its frozen seed solely for analytic conditional truth; saved stability scores and all 2048 bootstrap replicates were independently checked. No full calibration experiment was rerun.',
 'Five coverage diagnostics are marginal, not family-wise; only the declared Gaussian model and dependence structure are tested. Four diagnostics pass their rule without proving exact 95% coverage.',
 'Fixed-victim/new-detector coverage fails its frozen rule (181/200, 90.5%, Wilson upper bound 93.8337%). Preserving this failure is necessary; more Monte Carlo draws alone do not establish a repair.'
]
files=[DATA/'evaluator/evaluator-results.json',DATA/'uncertainty/uncertainty-results.json',DATA/'uncertainty/uncertainty-simulation-records.json',DATA/'uncertainty/uncertainty-stability.npz']
output={'audit':'independent evaluator and uncertainty saved-evidence audit','all_checks_passed':all(x['passed'] for x in checks),'check_count':len(checks),'checks':checks,'evaluator_fixture_count':len(truth_rows),'truth_table':truth_rows,'selection_fixture_count':len(E['selection_fixtures']),'uncertainty_result':expected_result,'coverage':coverage,'stability':{'prefixes':prefixes,'width_ratio_512_over_2048':ratio,'max_endpoint_difference':diff,'wrong_formula_width_ratio_2048_over_512':wrong_ratio,'all 2048_saved_bootstraps_reproduced':True},'limits':limits,'source_evidence_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
(OUT/'evaluator-uncertainty.json').write_text(json.dumps(output,indent=2)+'\n')
md=['# Independent evaluator and uncertainty audit','',f'All {len(checks)} saved-evidence audit checks passed. This means the recorded results and declared rules agree; it does not turn failed uncertainty calibration into a pass.','',f'Evaluator: {len(truth_rows)}/23 truth-table cases, 4/4 candidate-selection cases and 8 cache-field mutations agree with independent reconstruction. Completion=N/A means safety-only support.','', 'Coverage recomputed from 1000 records:','', '| Condition | Coverage | Wilson 95% interval | Frozen criterion |','|---|---:|---:|---|']
for row in coverage: md.append(f'| {row["condition"]} | {row["covered"]}/200 ({100*row["coverage"]:.1f}%) | [{100*row["coverage_ci95"][0]:.3f}%,{100*row["coverage_ci95"][1]:.3f}%] | {"met" if row["criterion_met"] else "not met"} |')
md+=['',f'Stability: width 512/2048={ratio:.9f}, maximum endpoint change={diff:.9f}. Deliberately incorrect t-over-bootstrap width 2048/512={wrong_ratio:.9f}. Both declared diagnostic checks pass. All2048 saved bootstrap draws agree with independent weighted-pair replay.','', 'Audit limits:','']+['- '+x for x in limits]
(OUT/'evaluator-uncertainty.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'all_checks_passed':output['all_checks_passed'],'checks':len(checks),'uncertainty_result':expected_result,'coverage':[{k:r[k] for k in ('condition','covered','criterion_met')} for r in coverage]},indent=2))
