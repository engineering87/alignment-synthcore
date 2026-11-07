"""Independent post-run reconstruction; imports no experiment or testbed code."""
from collections import deque
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import numpy as np

BASE = Path('/Users/jaesonbooker/Documents/Codex/2026-10-23/ok/work/four-study-lab/synthcore/results/evaluation-calibration-v1')
OUT = Path(__file__).with_suffix('.json')


def read(path): return json.loads(path.read_text())
def near(a, b, tol=1e-12):
    assert abs(float(a)-float(b)) <= tol, (a, b)


def terminal_audit():
    root = BASE / 'terminal'
    worlds = {w['name']: w for w in read(root/'terminal-worlds.json')}
    rows = read(root/'terminal-records.json')
    differences = read(root/'terminal-penalty-differences.json')
    summary = read(root/'terminal-summary.json')
    penalty_tables = {}
    events = 0
    for d in differences:
        a, b = np.array(d['terminal_penalty']), np.array(d['absorbing_penalty'])
        near(np.max(np.abs(a-b)), d['max_absolute_penalty_difference'])
        for enc in ('terminal', 'absorbing'):
            penalty_tables[(d['fixture'], d['penalty'], d['convention'], enc)] = np.array(d[enc+'_penalty'])
    for w in worlds.values():
        for s, goal in enumerate(w['goals']):
            if goal:
                assert w['transitions'][s] == [s]*5
                for key in ('reward', 'completion', 'harm', 'work'):
                    assert all(x == 0 for x in w[key][s])
    for row in rows:
        w = worlds[row['fixture']]
        state = w['start']; reward = discounted = 0.; goals = harms = work = 0
        log = []
        for t in range(8):
            a = row['policy'][t][state]; ns = w['transitions'][state][a]
            done = row['encoding'] == 'terminal' and w['goals'][ns]
            expected = dict(t=t, state=state, action=a, next_state=ns,
                            public_reward=w['reward'][state][a], task_completed=w['completion'][state][a],
                            harm=w['harm'][state][a], work=w['work'][state][a],
                            terminated=bool(done), truncated=t == 7 and not done)
            log.append(expected)
            reward += w['reward'][state][a]; discounted += .95**t*w['reward'][state][a]
            goals += w['completion'][state][a]; harms += w['harm'][state][a]; work += w['work'][state][a]
            state = ns
            if done: break
        assert log == row['events']
        metrics = dict(task_completed=goals > 0, harm_events=harms, safe_completion=goals > 0 and harms == 0,
                       work_events=work, public_return=reward, discounted_public_return=discounted,
                       transitions_recorded=len(log), terminated=bool(done), truncated=not done, final_state=state)
        assert metrics == row['metrics']
        events += len(log)
        if 'penalty' in row:
            P = penalty_tables[(row['fixture'], row['penalty'], row['convention'], row['encoding'])]
            # Independent recursive Bellman evaluation of all states and remaining horizons.
            @lru_cache(None)
            def optimal(t, s):
                if t == 8 or (row['encoding'] == 'terminal' and w['goals'][s]): return 0.
                return max(w['reward'][s][a]-row['lambda']*P[s,a]+.95*optimal(t+1,w['transitions'][s][a]) for a in range(5))
            near(optimal(0, w['start']), row['planned_penalized_value'])
            for t in range(8):
                for s in range(len(w['states'])):
                    if row['encoding'] == 'terminal' and w['goals'][s]: continue
                    a = row['policy'][t][s]
                    near(w['reward'][s][a]-row['lambda']*P[s,a]+.95*optimal(t+1,w['transitions'][s][a]), optimal(t,s))
    methods = [r for r in rows if 'penalty' in r]
    hazard = [r for r in methods if r['fixture'] == 'avoidable_harm_on_completion']
    failures = [r for r in methods if not r['metrics']['task_completed']]
    changes = []
    for row in methods:
        if row['encoding'] != 'terminal': continue
        other = next(x for x in methods if x['encoding']=='absorbing' and all(x[k]==row[k] for k in ('fixture','penalty','convention','lambda')))
        keys = ('task_completed','harm_events','safe_completion','public_return','work_events')
        if any(row['metrics'][k] != other['metrics'][k] for k in keys): changes.append([row['fixture'],row['penalty'],row['convention'],row['lambda']])
    assert len(methods) == summary['method_cells'] == 78
    assert len(rows)-len(methods) == summary['control_cells'] == 14
    assert len(failures) == len(summary['completion_failures']) == 4
    hashes = read(root/'terminal-sha256.json')
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest for name,digest in hashes.items())
    return dict(passed=True, records=len(rows), reconstructed_events=events, method_cells=len(methods),
                method_harm_fixture_cells=len(hazard), harmful_cells=sum(r['metrics']['harm_events']>0 for r in hazard),
                completion_failures=len(failures), failure_description='Movement penalty lambda 1 on necessary harmless work, both encodings and both conventions.',
                task_harm_outcome_changes_between_encodings=changes,
                native_penalty_differences=[dict(fixture=d['fixture'],penalty=d['penalty'],max_absolute=d['max_absolute_penalty_difference']) for d in differences if d['convention']=='native_terminal_metadata' and d['max_absolute_penalty_difference']>1e-12],
                every_policy_structural_equivalence_verified=True, saved_hashes_verified=len(hashes),
                limits='Complete graph, three purpose-built fixtures, no learned policies. AUP changes auxiliary continuation as well as exemption; canonical invariance is constructed. No safety repair demonstrated.')


MOVES = [(-1,0),(1,0),(0,-1),(0,1)]
def trajectory(level, start, tape):
    p = tuple(start); goal = tuple(level['cheese']); done = False; count = 0; result = 0.
    for a in tape:
        if a == -1:
            assert done
            continue
        assert not done and 0 <= a < 4
        dr,dc = MOVES[a]; new = (p[0]+dr,p[1]+dc)
        if not level['walls'][new[0]][new[1]]: p = new
        count += 1
        if p == goal: result = 1-count/38; done = True
    assert len(tape) == 19
    return result, count


def distance(level, start):
    queue = deque([(tuple(start),0)]); seen = {tuple(start)}
    while queue:
        pos,d = queue.popleft()
        if pos == tuple(level['cheese']): return d
        for dr,dc in MOVES:
            n=(pos[0]+dr,pos[1]+dc)
            if not level['walls'][n[0]][n[1]] and n not in seen:
                seen.add(n); queue.append((n,d+1))
    return -1


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); out = np.empty(len(values))
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        for i in order[start:end]: out[i]=(start+end-1)/2
        start=end
    return out


def replay_audit():
    root=BASE/'replay'; rows=read(root/'records.json'); summary=read(root/'summary.json')
    reconstructed=[]; assessment_episodes=holdout_replays=0; reference_entries=0
    tie_rows=[]
    for row in rows:
        scores=[]; targets=[]; assessment_steps=holdout_steps=0; repetitions=[]; ages=[]
        keys=np.random.default_rng(row['generation_seed']+(90000 if row['family']=='lamp' else 80000)).random(24)
        assert keys.tolist()==row['tie_keys']
        for i,level in enumerate(row['levels']):
            receipt=row['receipts'][i]; hist=receipt['receipts']; returns=[]; starts=[]
            for j,item in enumerate(hist):
                ret,steps=trajectory(level,item['start'],row['assessment_actions'][i][j]); near(ret,item['return'])
                returns.append(ret); starts.append(tuple(item['start'])); assessment_steps+=steps
                assessment_episodes+=1
                same=[k for k in range(j+1) if starts[k]==starts[j]]
                assert item['repeat_count']==len(same) and item['first_seen']==(len(same)==1)
                assert item['reference_updated']==(len(same)==1 or ret>max(returns[k] for k in same[:-1]))
                deficits=[]
                for k in range(max(0,j-2),j+1):
                    ref=max(returns[z] for z in range(j+1) if starts[z]==starts[k])
                    deficits.append(ref-returns[k])
                near(sum(deficits)/len(deficits),item['score']); reference_entries+=1
            assert len(hist)==12
            near(hist[-1]['score'],receipt['score']); scores.append(receipt['score'])
            unique=set(starts); assert len(unique)==receipt['unique_starts']
            repeat=1-len(unique)/12; near(repeat,receipt['repeated_fraction']); repetitions.append(repeat)
            reference_age=[]; first_age=[]
            for s in unique:
                ids=[k for k in range(12) if starts[k]==s]
                maximum=max(returns[k] for k in ids)
                # Strict updates mean the first occurrence of the final maximum sets its age.
                updated=min(k for k in ids if returns[k]==maximum)
                reference_age.append(11-updated); first_age.append(11-min(ids))
            age=sum(reference_age)/len(reference_age); near(age,receipt['reference_age_mean']); ages.append(age)
            near(sum(first_age)/len(first_age),receipt['first_seen_age_mean']); assert receipt['evictions']==0
            if row['arm']=='repeated': assert len(unique)==1
            differences=[]
            for j,start in enumerate(row['holdout_starts'][i]):
                ret,steps=trajectory(level,start,row['holdout_actions'][i][j]); near(ret,row['holdout_returns'][i][j])
                d=distance(level,start); optimum=1-d/38 if 0<d<=19 else 0.
                assert optimum+1e-12>=ret
                differences.append(optimum-ret); holdout_steps+=steps; holdout_replays+=1
            assert len(differences)==16
            targets.append(sum(differences)/16)
        assert assessment_steps==row['assessment_steps'] and holdout_steps==row['shared_holdout_steps']
        np.testing.assert_allclose(scores,row['scores'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(targets,row['private_diagnostic_regret'],rtol=0,atol=1e-12)
        selected=sorted(range(24),key=lambda i:(-scores[i],keys[i]))[:6]
        assert selected==row['selected_level_ids']
        constant=np.std(scores)<1e-12 or np.std(targets)<1e-12
        correlation=0 if constant else float(np.corrcoef(ranks(scores),ranks(targets))[0,1])
        enrichment=sum(targets[i] for i in selected)/6-sum(targets)/24
        checks={'spearman':correlation,'top_quarter_regret_enrichment':enrichment,'uniform_expected_regret':sum(targets)/24,
                'nonzero_score_fraction':sum(x>1e-12 for x in scores)/24,'score_variance':float(np.var(scores)),
                'mean_repeated_fraction':sum(repetitions)/24,'mean_reference_age':sum(ages)/24}
        for key,value in checks.items(): near(value,row[key])
        assert bool(constant)==row['constant_ranking']
        reconstructed.append(dict(family=row['family'],generation_seed=row['generation_seed'],arm=row['arm'],**checks))
        tied_cutoff=[i for i in range(24) if scores[i]==scores[selected[-1]]]
        tie_rows.append(dict(family=row['family'],seed=row['generation_seed'],arm=row['arm'],cutoff_tie_size=len(tied_cutoff),
                             selected_defaults=sum(i<12 for i in selected),selected_children=sum(i>=12 for i in selected)))
    gates={}
    for family in ('corner','lamp'):
        these=[r for r in reconstructed if r['family']==family and r['arm']=='repeated']
        metrics={k:sum(r[k] for r in these)/4 for k in ('spearman','top_quarter_regret_enrichment','nonzero_score_fraction')}
        metrics['advance']=metrics['spearman']>=.2 and metrics['top_quarter_regret_enrichment']>=.05 and metrics['nonzero_score_fraction']>=.25
        for k,v in metrics.items(): near(v,summary['family_pilot_gates'][family][k])
        gates[family]=metrics
    assert all(g['advance'] for g in gates.values())==summary['advance_to_learning_gain_study']
    paired=[]
    for r in rows:
        if r['arm']!='repeated': continue
        q=next(x for x in rows if x['family']==r['family'] and x['generation_seed']==r['generation_seed'] and x['arm']=='random')
        for k in ('levels','holdout_starts','holdout_actions','holdout_returns','private_diagnostic_regret','tie_keys'): assert r[k]==q[k]
        for i in range(24): assert r['receipts'][i]['receipts'][0]['start']==q['receipts'][i]['receipts'][0]['start']
        paired.append(dict(family=r['family'],seed=r['generation_seed'],**{k:r[k]-q[k] for k in ('spearman','top_quarter_regret_enrichment','assessment_steps')}))
    # Conditional input-model provenance: no training is reproduced or claimed here.
    for model in summary['input_models']:
        assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()==model['sha256']
    geometries=[hashlib.sha256(json.dumps(level,sort_keys=True).encode()).hexdigest()
                for row in rows if row['arm']=='repeated' for level in row['levels']]
    return dict(passed=True,records=len(rows),candidate_level_slots=len(geometries),distinct_saved_geometries=len(set(geometries)),assessment_episodes_replayed=assessment_episodes,
                reference_entries_checked=reference_entries,holdout_replays_including_duplicate_arm_records=holdout_replays,
                unique_holdout_episodes=holdout_replays//2,paired_candidate_and_holdout_identity_verified=True,
                independent_seeded_tie_keys_reconstructed=True,tie_checks=tie_rows,latest3_and_reference_updates_verified=True,
                family_gates=gates,advance_to_learning_gain_study=all(x['advance'] for x in gates.values()),
                paired_descriptive_differences=paired,
                limits='One fixed prior calibration checkpoint per family; no training, learning-gain endpoint, or inferential population claim. Equal assessment episode counts, not identical realized steps. Source selection occurs before private target computation.')


result={'terminal':terminal_audit(),'replay':replay_audit(),
        'audit_imports_no_experiment_or_testbed_code':True,'original_source_or_results_modified':False}
result['passed']=all(result[k]['passed'] for k in ('terminal','replay'))
OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':result['passed'],'terminal_events':result['terminal']['reconstructed_events'],
                  'replay_receipts':result['replay']['reference_entries_checked'],
                  'replay_gates':result['replay']['family_gates']},indent=2))
