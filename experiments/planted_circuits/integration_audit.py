"""Audit archived summaries, without downloading or executing the host LLM.

This is a saved-record consistency audit, not independent model reproduction.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/planted-circuits-v1'
METRICS = ('weights_rank', 'rows_rank', 'output_rank', 'ablation_rank_smallest')
CONFIGS = (('N1','N1'),('N2','N2'),('N2b','N2b'),('N2c-L9','N2c'),
           ('N2c-L18','N2c_L18'),('N2d','N2d'),('N2e','N2e'))


def read(name):
    return json.loads((RESULTS / name).read_text())


def audit():
    manifest = read('import-manifest.json')
    for item in manifest['files']:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'], item['path']
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    rows = []
    for label, suffix in CONFIGS:
        d = read(f'baselines_{suffix}.json')
        stock, plant = dict(d['stock']), dict(d['planted'])
        if label == 'N1':
            r = read('baselines_N1_rows.json')
            stock['rows_rank'], plant['rows_rank'] = r['stock_rows_rank'], r['planted_rows_rank']
        for side in (stock, plant):
            assert side['n_heads'] == 288
            assert all(isinstance(side[m], int) and 1 <= side[m] <= 288 for m in METRICS)
            assert side['ablation_rank_smallest'] + side['ablation_rank_largest'] == 289
        rows.append(dict(configuration=label,stock=[stock[m] for m in METRICS],
                         planted=[plant[m] for m in METRICS],
                         selected_rank_comparison_pass=all(plant[m] >= stock[m] for m in METRICS)))
    passes = [r['configuration'] for r in rows if r['selected_rank_comparison_pass']]
    assert passes == ['N2d', 'N2e'], passes
    false_fire = {}; causal = {}
    for v in ('n2d','n2e'):
        f = read(f'false_fire_{v}_L18_h11.json')
        assert f['stock']['seqs_with_p>0.5'] == 0
        assert f[v]['seqs_with_p>0.5'] == len(f[v]['first_fire_pos']) == 1
        # Script scans positions 1..255 in each of 16 chunks. Retained summaries
        # permit checking the integer implied by the fraction, not raw re-scoring.
        rate = f[v]['frac_p>0.5']; count = round(rate * 4080)
        assert abs(rate - count / 4080) < 1e-12
        false_fire[v] = dict(passages=16,scored_positions=4080,firing_passages=1,
                             positions_above_half=count,first_fire_pos=f[v]['first_fire_pos'])
        e = read(f'{v}_eval_L18_h11.json');s=e['state_swap'];i=e['iit']
        assert s['n_pairs'] == i['n'] == 12
        assert s['k_on_cols'] > .99 and s['B'] > .99
        assert s['other_heads'] < 1e-6 and s['upstream'] < 1e-6
        assert i['necessity_p'] < .001 and .30 < i['sufficiency_p'] < .34
        causal[v]=dict(late_columns=s['k_on_cols'],triggered=s['B'],
                       early_sufficiency=i['sufficiency_p'],early_necessity=i['necessity_p'])
    e = read('n2c_eval_L18_h11.json')['state_swap']
    assert e['k_on_cols'] < 1e-6 and e['head'] > .95
    e = read('n2e_eval_L18_h11.json')['state_swap']
    assert e['relay_no_latch'] > .2 and e['relay_values_zeroed'] < 1e-6
    return dict(passed=True,scope='Archived file integrity and summary consistency only; no model execution',
                source_files_verified=len(manifest['files']),rank_columns=list(METRICS),
                configurations=rows,false_fire=false_fire,causal_summaries=causal,
                benchmark_acceptance='not_met',reader_protocol='draft_not_frozen_not_run',
                limitations=['Checkpoints absent','No raw per-token probabilities or per-head score arrays supplied',
                             'No independent training, inference or causal-intervention reproduction',
                             'Selected-rank comparison does not establish chance detection'])


if __name__ == '__main__':
    print(json.dumps(audit(),indent=2))
