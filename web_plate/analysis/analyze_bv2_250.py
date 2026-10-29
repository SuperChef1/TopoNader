"""Read-only Bv2 completion audit and equal-FE-budget comparison."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from analyze_framework_evidence import ROOT, analyze, read

OUT = ROOT / 'analysis' / 'bv2_250_20261027'
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {
    'A42': 'Baseline-4bars-30coverage-seed42-20261021',
    'Bv1_42': 'TopoNader-4bars-30coverage-seed42-20261021',
    'C42': 'Ablation-C-4bars-30coverage-seed42',
    'D42': 'Ablation-D-4bars-30coverage-seed42',
    'Bv2_42': 'TopoNader-Bv2-4bars-30coverage-seed42',
    'Bv2_123': 'TopoNader-Bv2-4bars-30coverage-seed123',
}
results = {}
for group, name in NAMES.items():
    r = analyze(name, 250)
    run = ROOT / 'runs' / name
    rows = [json.loads(s) for s in (run/'iterations.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    rows = [v for v in rows if v['iteration'] <= 250]
    state = read(run/'state.json')
    versions, reasons = Counter(), Counter()
    intervals, policy_errors, scope_errors = [], [], Counter()
    previous = 0
    guard_only = []
    candidate_shortfalls = 0
    for a in r['audits']:
        i = a['iteration']
        trace = read(run/f'iteration_{i:04d}'/'audit/trace.json')
        feedback = read(run/f'iteration_{i:04d}'/'feedback.json')
        versions.update([trace.get('change_scope_policy', 'missing')])
        intervals.append(i-previous)
        previous = i
        decision = feedback.get('inheritance_decision')
        if decision:
            reasons.update(decision['reasons'])
            expected_guard = r['config']['inheritance']['enabled']
            if decision['thresholds']['enabled'] != expected_guard:
                policy_errors.append(i)
            if not decision['accepted'] and not any(x in decision['reasons'] for x in [
                'parent_preferred_by_original_ranking', 'no_distinct_child', 'nonfinite_metrics']):
                guard_only.append({'iteration':i, 'reasons':decision['reasons'],
                    'pairwise_improvement':decision['pairwise_improvement'],
                    'topk_regret_increase':decision['topk_regret_increase']})
        candidate_shortfalls += trace.get('accepted_candidate_count', 0) < 5
        for attempt in trace.get('candidate_attempts', []):
            scope_errors.update(attempt.get('change_scope', {}).get('errors', []))
    final = loadmat(run/'datasets/dataset_0251.mat')
    initial = loadmat(run/'datasets/dataset_0001.mat')
    initial_x = initial.get('inputsX', initial.get('inputs'))
    initial_y = initial.get('outputsX', initial.get('outputs'))
    initial_matches = bool(np.array_equal(final['inputsX'][:300], initial_x) and
                           np.array_equal(final['outputsX'][:300].reshape(-1), initial_y.reshape(-1)))
    y = final['outputsX'].reshape(-1)
    r.update(scope_versions=dict(versions), inheritance_reasons=dict(reasons),
        audit_intervals=intervals, policy_errors=policy_errors, guard_only_rejections=guard_only,
        scope_errors=dict(scope_errors), candidate_shortfalls=candidate_shortfalls,
        last_state_iteration=state['next_iteration'],
        iterations_contiguous=[v['iteration'] for v in rows] == list(range(1,251)),
        initial_segment_matches=initial_matches,
        dataset_best_matches=bool(np.isclose(np.max(y),r['best'],rtol=0,atol=1e-12)),
        new_fe_attempts=sum(v['requested_fe_samples'] for v in rows),
        failed_fe_attempts=sum(v['failed_fe_samples'] for v in rows),
        rollbacks=sum(v['rollback'] for v in rows),
        max_requested_coverage=max(v['coverage_actual_max'] for v in rows),
        first_best_iteration=next(v['iteration'] for v in rows if v['best_performance_so_far']==r['best']),
        targets={str(t):next((v['iteration'] for v in rows if v['best_performance_so_far']>=t),None)
                 for t in [1.,1.05,1.08,1.10]},
        curve={str(v['iteration']):v['best_performance_so_far'] for v in rows
               if v['iteration'] in [10,25,50,100,150,200,250]})
    assert r['iterations_contiguous'] and initial_matches and r['dataset_best_matches']
    assert not r['fe_chain_errors'] and not r['input_chain_errors'] and not policy_errors
    assert r['sample_count'] == r['expected_sample_count']
    results[group] = r
    print(group, json.dumps({k:v for k,v in r.items() if k not in [
        'config','audits','changes','trajectory','pool_diagnostics']},ensure_ascii=False))
    print('CHANGES',[(v['iteration'],v['child_fe']['pairwise_accuracy'],
         v['parent_fe']['pairwise_accuracy'],v['batch_improvement'],v['segment_best_gain'])
         for v in r['changes']])

preflight = read(ROOT/'analysis/bv2_rerun_20261025/preflight_hashes.json')
drift = []
for name, sha in preflight['files'].items():
    p = ROOT/name
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
        drift.append(name)

def flat(d, prefix=''):
    result = {}
    for k,v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):
            result.update(flat(v,key))
        else:
            result[key] = v
    return result

config_diffs = {}
base = flat(results['Bv2_42']['config'])
for g in ['Bv1_42','C42','D42','Bv2_123']:
    other = flat(results[g]['config'])
    other.setdefault('agents.proposal_source','llm')
    config_diffs[g] = {k:[base.get(k),other.get(k)] for k in sorted(base.keys()|other.keys())
                       if base.get(k)!=other.get(k)}

payload = {'runs':results,'preflight_hash_drift':drift,'config_differences_from_Bv2_42':config_diffs}
(OUT/'evidence.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print('PREFLIGHT_DRIFT',drift)
print('CONFIG_DIFFS',json.dumps(config_diffs,ensure_ascii=False))
print('Bv2_42_RELATIVE_FINAL', {g:100*(results['Bv2_42']['best']/results[g]['best']-1)
       for g in ['A42','Bv1_42','C42','D42']})
print('Bv2_42_RELATIVE_CURVE', {g:100*(results['Bv2_42']['mean_best_over_iterations']/results[g]['mean_best_over_iterations']-1)
       for g in ['A42','Bv1_42','C42','D42']})
print('Bv2_123_RELATIVE_42_FINAL',100*(results['Bv2_123']['best']/results['Bv2_42']['best']-1))
