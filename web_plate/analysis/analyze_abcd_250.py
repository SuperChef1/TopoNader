"""Audit A/B/C/D at the same 250-round FE budget, without changing runs."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from analyze_framework_evidence import ROOT, analyze, read

LIMIT = 250
OUT = ROOT/'analysis'/'abcd_250_20261025'
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {
    'A': 'Baseline-4bars-30coverage-seed42-20261021',
    'B': 'TopoNader-4bars-30coverage-seed42-20261021',
    'C': 'Ablation-C-4bars-30coverage-seed42',
    'D': 'Ablation-D-4bars-30coverage-seed42',
}
results = {}
for group, name in NAMES.items():
    result = analyze(name, LIMIT)
    run = ROOT/'runs'/name
    reasons, scopes, sources, scope_errors, operators = (Counter() for _ in range(5))
    audit_iterations, rejections_under_guard, samples = [], [], []
    disabled_guard_errors, random_route_errors, scope_versions = [], [], set()
    for audit in result['audits']:
        i = audit['iteration']
        d = run/f'iteration_{i:04d}'
        f = read(d/'feedback.json')
        decision = f.get('inheritance_decision')
        if decision:
            reasons.update(decision['reasons'])
            if group == 'C':
                p, c = decision['parent_metrics'], decision['child_metrics']
                rank = lambda x: (x['pairwise_accuracy'], x['kendall_tau'], -x['topk_regret'], -x['rmse'])
                expected = audit['parent_id'] != audit['provisional_id'] and rank(c)>rank(p)
                if expected != decision['accepted'] or decision['thresholds']['enabled']:
                    disabled_guard_errors.append(i)
                if decision['accepted'] and (decision['pairwise_improvement']<.03-1e-12 or
                                               decision['topk_regret_increase']>.02+1e-12):
                    rejections_under_guard.append({'iteration':i,
                        'pairwise_improvement':decision['pairwise_improvement'],
                        'topk_regret_increase':decision['topk_regret_increase']})
        trace = read(d/'audit/trace.json')
        audit_iterations.append(i)
        scope_versions.add(trace.get('change_scope_policy', 'missing'))
        sources.update([trace.get('proposal_source', 'llm')])
        if group == 'D':
            if trace.get('proposal_source') != 'random' or trace.get('usage'):
                random_route_errors.append(i)
            if (d/'agents').exists() and list((d/'agents').rglob('*.json')):
                random_route_errors.append(i)
            if f.get('post_reflection') is not None:
                random_route_errors.append(i)
        for attempt in trace.get('candidate_attempts', []):
            scopes['attempts'] += 1
            if attempt.get('accepted'):
                scopes['accepted'] += 1
            scope = attempt.get('change_scope', {})
            for error in scope.get('errors', []):
                scope_errors[error] += 1
            title = attempt.get('proposal', {}).get('title', '')
            if group == 'D':
                operators[title] += 1
        samples.append(trace.get('accepted_candidate_count',0))
    x = loadmat(run/'datasets/dataset_0251.mat')['outputsX'].reshape(-1)
    result.update(group=group, scope_versions=sorted(scope_versions),
        proposal_sources=dict(sources), inheritance_reasons=dict(reasons),
        scope_counts=dict(scopes), scope_errors=dict(scope_errors),
        audit_iterations=audit_iterations, audit_gaps=np.diff([0]+audit_iterations).tolist(),
        candidate_shortfalls=sum(n<5 for n in samples),
        disabled_guard_errors=disabled_guard_errors, random_route_errors=random_route_errors,
        accepted_C_rejected_under_guard=rejections_under_guard, random_operators=dict(operators),
        observed_top5=np.sort(x)[-5:].tolist(),
        best_matches_dataset=bool(np.isclose(np.max(x),result['best'],rtol=0,atol=1e-12)),
        targets={str(t):next((v['iteration'] for v in result['trajectory'] if v['best']>=t),None)
                 for t in [.9,1.,1.03,1.04,1.05,1.06,1.1]})
    results[group] = result
    print('GROUP',group)
    print(json.dumps({k:v for k,v in result.items() if k not in ['config','audits','changes','trajectory']},ensure_ascii=False))
    print('CURVE',[(v['iteration'],v['best']) for v in result['trajectory']
                   if v['iteration'] in [10,25,50,100,150,200,250]])
    print('CHANGES',[(a['iteration'],a['child_fe']['pairwise_accuracy'],
        a['parent_fe']['pairwise_accuracy'],a['segment_best_gain']) for a in result['changes']])
(OUT/'evidence.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('RELATIVE_BEST_TO_A', {g:100*(r['best']/results['A']['best']-1) for g,r in results.items()})
print('RELATIVE_MEAN_CURVE_TO_A', {g:100*(r['mean_best_over_iterations']/results['A']['mean_best_over_iterations']-1)
                                     for g,r in results.items()})
