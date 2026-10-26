"""Analyze the first 250 completed rounds without changing run artifacts."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

from analyze_framework_evidence import ROOT, analyze, read

LIMIT = 250
OUT = ROOT / 'analysis' / 'seed42_250_20261023'
OUT.mkdir(parents=True, exist_ok=True)
names = [f'{prefix}-4bars-30coverage-seed42-20261021' for prefix in ['Baseline', 'TopoNader']]
runs = [analyze(name, LIMIT) for name in names]
troot = ROOT / 'runs' / names[1]
audits, scope_counts, reasons, interventions = [], Counter(), Counter(), []
policy_errors = []
for row in runs[1]['audits']:
    i = row['iteration']
    d = troot / f'iteration_{i:04d}'
    feedback = read(d / 'feedback.json')
    decision = feedback.get('inheritance_decision')
    trace = read(d / 'audit/trace.json')
    if trace.get('change_scope_policy') != 'proposal_scope_v1':
        policy_errors.append([i, 'missing_scope_policy'])
    for attempt in trace['candidate_attempts']:
        scope = attempt.get('change_scope')
        scope_counts['attempts'] += 1
        if scope:
            scope_counts['scope_checked'] += 1
            if scope['errors']:
                scope_counts['scope_rejected'] += 1
                scope_counts.update('error:'+e.split(':')[0] for e in scope['errors'])
            if attempt.get('accepted'):
                scope_counts['accepted'] += 1
                if not scope['valid']:
                    policy_errors.append([i, 'accepted_invalid_scope'])
                if scope.get('supporting_changes'):
                    scope_counts['accepted_with_supporting_changes'] += 1
        elif attempt.get('accepted'):
            policy_errors.append([i, 'accepted_without_scope_audit'])
    if decision is None:
        policy_errors.append([i, 'missing_inheritance_decision'])
        continue
    reasons.update(decision['reasons'])
    parent, child = decision['parent_metrics'], decision['child_metrics']
    rank = lambda m: (m['pairwise_accuracy'], m['kendall_tau'], -m['topk_regret'], -m['rmse'])
    old_accept = row['parent_id'] != row['provisional_id'] and rank(child) > rank(parent)
    thresholds = decision['thresholds']
    expected = (old_accept and child['sample_count'] >= thresholds['min_feedback_samples']
                and decision['pairwise_improvement'] >= thresholds['min_pairwise_improvement']-1e-12
                and decision['topk_regret_increase'] <= thresholds['max_topk_regret_increase']+1e-12)
    if thresholds['min_pairwise_accuracy'] is not None:
        expected = expected and child['pairwise_accuracy'] >= thresholds['min_pairwise_accuracy']
    if expected != decision['accepted']:
        policy_errors.append([i, 'decision_mismatch'])
    if old_accept and not decision['accepted']:
        interventions.append({'iteration': i, 'reasons': decision['reasons'],
                              'pairwise_gain': decision['pairwise_improvement'],
                              'regret_increase': decision['topk_regret_increase']})
    previous = read(troot / f'iteration_{i-1:04d}/feedback.json') if i > 1 else {}
    trigger = previous.get('provisional_metrics', {}).get('trigger_decision', {})
    audits.append({'iteration': i, 'decision': decision, 'trigger_reasons': trigger.get('reasons'),
                   'changed': row['changed'], 'rollback': row['rollback']})
audit_rounds = [a['iteration'] for a in audits]
gaps = np.diff([0]+audit_rounds).tolist()
if any(g < runs[1]['config']['trigger']['min_audit_interval'] for g in gaps):
    policy_errors.append(['cooldown_violation'])
config_differences = {}
def compare(a,b,path=''):
    for k in a:
        if isinstance(a[k], dict): compare(a[k], b[k], path+k+'.')
        elif a[k] != b[k]: config_differences[path+k] = [a[k],b[k]]
compare(runs[0]['config'], runs[1]['config'])
result = {'limit': LIMIT, 'runs': runs, 'config_differences': config_differences,
          'scope_counts': dict(scope_counts), 'inheritance_reason_counts': dict(reasons),
          'new_guard_interventions': interventions, 'audit_rounds': audit_rounds,
          'audit_gaps': gaps, 'policy_errors': policy_errors, 'audits': audits}
(OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for r in runs:
    print('RUN',r['name'])
    print(json.dumps({k:v for k,v in r.items() if k not in ['config','audits','changes','trajectory']},ensure_ascii=False))
    print('CURVE',[(v['iteration'],v['best']) for v in r['trajectory'] if v['iteration'] in [10,25,50,100,150,200,250]])
    print('TARGET_ROUNDS', {target:next((v['iteration'] for v in r['trajectory'] if v['best']>=target),None)
                            for target in [0.9,1.,1.05,1.06,1.1]})
    print('CHANGES', [(v['iteration'], v['child_fe']['pairwise_accuracy'],
                       v['parent_fe']['pairwise_accuracy'],v['segment_best_gain']) for v in r['changes']])
print('POLICY',json.dumps({k:v for k,v in result.items() if k!='runs' and k!='audits'},ensure_ascii=False))
