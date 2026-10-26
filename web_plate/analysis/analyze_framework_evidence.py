"""Read-only analysis of completed experiment records; no FE or LLM calls."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'framework_evidence_20261021'
OUT.mkdir(parents=True, exist_ok=True)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def average(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def analyze(name, limit):
    r = ROOT / 'runs' / name
    rows = [json.loads(l) for l in (r / 'iterations.jsonl').read_text().splitlines() if l.strip()]
    rows = [x for x in rows if x['iteration'] <= limit]
    cfg = read(r / 'config.snapshot.json')
    audits, changes, pools, counts = [], [], [], []
    epochs = 0
    tokens_in = tokens_out = calls = 0
    llm_seconds = 0.0
    fe_chain_errors, input_chain_errors = [], []
    final = loadmat(r / 'datasets' / f'dataset_{limit+1:04d}.mat')
    x, y = final['inputsX'], final['outputsX'].reshape(-1)
    offset = rows[0]['dataset_size_before']
    initial = read(r / 'iteration_0001' / 'manifest.json')
    for row in rows:
        i = row['iteration']
        d = r / f'iteration_{i:04d}'
        m, f = read(d / 'manifest.json'), read(d / 'feedback.json')
        if m['sample_count'] != offset:
            input_chain_errors.append(i)
        q, z = loadmat(d / 'fe_request.mat'), loadmat(d / 'fe_result.mat')['outputs'].reshape(-1)
        valid = np.isfinite(z) & (z > 0)
        if not (np.array_equal(x[offset:offset+sum(valid)], q['inputs'].reshape(-1, cfg['data']['num_bars'], 4)[valid])
                and np.array_equal(y[offset:offset+sum(valid)], z[valid])):
            fe_chain_errors.append(i)
        offset += sum(valid)
        for metrics_file in d.rglob('*.metrics.json'):
            checkpoint_count = 1  # Each metrics file corresponds to one trained checkpoint.
            counts.append(checkpoint_count)
        epochs += len(m['training'].get('training_history', []))
        search = m.get('search', {})
        local = search.get('elite_local_search', {}).get('candidate_pool_statistics', [])
        global_stats = search.get('global_exploration', {}).get('candidate_pool_statistics')
        for s in local:
            pools.append(('local', s))
        if isinstance(global_stats, dict):
            pools.append(('global', global_stats))
        if row['audit_performed']:
            trace = read(d / 'audit' / 'trace.json')
            child, parent = f['provisional_metrics'], f['parent_metrics']
            a = {'iteration': i, 'changed': row['architecture_changed'], 'rollback': row['rollback'],
                 'parent_id': m['parent_architecture_id'], 'provisional_id': m['provisional_architecture_id'],
                 'inherited_id': f['inherited_architecture_id'], 'child_fe': child, 'parent_fe': parent,
                 'parent_validation': trace.get('parent_diagnostics', {}).get('validation'),
                 'winner_validation': m['training']['validation'],
                 'candidate_count': sum(bool(v.get('accepted')) for v in trace.get('candidate_attempts', [])),
                 'quick_trained': len(trace.get('quick_screen', {})),
                 'batch_improvement': row['best_performance_so_far'] - row['best_performance_before']}
            audits.append(a)
            if row['architecture_changed']:
                next_audit = next((v['iteration'] for v in rows if v['iteration'] > i and v['audit_performed']), limit+1)
                end = min(next_audit-1, limit)
                a['next_audit_or_end'] = end
                a['segment_best_gain'] = rows[end-1]['best_performance_so_far'] - row['best_performance_before']
                changes.append(a)
        for p in (d / 'agents').rglob('*.json') if (d / 'agents').exists() else []:
            obj = read(p)
            if 'prompt_tokens' in obj:
                calls += 1
                tokens_in += obj.get('prompt_tokens', 0)
                tokens_out += obj.get('completion_tokens', 0)
                llm_seconds += obj.get('elapsed_seconds', 0)
    diagnostics = {}
    for kind in ['local', 'global']:
        ps = [s for k,s in pools if k == kind]
        total = sum(s.get('input_candidate_count', 0) for s in ps)
        diagnostics[kind] = {
            'pool_count': len(ps), 'input_candidates': total,
            'repair_fraction': sum(s.get('coverage_repaired_count', 0) for s in ps)/total if total else None,
            'postrepair_duplicate_fraction': sum(s.get('final_structure_duplicate_count', 0) for s in ps)/total if total else None,
            'unique_yield_fraction': sum(s.get('accepted_unique_count', 0) for s in ps)/total if total else None,
            'mean_pool_spearman': average([s.get('ranking_stability', {}).get('spearman_rho') for s in ps]),
            'mean_pool_top10_overlap': average([s.get('ranking_stability', {}).get('top_k_overlap_fraction') for s in ps])}
    return {'name': name, 'iterations': limit, 'initial_hash': initial['dataset_sha256'],
            'config': cfg, 'sample_count': len(y), 'expected_sample_count': int(offset),
            'fe_chain_errors': fe_chain_errors, 'input_chain_errors': input_chain_errors,
            'best': rows[-1]['best_performance_so_far'],
            'mean_best_over_iterations': average([v['best_performance_so_far'] for v in rows]),
            'mean_pairwise': average([v['pairwise_accuracy'] for v in rows]),
            'mean_topk_regret': average([v['topk_regret'] for v in rows]),
            'mean_uncertainty_error_correlation': average([v['uncertainty_error_correlation'] for v in rows]),
            'positive_improvement_rounds': sum(v['best_performance_so_far'] > v['best_performance_before']+1e-12 for v in rows),
            'trained_checkpoint_count': len(counts), 'selected_training_epochs': epochs,
            'recorded_llm_calls': calls, 'recorded_prompt_tokens': tokens_in,
            'recorded_completion_tokens': tokens_out, 'recorded_llm_elapsed_seconds': llm_seconds,
            'audits': audits, 'changes': changes, 'pool_diagnostics': diagnostics,
            'trajectory': [{'iteration': v['iteration'], 'best': v['best_performance_so_far']} for v in rows]}


def main():
    results = []
    for suffix in ['4bars-30coverage-original-load-20261020', '4bars-30coverage', '6bars-30coverage']:
        names = [prefix+'-'+suffix for prefix in ['Baseline', 'TopoNader']]
        limit = min(read(ROOT/'runs'/n/'summary.json')['last_iteration'] for n in names)
        pair = [analyze(n, limit) for n in names]
        results.append({'case': suffix, 'common_iterations': limit, 'runs': pair})
    (OUT / 'evidence.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for pair in results:
        print('CASE', pair['case'], 'common', pair['common_iterations'])
        for r in pair['runs']:
            print(json.dumps({k:v for k,v in r.items() if k not in ['config','audits','changes','trajectory']}, ensure_ascii=False))
            print('CHANGES', [(v['iteration'], v['child_fe']['pairwise_accuracy'],
                               v['parent_fe']['pairwise_accuracy'] if v['parent_fe'] else None,
                               v['segment_best_gain']) for v in r['changes']])


if __name__ == '__main__':
    main()
