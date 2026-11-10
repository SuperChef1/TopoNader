"""Read completed random-human audits without modifying experiment data."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import csv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

results = []
for run in sorted((ROOT / 'runs').glob('Ablation-Human-NoReflector-NoProposer*')):
    cfg = read(run / 'config.snapshot.json')
    assert cfg['agents']['reflector_enabled'] is False
    assert cfg['agents']['proposer_enabled'] is False
    assert cfg['agents']['proposer_disabled_mode'] == 'human_suggestions'
    rows = [json.loads(line) for line in (run / 'iterations.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if (run / f"iteration_{r['iteration']:04d}" / 'iteration_summary.json').exists()]
    audits = []
    candidate_rows = []
    for r in rows:
        if not r['audit_performed']:
            continue
        i = r['iteration']
        idir = run / f'iteration_{i:04d}'
        t = read(idir / 'audit/trace.json')
        assert t['candidate_generation_mode'] == 'human_suggestions_modifier'
        assert t['human_suggestion_policy']['library_sha256'] == cfg['agents']['human_suggestions_sha256']
        chosen = read(idir / 'provisional_architecture.json')
        decision = r.get('inheritance_decision')
        rule = chosen.get('metadata', {}).get('human_suggestion', {}).get('id') if r['provisional_architecture_id'] != r['parent_architecture_id'] else None
        changes = chosen.get('metadata', {}).get('change_scope', {}).get('actual_changes', {}) if rule else {}
        parent_val = read(idir / 'audit/full' / (r['parent_architecture_id'] + '.metrics.json'))
        child_val = read(idir / 'audit/full' / (r['provisional_architecture_id'] + '.metrics.json'))
        later = [x for x in rows if i <= x['iteration'] < i+5]
        end = later[-1]
        entry = {
            'iteration': i, 'rule': rule, 'changes': changes,
            'accepted': bool(decision and decision['accepted']),
            'rollback': r['rollback'],
            'parent_id': r['parent_architecture_id'], 'provisional_id': r['provisional_architecture_id'],
            'validation_parent': parent_val, 'validation_child': child_val,
            'inheritance_decision': decision,
            'fe_best_before': r['best_performance_before'], 'fe_best_after': r['best_performance_so_far'],
            'fe_improvement_pct': 100*(r['best_performance_so_far']/r['best_performance_before']-1),
            'fe_best_through_5_rounds': end['best_performance_so_far'],
            'window_end_iteration': end['iteration'],
            'fe_5_round_improvement_pct': 100*(end['best_performance_so_far']/r['best_performance_before']-1),
            'trace_path': str(idir / 'audit/trace.json'),
            'summary_path': str(idir / 'iteration_summary.json'),
            'sampled_rules': [a['proposal']['human_suggestion_id'] for a in t['candidate_attempts'] if a.get('accepted')],
        }
        audits.append(entry)
        for a in t['candidate_attempts']:
            if not a.get('accepted'):
                continue
            cpath = idir / 'audit/full' / (a['architecture_id'] + '.metrics.json')
            if cpath.exists():
                val = read(cpath)
                candidate_rows.append({'iteration':i, 'rule':a['proposal']['human_suggestion_id'],
                    'changes':a['change_scope']['actual_changes'],
                    'parent_pairwise':parent_val['pairwise_accuracy'], 'child_pairwise':val['pairwise_accuracy'],
                    'pairwise_gain':val['pairwise_accuracy']-parent_val['pairwise_accuracy'],
                    'parent_rmse':parent_val['rmse'], 'child_rmse':val['rmse'],
                    'rmse_reduction_pct':100*(1-val['rmse']/parent_val['rmse']),
                    'parent_regret':parent_val['topk_regret'], 'child_regret':val['topk_regret'],
                    'chosen':a['architecture_id']==r['provisional_architecture_id'],
                    'inherited':a['architecture_id']==r['inherited_architecture_id'],
                    'metrics_path':str(cpath)})
    jumps = sorted([{'iteration':r['iteration'], 'gain_pct':100*(r['best_performance_so_far']/r['best_performance_before']-1),
        'before':r['best_performance_before'], 'after':r['best_performance_so_far'],
        'audit':r['audit_performed'], 'changed':r['architecture_changed'],
        'architecture_id':r['provisional_architecture_id']} for r in rows], key=lambda x:x['gain_pct'],reverse=True)
    results.append({'run':run.name, 'snapshot_completed_through':rows[-1]['iteration'],
        'config':cfg, 'final_best':rows[-1]['best_performance_so_far'], 'audits':audits,
        'full_candidates':candidate_rows,'largest_fe_jumps':jumps[:12]})

payload = {'captured_at_utc':datetime.now(timezone.utc).isoformat(), 'runs':results}
(OUT / 'evidence.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
csv_rows = []
for run in results:
    for a in run['audits']:
        d = a['inheritance_decision'] or {}
        parent, child = d.get('parent_metrics', {}), d.get('child_metrics', {})
        if a['accepted']:
            assert a['rule'] in a['sampled_rules']
            assert a['parent_id'] != a['provisional_id']
            assert abs(child['pairwise_accuracy']-parent['pairwise_accuracy']-d['pairwise_improvement']) < 1e-12
        csv_rows.append({'run':run['run'], 'iteration':a['iteration'], 'rule':a['rule'],
            'accepted':a['accepted'], 'changes':json.dumps(a['changes']),
            'parent_pairwise':parent.get('pairwise_accuracy'), 'child_pairwise':child.get('pairwise_accuracy'),
            'pairwise_gain_pp':100*d['pairwise_improvement'] if d else None,
            'rmse_reduction_pct':100*(1-child['rmse']/parent['rmse']) if d else None,
            'parent_topk_regret':parent.get('topk_regret'), 'child_topk_regret':child.get('topk_regret'),
            'fe_best_before':a['fe_best_before'], 'fe_best_after':a['fe_best_after'],
            'fe_improvement_pct':a['fe_improvement_pct'], 'summary_path':a['summary_path']})
with (OUT / 'audits.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)

labels = {
    'predictor_silu':'预测头激活 LeakyReLU → SiLU', 'dropout_high':'Dropout 0.10 → 0.20',
    'encoder_silu':'编码器激活 ReLU → SiLU', 'latent_relax':'潜变量维度 12 → 18',
    'encoder_deep':'编码器加深 [1024,512,256] → [1024,512,256,128]',
    'encoder_wide':'编码器加宽 [1024,512,256,128] → [1280,640,320,160]',
    'encoder_residual':'普通 MLP 编码器 → 残差 MLP 编码器',
}
report = [
    '# 随机人工建议：哪些修改改善了结果', '',
    f"证据快照：{datetime.fromisoformat(payload['captured_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}（北京时间）。只读取已写入 iteration_summary.json 的完成轮次。", '',
    '结论：原 250 轮实验中，预测头 SiLU 的当轮 FE 最优值增幅最大（15.17%）；Dropout 0.20、编码器 SiLU 和潜变量扩容也通过了父子继承检查。后两者主要表现为预测质量改善，当轮未刷新 FE 最优值。CurrentOnly 另有加深、加宽与残差编码器的收益。', '',
    '## 分析口径', '',
    '- 这里的“经验”指人工库规则被随机抽取、实现并评估的事件，不是 Reflector 生成的经验。',
    '- 仅纳入 trace.candidate_generation_mode=human_suggestions_modifier 且关闭 Reflector/Proposer 的运行。旧 modifier_direct 和 human_suggestions_select 不混入。',
    '- 排序准确率使用 inheritance_decision 中同一轮、同一批真实 FE 样本的父子对比；变化按百分点计算。RMSE 降幅=(父-子)/父。',
    '- 当轮 FE 增幅=(当轮 best_so_far/轮前 best_so_far-1)，越大越好。这是该轮找到更优设计的描述，缺少父模型独立选点的反事实 FE 对照，不能视为规则的因果增益。',
    '- 所有继承成功事件都列出；没有以“大幅”的任意阈值事后筛选。', '',
]
for run in results:
    short='原实验' if 'CurrentOnly' not in run['run'] else 'CurrentOnly'
    report += [f'## {short}', '', f"运行：`{run['run']}`。已完成 {run['snapshot_completed_through']} 轮，审查 {len(run['audits'])} 次，人工候选实现通过 {sum(len(a['sampled_rules']) for a in run['audits'])} 次，继承 {sum(a['accepted'] for a in run['audits'])} 次；最后 FE 最优值 {run['final_best']:.9f}。", '',
        '| 轮次 | 继承的规则 | 同批 FE 排序准确率 父→子 | 增益/百分点 | 同批 FE RMSE 降幅 | 当轮 FE 最优值 前→后 | 当轮 FE 增幅 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for a in run['audits']:
        if not a['accepted']:
            continue
        d=a['inheritance_decision']; p=d['parent_metrics']; c=d['child_metrics']
        report += [f"| {a['iteration']} | {labels.get(a['rule'],a['rule'])} (`{a['rule']}`) | {100*p['pairwise_accuracy']:.2f}% → {100*c['pairwise_accuracy']:.2f}% | +{100*d['pairwise_improvement']:.2f} | {100*(1-c['rmse']/p['rmse']):+.2f}% | {a['fe_best_before']:.6f} → {a['fe_best_after']:.6f} | +{a['fe_improvement_pct']:.4f}% |"]
    report += ['', '继承事件的原始证据：', '']
    for a in run['audits']:
        if a['accepted']:
            report += [f"- 第 {a['iteration']} 轮：[{a['rule']}](/{Path(a['summary_path']).as_posix()})；修改范围与随机候选见 [trace](/{Path(a['trace_path']).as_posix()})。"]
    report += ['', '完整训练后的验证集排序增益最大的候选（不等于 FE 验证成功）：', '',
        '| 轮次 | 规则 | 验证集排序增益/百分点 | 验证集 RMSE 降幅 | 被临时选中 | 最终继承 |',
        '|---:|---|---:|---:|---|---|']
    for c in sorted(run['full_candidates'],key=lambda x:x['pairwise_gain'],reverse=True)[:5]:
        report += [f"| {c['iteration']} | {c['rule']} | +{100*c['pairwise_gain']:.2f} | {c['rmse_reduction_pct']:+.2f}% | {c['chosen']} | {c['inherited']} |"]
    report += ['']
report += [
    '## 不能忽略的反例与限制', '',
    '1. 两组第 5 轮的预测头 SiLU 事件具有几乎完全相同的数据与数值，不能当作两个独立随机种子的重复验证。两组均为 seed42。',
    '2. 第 5 轮排序准确率虽从 15.26% 升至 41.05%，仍低于通常无并列随机排序的 50% 参照；RMSE 反而恶化约 0.35%，Top-k regret 仍约 0.998。不能仅凭 +25.79 个百分点称其为可靠代理。',
    '3. 原实验第 53 轮 encoder_silu 的验证集排序 +4.79 个百分点、RMSE 降低 17.18%，但真实 FE 父子排序只 +2.63 个百分点，未达到 3 个百分点继承门槛；第 64 轮才继承。',
    '4. 原实验第 141 轮 dropout_standard 的验证集排序 +5.49 个百分点，但真实 FE 对比只 +2.63 个百分点，也未继承。',
    '5. CurrentOnly 第 30 轮加宽编码器的验证集排序 +4.51 个百分点，但真实 FE 排序 -8.95 个百分点，被回退；同一规则第 35 轮为 +15.79 个百分点并获继承。规则效果依赖模型、训练数据和批次。',
    '6. 原实验第 192 轮移除 LayerNorm：真实 FE 排序准确率 88.42% → 74.21%，下降 14.21 个百分点，被回退。',
    '7. 最大的全程当轮 FE 跳升出现在第 2 轮，+45.79%，当时没有审查或架构变更。因此大幅优化也来自原架构下的搜索与数据积累，不能全部归给人工建议。',
    '8. 父子对比每批只有 20 个 FE 设计；由候选模型引导选点，存在选择偏向。190 个样本对不是 190 个独立观测。未进行显著性或多种子稳定性检验。',
    '9. 原实验第 247 轮 latent_relax 的 FE 预测 RMSE 降低 40.54%，但剩余到第 250 轮的 FE 最优值累计只约 +0.0000375%，没有实质性的最终目标改进证据。',
    '10. 关闭 Reflector 不代表 Modifier 完全没有历史上下文；原实验与 CurrentOnly 应分开解释。', '',
    '## 复核文件', '',
    '- `evidence.json`：所有完成审查、完整训练候选和主要 FE 跳升的原始数值汇总。',
    '- `audits.csv`：逐次审查的平面表，可查看成功和失败事件。',
    '- `analyze.py`：纯读取实验文件的复现脚本；运行会更新本分析目录中的派生文件。', '',
]
(OUT / 'report.md').write_text('\n'.join(report),encoding='utf-8')
for run in results:
    print('\nRUN',run['run'],'completed',run['snapshot_completed_through'],'best',run['final_best'], 'audits',len(run['audits']))
    for a in run['audits']:
        d=a['inheritance_decision'] or {}
        print(json.dumps({'i':a['iteration'],'rule':a['rule'],'accepted':a['accepted'],'changes':a['changes'],
            'parent_pairwise':d.get('parent_metrics',{}).get('pairwise_accuracy'),
            'child_pairwise':d.get('child_metrics',{}).get('pairwise_accuracy'),
            'gain':d.get('pairwise_improvement'),'regret_increase':d.get('topk_regret_increase'),
            'val_pairwise_gain':a['validation_child']['pairwise_accuracy']-a['validation_parent']['pairwise_accuracy'],
            'fe_before':a['fe_best_before'],'fe_after':a['fe_best_after'],'fe_gain_pct':a['fe_improvement_pct'],
            'window_gain_pct':a['fe_5_round_improvement_pct']},ensure_ascii=False))
    print('TOP FULL VALIDATION',json.dumps(sorted(run['full_candidates'],key=lambda x:x['pairwise_gain'],reverse=True)[:5]))
    print('TOP FE JUMPS',json.dumps(run['largest_fe_jumps']))
