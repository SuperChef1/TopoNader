"""Compare recorded Modifier JSON against proposals and inherited parent state."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from topo_nader.agents.base import parse_json_object
from topo_nader.architecture.spec import TopoNaderArchitectureSpec

FIELDS = set(TopoNaderArchitectureSpec().to_dict()) - {'name', 'metadata'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def differences(before, after):
    return {key: {'before': before[key], 'after': after[key]}
            for key in sorted(FIELDS) if before[key] != after[key]}

results = []
for run in sorted((ROOT/'runs').iterdir()):
    if not any(name in run.name for name in ('NoReflector','NoProposer')):
        continue
    if not (run/'iterations.jsonl').exists():
        continue
    rows = [json.loads(line) for line in (run/'iterations.jsonl').read_text().splitlines() if line.strip()]
    cfg = read(run/'config.snapshot.json')
    current = read(run/'iteration_0001/provisional_architecture.json')
    counts, modes, errors = Counter(), Counter(), Counter()
    accepted_issues, extra_events, rejected_examples, inactive_events, shortfalls = [], [], [], [], []
    for row in rows:
        i = row['iteration']
        idir = run/f'iteration_{i:04d}'
        assert TopoNaderArchitectureSpec.from_dict(current).architecture_id == row['parent_architecture_id']
        if row.get('audit_performed'):
            trace = read(idir/'audit/trace.json')
            counts['audits'] += 1
            mode = trace['candidate_generation_mode']
            modes[mode] += 1
            if trace['accepted_candidate_count'] < trace['requested_candidate_count']:
                shortfalls.append({'iteration':i,'requested':trace['requested_candidate_count'],
                                   'accepted':trace['accepted_candidate_count']})
            raw_responses = []
            for u in trace.get('usage',[]):
                if u.get('agent') != 'TopoNaderModifier':
                    continue
                try:
                    payload = parse_json_object(u['content'])
                    if isinstance(payload,dict) and isinstance(payload.get('architecture'),dict):
                        payload = payload['architecture']
                    if not isinstance(payload,dict) or 'input_dim' not in payload:
                        continue
                    spec = TopoNaderArchitectureSpec.from_dict(payload)
                    raw_responses.append((spec.architecture_id,spec.to_dict()))
                except (ValueError,TypeError,KeyError):
                    continue
            for attempt in trace.get('candidate_attempts',[]):
                counts['attempts'] += 1
                if not attempt.get('accepted'):
                    counts['rejected'] += 1
                    for error in attempt.get('errors',[]):
                        errors[error.split(':')[0]] += 1
                        if len(rejected_examples) < 3:
                            rejected_examples.append({'iteration':i,'error':error[:250]})
                    continue
                counts['accepted'] += 1
                proposal = attempt['proposal']
                scope = attempt['change_scope']
                matches = [payload for aid,payload in raw_responses
                           if aid == attempt['architecture_id']
                           and differences(current,payload) == scope['actual_changes']]
                issues = []
                if not matches:
                    issues.append('raw_response_architecture_not_matched_to_recorded_diff')
                    actual = scope['actual_changes']
                    after = {**current,**{k:v['after'] for k,v in actual.items()}}
                else:
                    counts['accepted_raw_json_verified'] += 1
                    after = matches[0]
                    actual = differences(current,after)
                mismatch = {k:{'expected':v,'actual':after.get(k)} for k,v in proposal['changes'].items()
                            if after.get(k) != v}
                extras = sorted(set(actual)-set(proposal['changes']))
                if mismatch:
                    issues.append('declared_targets_not_implemented')
                if not scope.get('valid'):
                    issues.append('accepted_scope_invalid')
                if mode.startswith('human_suggestions'):
                    if extras:
                        issues.append('extra_fields_in_human_suggestion_candidate')
                    if not scope.get('human_suggestion_exact_targets_only'):
                        issues.append('missing_human_strict_target_marker')
                if issues:
                    accepted_issues.append({'iteration':i,'architecture_id':attempt['architecture_id'],
                                            'issues':issues,'mismatch':mismatch,'extras':extras})
                inactive = sorted(set(proposal['changes']) & {'predictor_depth','attention_heads'}) \
                    if after['predictor_type']=='mlp' else []
                if inactive:
                    inactive_events.append({'iteration':i,'architecture_id':attempt['architecture_id'],
                        'inactive_fields':inactive,'planned_changes':proposal['changes'],
                        'provisional_winner':row['provisional_architecture_id']==attempt['architecture_id'],
                        'inherited':row['inherited_architecture_id']==attempt['architecture_id']})
                if extras:
                    extra_events.append({'iteration':i,'architecture_id':attempt['architecture_id'],
                        'proposal':proposal,'actual_changes':actual,
                        'supporting_changes':scope.get('supporting_changes'),
                        'provisional_winner':row['provisional_architecture_id']==attempt['architecture_id'],
                        'inherited':row['inherited_architecture_id']==attempt['architecture_id'],
                        'predictor_type':after['predictor_type'],
                        'inactive_mlp_target_fields':sorted(set(proposal['changes']) & {'predictor_depth','attention_heads'})
                            if after['predictor_type']=='mlp' else [],
                        'trace_path':str(idir/'audit/trace.json'),
                        'architecture_path':str(idir/'provisional_architecture.json'),
                        'summary_path':str(idir/'iteration_summary.json')})
        provisional = read(idir/'provisional_architecture.json')
        if row['inherited_architecture_id'] == row['provisional_architecture_id']:
            current = provisional
        else:
            assert row['inherited_architecture_id'] == row['parent_architecture_id']
    results.append({'run':run.name,'completed_iterations':rows[-1]['iteration'],
        'mode_counts':dict(modes),'counts':dict(counts),'rejected_error_categories':dict(errors),
        'accepted_issues':accepted_issues,'extra_events':extra_events,'rejected_examples':rejected_examples,
        'inactive_mlp_target_events':inactive_events,'candidate_shortfalls':shortfalls,
        'human_library_hash':cfg['agents'].get('human_suggestions_sha256')})

import torch
from topo_nader.architecture import TopoNaderModelFactory
checkpoint_checks = []
for r in results:
    for event in r['extra_events']:
        if not event['inactive_mlp_target_fields']:
            continue
        idir = Path(event['summary_path']).parent
        summary = read(idir/'iteration_summary.json')
        checks = {}
        for role,aid in [('parent',summary['parent_architecture_id']),('child',event['architecture_id'])]:
            ckpt = torch.load(idir/'audit/full'/f'{aid}.pth',map_location='cpu',weights_only=True)
            spec = TopoNaderArchitectureSpec.from_dict(ckpt['architecture'])
            model = TopoNaderModelFactory.build(spec)
            model.load_state_dict(ckpt['state_dict'],strict=True)
            checks[role] = {'predictor_type':spec.predictor_type,'predictor_depth_field':spec.predictor_depth,
                'predictor_weight_shapes':{k:list(v.shape) for k,v in ckpt['state_dict'].items()
                                           if k.startswith('predictor.')},
                'checkpoint_matches_current_factory':True}
        checkpoint_checks.append({'run':r['run'],'iteration':event['iteration'],**checks})

report = {'captured_at_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'method':'Reconstruct inherited parent state chronologically; independently compare raw Modifier JSON, proposal targets and recorded scope for every accepted candidate.',
          'runs':results,'checkpoint_checks_for_extra_changes':checkpoint_checks}
(Path(__file__).parent/'evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for r in results:
    print(json.dumps({k:v for k,v in r.items() if k not in {'extra_events','rejected_examples'}},ensure_ascii=False))
    for event in r['extra_events']:
        print('EXTRA',json.dumps(event,ensure_ascii=False))

lines = [
    '# Modifier 是否按提案修改：运行记录核对', '',
    f"核对时间：{report['captured_at_beijing']}（北京时间）。", '',
    '结论：三个人工建议组共 306 个通过候选检查的架构，原始 Modifier JSON 均与提案的目标字段和值一致，没有额外结构修改。该结论不等于每次服务调用都成功，也不能推广为所有自主提案实验都没有语义偏差。', '',
    '## 核对方法与范围', '',
    '逐轮恢复继承父架构；解析审查记录中的原始 Modifier 输出；重新计算全部结构字段的父子差异；与 proposal.changes、记录的 actual_changes 和实际继承 ID 对照。未启动 LLM、训练或 FE，也未修改运行数据。', '',
    '通过候选检查不等于通过零成本筛选、完整训练或最终继承。下表计数是候选通过数。name/metadata 属于说明性字段，不计作结构修改。', '',
    '| 实验 | 已完成轮数 | 审查次数 | 通过候选数 | 原始 JSON 已核实数 | 未兑现声明目标数 | 额外结构修改数 |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for r in results:
    c = r['counts']
    mismatches = sum('declared_targets_not_implemented' in x['issues'] for x in r['accepted_issues'])
    lines.append(f"| {r['run']} | {r['completed_iterations']} | {c['audits']} | {c['accepted']} | {c['accepted_raw_json_verified']} | {mismatches} | {len(r['extra_events'])} |")
lines += [
    '', '## 自主提案 NoReflector-CurrentOnly 的两次额外修改', '',
    '两次均为 proposal.changes 只指定 predictor_depth: 3→2；Modifier 的额外字段在 metadata.supporting_changes 中附有说明，因此旧的宽松范围检查放行。这些解释不是维度兼容的必要条件，不能视为严格执行单字段提案。', '',
    '| 轮次 | 提案之外的实际修改 | 是否作为该轮临时模型 | 最终是否继承 |',
    '|---:|---|---|---|',
]
event_links = []
for r in results:
    for e in r['extra_events']:
        extras = {k:v for k,v in e['actual_changes'].items() if k not in e['proposal']['changes']}
        desc = '; '.join(f"{k}: {v['before']} → {v['after']}" for k,v in extras.items())
        lines.append(f"| {e['iteration']} | {desc} | {e['provisional_winner']} | {e['inherited']} |")
        event_links.append(f"- 原始架构：[第 {e['iteration']} 轮](/{Path(e['architecture_path']).as_posix()})；最终反馈：[iteration_summary](/{Path(e['summary_path']).as_posix()})。")
lines += ['',*event_links]
lines += [
    '', '两次均训练、进入 FE 选点评估，然后被回退。回退不会撤销已消耗的评估或已加入的数据，不能据此认定对搜索轨迹没有影响。', '',
    '更重要的是，两次预测器均为 MLP。factory.py 的 MLP 分支固定为 Linear→activation→Dropout→Linear，不读取 predictor_depth。检查父子训练检查点后，202 轮头部权重形状相同；224 轮的有效变化是头部宽度从 96 改为 64，而非深度从 3 改为 2。checkpoint_checks_for_extra_changes 保存了父子权重形状和工厂加载核对。', '',
    '自主提案中出现无效字段并非全部都是错误实验：切换到 MLP 时带上非活动的 attention_heads/predictor_depth 不会改变网络。不能把这些字段的数字变化当成有效网络变化；详细事件见 evidence.json 的 inactive_mlp_target_events。', '',
    '## 人工随机主组的候选短缺', '',
    'Ablation-Human-NoReflector-NoProposer 共 252 次候选尝试：111 次通过，141 次记录为 generation_or_parse_error。进一步核对原始错误，这 141 次均为请求/连接失败：1 次 HTTP 503、72 次 SSL 连接错误、68 次其他请求失败，并非 141 次架构不按提案修改。', '',
    '第 174、185 轮各请求 5 个候选但取得 0 个；第 192 轮请求 5 个、取得 1 个。该组累计是 111 个通过候选，而不是计划的 125 个。这会改变搜索机会、重试费用与可比性，应随结果报告。', '',
    '另外两个人工建议组在记录的候选层面均取得所请求的数量。旧版 NoReflector-NoProposer 的 65 个候选虽然兑现了同一响应中的声明目标，但其提案也是 Modifier 自己生成，不能声称它在执行外部固定人工建议。仅 NoProposer 的旧目录没有完成轮次，没有可核实的候选运行结果。', '',
    '## 复核文件', '',
    '- audit.py：复核脚本，读取历史运行和两次异常事件的检查点。',
    '- evidence.json：逐组计数、额外修改事件、非活动字段事件、候选短缺及检查点核对。', '',
    '本轮仅做核对，不更改 Modifier 策略、模型工厂或已有实验。', '',
]
(Path(__file__).parent/'report.md').write_text('\n'.join(lines),encoding='utf-8')
