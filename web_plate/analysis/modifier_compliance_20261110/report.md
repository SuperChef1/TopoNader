# Modifier 是否按提案修改：运行记录核对

核对时间：2026-11-10T11:45:35.562009+08:00（北京时间）。

结论：三个人工建议组共 306 个通过候选检查的架构，原始 Modifier JSON 均与提案的目标字段和值一致，没有额外结构修改。该结论不等于每次服务调用都成功，也不能推广为所有自主提案实验都没有语义偏差。

## 核对方法与范围

逐轮恢复继承父架构；解析审查记录中的原始 Modifier 输出；重新计算全部结构字段的父子差异；与 proposal.changes、记录的 actual_changes 和实际继承 ID 对照。未启动 LLM、训练或 FE，也未修改运行数据。

通过候选检查不等于通过零成本筛选、完整训练或最终继承。下表计数是候选通过数。name/metadata 属于说明性字段，不计作结构修改。

| 实验 | 已完成轮数 | 审查次数 | 通过候选数 | 原始 JSON 已核实数 | 未兑现声明目标数 | 额外结构修改数 |
|---|---:|---:|---:|---:|---:|---:|
| Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42 | 250 | 25 | 111 | 111 | 0 | 0 |
| Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42 | 95 | 11 | 55 | 55 | 0 | 0 |
| Ablation-HumanSelect-NoProposer-4bars-30coverage-seed42 | 250 | 28 | 140 | 140 | 0 | 0 |
| Ablation-NoReflector-4bars-30coverage-seed42 | 125 | 17 | 85 | 85 | 0 | 0 |
| Ablation-NoReflector-CurrentOnly-4bars-30coverage-seed42 | 250 | 29 | 145 | 145 | 0 | 2 |
| Ablation-NoReflector-NoProposer-4bars-30coverage-seed42 | 96 | 13 | 65 | 65 | 0 | 0 |

## 自主提案 NoReflector-CurrentOnly 的两次额外修改

两次均为 proposal.changes 只指定 predictor_depth: 3→2；Modifier 的额外字段在 metadata.supporting_changes 中附有说明，因此旧的宽松范围检查放行。这些解释不是维度兼容的必要条件，不能视为严格执行单字段提案。

| 轮次 | 提案之外的实际修改 | 是否作为该轮临时模型 | 最终是否继承 |
|---:|---|---|---|
| 202 | dropout: 0.1 → 0.05 | True | False |
| 224 | predictor_width: 96 → 64 | True | False |

- 原始架构：[第 202 轮](/D:/Topo_Nader/web_plate/runs/Ablation-NoReflector-CurrentOnly-4bars-30coverage-seed42/iteration_0202/provisional_architecture.json)；最终反馈：[iteration_summary](/D:/Topo_Nader/web_plate/runs/Ablation-NoReflector-CurrentOnly-4bars-30coverage-seed42/iteration_0202/iteration_summary.json)。
- 原始架构：[第 224 轮](/D:/Topo_Nader/web_plate/runs/Ablation-NoReflector-CurrentOnly-4bars-30coverage-seed42/iteration_0224/provisional_architecture.json)；最终反馈：[iteration_summary](/D:/Topo_Nader/web_plate/runs/Ablation-NoReflector-CurrentOnly-4bars-30coverage-seed42/iteration_0224/iteration_summary.json)。

两次均训练、进入 FE 选点评估，然后被回退。回退不会撤销已消耗的评估或已加入的数据，不能据此认定对搜索轨迹没有影响。

更重要的是，两次预测器均为 MLP。factory.py 的 MLP 分支固定为 Linear→activation→Dropout→Linear，不读取 predictor_depth。检查父子训练检查点后，202 轮头部权重形状相同；224 轮的有效变化是头部宽度从 96 改为 64，而非深度从 3 改为 2。checkpoint_checks_for_extra_changes 保存了父子权重形状和工厂加载核对。

自主提案中出现无效字段并非全部都是错误实验：切换到 MLP 时带上非活动的 attention_heads/predictor_depth 不会改变网络。不能把这些字段的数字变化当成有效网络变化；详细事件见 evidence.json 的 inactive_mlp_target_events。

## 人工随机主组的候选短缺

Ablation-Human-NoReflector-NoProposer 共 252 次候选尝试：111 次通过，141 次记录为 generation_or_parse_error。进一步核对原始错误，这 141 次均为请求/连接失败：1 次 HTTP 503、72 次 SSL 连接错误、68 次其他请求失败，并非 141 次架构不按提案修改。

第 174、185 轮各请求 5 个候选但取得 0 个；第 192 轮请求 5 个、取得 1 个。该组累计是 111 个通过候选，而不是计划的 125 个。这会改变搜索机会、重试费用与可比性，应随结果报告。

另外两个人工建议组在记录的候选层面均取得所请求的数量。旧版 NoReflector-NoProposer 的 65 个候选虽然兑现了同一响应中的声明目标，但其提案也是 Modifier 自己生成，不能声称它在执行外部固定人工建议。仅 NoProposer 的旧目录没有完成轮次，没有可核实的候选运行结果。

## 复核文件

- audit.py：复核脚本，读取历史运行和两次异常事件的检查点。
- evidence.json：逐组计数、额外修改事件、非活动字段事件、候选短缺及检查点核对。

本轮仅做核对，不更改 Modifier 策略、模型工厂或已有实验。
