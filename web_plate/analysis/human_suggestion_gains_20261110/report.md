# 随机人工建议：哪些修改改善了结果

证据快照：2026-11-10 11:13:41（北京时间）。只读取已写入 iteration_summary.json 的完成轮次。

结论：原 250 轮实验中，预测头 SiLU 的当轮 FE 最优值增幅最大（15.17%）；Dropout 0.20、编码器 SiLU 和潜变量扩容也通过了父子继承检查。后两者主要表现为预测质量改善，当轮未刷新 FE 最优值。CurrentOnly 另有加深、加宽与残差编码器的收益。

## 分析口径

- 这里的“经验”指人工库规则被随机抽取、实现并评估的事件，不是 Reflector 生成的经验。
- 仅纳入 trace.candidate_generation_mode=human_suggestions_modifier 且关闭 Reflector/Proposer 的运行。旧 modifier_direct 和 human_suggestions_select 不混入。
- 排序准确率使用 inheritance_decision 中同一轮、同一批真实 FE 样本的父子对比；变化按百分点计算。RMSE 降幅=(父-子)/父。
- 当轮 FE 增幅=(当轮 best_so_far/轮前 best_so_far-1)，越大越好。这是该轮找到更优设计的描述，缺少父模型独立选点的反事实 FE 对照，不能视为规则的因果增益。
- 所有继承成功事件都列出；没有以“大幅”的任意阈值事后筛选。

## 原实验

运行：`Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42`。已完成 250 轮，审查 25 次，人工候选实现通过 111 次，继承 4 次；最后 FE 最优值 1.090608101。

| 轮次 | 继承的规则 | 同批 FE 排序准确率 父→子 | 增益/百分点 | 同批 FE RMSE 降幅 | 当轮 FE 最优值 前→后 | 当轮 FE 增幅 |
|---:|---|---:|---:|---:|---:|---:|
| 5 | 预测头激活 LeakyReLU → SiLU (`predictor_silu`) | 15.26% → 41.05% | +25.79 | -0.35% | 0.577696 → 0.665352 | +15.1735% |
| 11 | Dropout 0.10 → 0.20 (`dropout_high`) | 42.11% → 55.26% | +13.16 | +15.32% | 0.762323 → 0.769105 | +0.8897% |
| 64 | 编码器激活 ReLU → SiLU (`encoder_silu`) | 80.00% → 83.68% | +3.68 | +33.29% | 0.983493 → 0.983493 | +0.0000% |
| 247 | 潜变量维度 12 → 18 (`latent_relax`) | 78.42% → 83.68% | +5.26 | +40.54% | 1.090608 → 1.090608 | +0.0000% |

继承事件的原始证据：

- 第 5 轮：[predictor_silu](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0005/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0005/audit/trace.json)。
- 第 11 轮：[dropout_high](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0011/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0011/audit/trace.json)。
- 第 64 轮：[encoder_silu](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0064/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0064/audit/trace.json)。
- 第 247 轮：[latent_relax](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0247/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42/iteration_0247/audit/trace.json)。

完整训练后的验证集排序增益最大的候选（不等于 FE 验证成功）：

| 轮次 | 规则 | 验证集排序增益/百分点 | 验证集 RMSE 降幅 | 被临时选中 | 最终继承 |
|---:|---|---:|---:|---|---|
| 141 | dropout_standard | +5.49 | +5.04% | True | False |
| 53 | encoder_silu | +4.79 | +17.18% | True | False |
| 53 | norm_batch | +4.64 | +14.34% | False | False |
| 141 | encoder_wide | +4.42 | +5.87% | False | False |
| 64 | encoder_silu | +4.08 | +16.42% | True | True |

## CurrentOnly

运行：`Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42`。已完成 95 轮，审查 11 次，人工候选实现通过 55 次，继承 4 次；最后 FE 最优值 1.063873036。

| 轮次 | 继承的规则 | 同批 FE 排序准确率 父→子 | 增益/百分点 | 同批 FE RMSE 降幅 | 当轮 FE 最优值 前→后 | 当轮 FE 增幅 |
|---:|---|---:|---:|---:|---:|---:|
| 5 | 预测头激活 LeakyReLU → SiLU (`predictor_silu`) | 15.26% → 41.05% | +25.79 | -0.35% | 0.577696 → 0.665352 | +15.1735% |
| 11 | 编码器加深 [1024,512,256] → [1024,512,256,128] (`encoder_deep`) | 43.16% → 53.16% | +10.00 | +13.59% | 0.810883 → 0.833611 | +2.8029% |
| 35 | 编码器加宽 [1024,512,256,128] → [1280,640,320,160] (`encoder_wide`) | 52.63% → 68.42% | +15.79 | +6.50% | 0.903983 → 0.911187 | +0.7969% |
| 73 | 普通 MLP 编码器 → 残差 MLP 编码器 (`encoder_residual`) | 84.74% → 89.47% | +4.74 | +7.53% | 1.050065 → 1.050065 | +0.0000% |

继承事件的原始证据：

- 第 5 轮：[predictor_silu](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0005/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0005/audit/trace.json)。
- 第 11 轮：[encoder_deep](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0011/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0011/audit/trace.json)。
- 第 35 轮：[encoder_wide](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0035/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0035/audit/trace.json)。
- 第 73 轮：[encoder_residual](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0073/iteration_summary.json)；修改范围与随机候选见 [trace](/D:/Topo_Nader/web_plate/runs/Ablation-Human-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42/iteration_0073/audit/trace.json)。

完整训练后的验证集排序增益最大的候选（不等于 FE 验证成功）：

| 轮次 | 规则 | 验证集排序增益/百分点 | 验证集 RMSE 降幅 | 被临时选中 | 最终继承 |
|---:|---|---:|---:|---|---|
| 11 | encoder_deep | +5.94 | -7.27% | True | True |
| 73 | encoder_residual | +4.89 | +10.19% | True | True |
| 30 | encoder_wide | +4.51 | +5.11% | True | False |
| 35 | encoder_wide | +4.24 | +3.44% | True | True |
| 30 | encoder_shallow | +4.19 | -2.01% | False | False |

## 不能忽略的反例与限制

1. 两组第 5 轮的预测头 SiLU 事件具有几乎完全相同的数据与数值，不能当作两个独立随机种子的重复验证。两组均为 seed42。
2. 第 5 轮排序准确率虽从 15.26% 升至 41.05%，仍低于通常无并列随机排序的 50% 参照；RMSE 反而恶化约 0.35%，Top-k regret 仍约 0.998。不能仅凭 +25.79 个百分点称其为可靠代理。
3. 原实验第 53 轮 encoder_silu 的验证集排序 +4.79 个百分点、RMSE 降低 17.18%，但真实 FE 父子排序只 +2.63 个百分点，未达到 3 个百分点继承门槛；第 64 轮才继承。
4. 原实验第 141 轮 dropout_standard 的验证集排序 +5.49 个百分点，但真实 FE 对比只 +2.63 个百分点，也未继承。
5. CurrentOnly 第 30 轮加宽编码器的验证集排序 +4.51 个百分点，但真实 FE 排序 -8.95 个百分点，被回退；同一规则第 35 轮为 +15.79 个百分点并获继承。规则效果依赖模型、训练数据和批次。
6. 原实验第 192 轮移除 LayerNorm：真实 FE 排序准确率 88.42% → 74.21%，下降 14.21 个百分点，被回退。
7. 最大的全程当轮 FE 跳升出现在第 2 轮，+45.79%，当时没有审查或架构变更。因此大幅优化也来自原架构下的搜索与数据积累，不能全部归给人工建议。
8. 父子对比每批只有 20 个 FE 设计；由候选模型引导选点，存在选择偏向。190 个样本对不是 190 个独立观测。未进行显著性或多种子稳定性检验。
9. 原实验第 247 轮 latent_relax 的 FE 预测 RMSE 降低 40.54%，但剩余到第 250 轮的 FE 最优值累计只约 +0.0000375%，没有实质性的最终目标改进证据。
10. 关闭 Reflector 不代表 Modifier 完全没有历史上下文；原实验与 CurrentOnly 应分开解释。

## 复核文件

- `evidence.json`：所有完成审查、完整训练候选和主要 FE 跳升的原始数值汇总。
- `audits.csv`：逐次审查的平面表，可查看成功和失败事件。
- `analyze.py`：纯读取实验文件的复现脚本；运行会更新本分析目录中的派生文件。
