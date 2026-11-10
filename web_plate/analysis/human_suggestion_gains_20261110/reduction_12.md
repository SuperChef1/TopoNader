# 人工建议库：22 条精简为 12 条

2026-11-10。用户选择保留约 10–12 条及更多对照选项，采用 12 条新版本。

这是人工预设建议库的精简，不清理运行中累积的 architecture_experiences.jsonl。
新库：`topo_nader/agents/human_suggestions_v2_12.json`，策略版本 `surrogate_human_v2_12`。

| 类别 | 保留的规则 ID | 数量 | 选择理由 |
|---|---|---:|---|
| 编码器类型 | encoder_residual / encoder_plain | 2 | 残差曾继承，保留普通 MLP 作为对照和回退方向 |
| 编码器宽度 | encoder_wide / encoder_narrow | 2 | 加宽曾继承，但也有失败；保留缩窄以限制容量与参数成本 |
| 编码器深度 | encoder_deep / encoder_shallow | 2 | 加深曾继承，保留减层以避免只允许模型增大 |
| 潜变量容量 | latent_relax / latent_compress | 2 | 扩容曾改善预测，压缩保留相反方向；没有最终 FE 收益保证 |
| 编码器激活 | encoder_silu | 1 | 曾降低同批 FE 预测 RMSE，并通过继承检查 |
| 预测头激活 | predictor_silu | 1 | 第 5 轮出现明显 FE 目标跳升；排序仍弱，非稳定有效证明 |
| Dropout | dropout_high / dropout_standard | 2 | 保留 0.20 与 0.10 两档，允许增强与恢复标准正则化 |
| 合计 | | 12 | 7 种曾继承规则，加 5 个对照方向 |

删除的 10 条（只从新库移除）：

- predictor_mlp、predictor_transformer：固定预测器模型族，缩小搜索范围。
- norm_batch、norm_layer、norm_none：不在此精简实验中搜索归一化方式。
- encoder_relu、encoder_gelu：编码器只保留 SiLU 作为激活替换方向。
- predictor_relu、predictor_gelu：预测头只保留 SiLU 作为激活替换方向。
- dropout_low：由 3 档减为 0.10/0.20 两档，移除 0.05。

删除依据是控制实验范围与减少近似选项，不等于这些规则无效。
例如 norm_batch 在第 53 轮的完整训练验证集上有改善，但没有作为候选胜者接受真实 FE 对比，证据不足以判定其无效。

约束与解释：

1. 原 v1 库的字节和所有旧配置不变。按配置 SHA256 分派新旧库，历史 22 条运行可以继续复现；不能把旧实验中途换成 12 条。
2. 新实验名称为 Human12 或 Human12Select，使用新目录和原始初始数据从第 1 轮开始。没有启动真实 LLM、训练或 FE 实验。
3. 保留规则的操作、数值与来源字段逐条沿用 v1，仅缩减菜单；没有同时改变训练、FE、候选数量或继承条件。
4. 缩减菜单不代表同比节省费用：每轮仍请求相同数量候选，LLM 调用和训练成本取决于实际审查与重试。更小菜单也会改变抽样概率和随机序列。
5. 该方案参考了同一个 seed42 的已知结果，存在事后选择偏差。需要冻结规则后用新种子配对比较，不能把历史同 seed42 的结果当成独立验证。
6. 编码器/预测头改成 SiLU 后，这两个方向会成为 no-op；新库没有逆向激活规则。其他结构和 Dropout 有对照方向。达到容量边界时，可用项可能少于 12，甚至少于请求数量，继续沿用短缺记录，不额外扩展规则。
7. 去掉归一化和预测器族切换后，当前研究问题变为在这些结构固定的条件下搜索容量、连接、SiLU 和 Dropout；不再覆盖原 22 条菜单的完整范围。

可直接选用的配置：

- `configs/Ablation-Human12-NoReflector-NoProposer-4bars-30coverage-seed42.json`
- `configs/Ablation-Human12-NoReflector-NoProposer-CurrentOnly-4bars-30coverage-seed42.json`
- `configs/Ablation-Human12-NoProposer-4bars-30coverage-seed42.json`
- `configs/Ablation-Human12Select-NoProposer-4bars-30coverage-seed42.json`

证据与复核：本目录 `report.md`、`evidence.json`、`audits.csv`。这些记录证明保留项在所观察事件中的表现，不证明规则跨阶段或跨随机种子有效。
