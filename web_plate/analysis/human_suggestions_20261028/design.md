# 面向 FE 排序代理的人工预设建议基线

日期：2026-10-28。检索和实现均服务于当前参数化代理模型；没有启动真实 LLM、训练或 FE 实验。已有运行目录和配置不迁移、不改名。

## 检索结论与限制

本次检索尚未定位到同时满足以下条件的现成公开建议库：面向低维几何参数到标量 FE 响应的神经代理、提供可复用的局部结构修改清单、可直接交给 Modifier 实现，并与当前固定训练/FE预算兼容。这是有限检索的结果，不是“相关工作不存在”或新颖性证明。

查阅的邻近原始研究包括：

- [LeMo-NADe](https://arxiv.org/abs/2402.18443)：专家系统驱动 LLM 进行神经架构发现，面向图像分类等任务；不能直接称其为本项目的代理模型规则库。
- [NADER 官方 manual_v1.json](https://github.com/yang-ze-kang/NADER/blob/20770bd694b3a9f616a7dc851442258b99e03e38/data/inspirations/manual_v1.json)：实际取得并审阅，固定到提交20770bd694b3a9f616a7dc851442258b99e03e38。文件有8项 mutate 和2项 cross 建议，包括残差、瓶颈、反向瓶颈、分组卷积、SE、深度可分离卷积、增大卷积核、交换激活和归一化次序，以及串联、并联。
- [LLM-SAEA](https://arxiv.org/html/2507.02892v1)：在线选择 GP、RBF、PRS、KNN 等代理及采样准则的组合；不是当前神经代理的局部架构修改库。
- [Revisiting Deep Learning Models for Tabular Data](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)及[作者代码](https://github.com/yandex-research/rtdl-revisiting-models)：支持把 MLP、残差模型、Transformer 类方法作为可比较的架构先验；不保证它们在 FE 排序任务中的表现，也不等于当前 bar-token Transformer 是 FT-Transformer。
- 检索到 LLM-Driven Framework for Industrial Design Automation（doi:10.1016/j.eng.2026.04.009）的代理构建/模板相关条目，但全文获取受限，未把其未核实细节当作建议库依据。

精确查询包括 surrogate model / neural architecture / expert rules / knowledge base、LeMo-NADe expert system，以及 NADER 仓库目录和人工建议文件。未把“surrogate-assisted NAS 中预测架构精度的代理”与“为真实 FE 建立的响应代理”混为一谈。

NADER 代码的目录、人工建议文件及相关入口/提示词已保存于本目录，供来源复核；不声称已完整复现其所有消融脚本。该仓库目录树的SHA与上述提交一致。

## 迁移与建议库

库文件：`topo_nader/agents/human_suggestions_v1.json`，版本 `surrogate_human_v1`，共22条。

| 类别 | 条数 | 具体目标与解释 |
|---|---:|---|
| 编码器连接 | 2 | residual_mlp / mlp；残差动机迁移自NADER，普通MLP作为对照 |
| 编码器宽度 | 2 | 全部隐藏宽度×0.75 / ×1.25，按16取整并限制配置边界 |
| 编码器深度 | 2 | 删除末层 / 增加一个宽度约为末层一半的隐藏层 |
| 潜变量容量 | 2 | latent_dim×0.75 / ×1.5，作为压缩与放宽压缩的假设 |
| 预测器 | 2 | 已实现的MLP / bar-token Transformer，不新增模型族 |
| 归一化 | 3 | batch_norm / layer_norm / none |
| 编码器激活 | 3 | relu / gelu / silu |
| 预测头激活 | 3 | relu / gelu / silu，不改变Transformer内部硬编码激活 |
| dropout | 3 | 0.05 / 0.10 / 0.20，保留非零MC-dropout |

数值幅度、取整规则和候选顺序是本项目设计选择，不是文献给出的最优值。所有建议只是待验证的架构假设，不承诺提高误差、排序或最终FE目标。

归一化、正则化和激活的来源分别为 [Batch Normalization](https://arxiv.org/abs/1502.03167)、[Layer Normalization](https://arxiv.org/abs/1607.06450)、[Dropout](https://jmlr.org/papers/v15/srivastava14a.html)、[GELU](https://arxiv.org/abs/1606.08415)、[Swish](https://arxiv.org/abs/1710.05941)。这些是一般网络方法来源，不构成针对本FE任务的实验验证。

只保留NADER残差的直接结构类比和瓶颈的潜表示容量类比。CNN卷积、卷积核、SE、任意图并联/串联和归一化顺序调整不在当前schema中，明确排除。潜变量放宽压缩不能冒称复现CNN反向瓶颈，增加隐藏层也不是NADER图拼接。

## 新实验的实际定义

设置 `agents.proposer_enabled=false` 且 `agents.proposer_disabled_mode="human_suggestions"`：

1. 以 `random_seed` 派生独立的 `human_suggestions:<iteration>` 随机流。
2. 在每次审查按种子置乱22条规则，不依赖预测指标、FE表现或LLM评分排序。
3. 程序计算精确changes，检查无效/重复/不适用项和参数边界，生成外部提案。预检查不消耗训练的全局CPU随机流。
4. Modifier只负责把外部提案实现为完整架构，不得选择新提案或更改额外结构字段。即使写了配套理由，额外结构变化仍被程序拒绝。
5. 使用原有零成本筛选、训练预算、候选数、重试上限、搜索、FE和继承保护。候选不足照常记录，不默默放宽规则。

Transformer从MLP切回时，必要的无效head/depth兼容字段由程序显式加入changes；不留给LLM自由规划。JSON库哈希由新配置固定，初始化检查，修改库需另建实验。它不代替整体源码版本管理。

Reflector保留时，诊断作为实现上下文、事后反思照常记录；它不能重排规则或改写固定目标。关闭时原始记录可作为实现上下文。这意味着人工建议版对Reflector的研究问题主要是实现/修正帮助，而不是LLM选择建议的能力。

人工建议版比普通LLM提案路线的范围约束更明确：所有变化已由外部目标指定，不允许LLM自行决定配套结构。因而这是一项“人工预设提案策略 vs LLM规划策略”的基线，不能把全部差异纯粹归因于某个角色是否存在。LLM调用次数、token和检查开销也不完全相同，需单列报告。

## 独立配置与运行

旧 `Ablation-NoProposer-*` 和 `Ablation-NoReflector-NoProposer-*` 配置仍为 `modifier_direct`，可继续复现实有自主规划定义。旧快照缺少新模式字段时仍解释为modifier_direct；不会自动迁移。中途改成人工建议模式会被配置一致性检查拒绝。

新增：

- `configs/Ablation-Human-NoProposer-4bars-30coverage-seed42.json`：保留Reflector，程序选择人工建议，Modifier实现。
- `configs/Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42.json`：同时关闭Reflector，程序选择人工建议，Modifier实现。

两者均继承冻结Bv2 seed42的其余设置，独立运行目录与JSON同名。初始数据相同，固定1至250轮。MATLAB示例：

```matlab
cd('D:\Topo_Nader\web_plate');
addpath(fullfile(pwd, 'matlab'));
setenv('TOPONADER_PYTHON', 'D:\Topo_Nader\.venv\Scripts\python.exe');
name = 'Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42';
TopoNader_main(fullfile(pwd, 'data', '300ntrain_train_step1.mat'), ...
    1, 250, fullfile(pwd, 'runs', name), ...
    fullfile(pwd, 'configs', [name '.json']));
```

另一组把name改成 `Ablation-Human-NoProposer-4bars-30coverage-seed42`。考虑既往显存不足，建议顺序运行。不要从旧消融累计数据续接；若旧实验仍在运行，可按旧定义继续，但不能把两种策略的结果合并。

## 研究解释与检查

建议库在已观察过既往实验（包括BatchNorm变体表现）后建立，虽然纳入双向/对照选项，也仍是事后构建的探索性设计，不能声称对已看过的数据进行了预注册独立验证。冻结此版本再评估新运行；若简化基线优于完整框架，也应如实保留结果。

通过75项单元/集成测试，包括4/6杆、确定性置乱、输入维度和参数约束、RNG隔离、禁止额外结构修改、库哈希校验、旧模式恢复与拒绝原目录切换模式，以及CPU准备—反馈闭环。没有做真实LLM语义遵循度或FE收益验证。
