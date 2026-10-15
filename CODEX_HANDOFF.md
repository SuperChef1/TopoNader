# TopoNader Codex 迁移交接

更新时间：2026-10-12（Asia/Shanghai）

这份文件用于把当前电脑上关于 TopoNader 的关键对话背景、代码现状、实验状态和待办事项交给另一台电脑上的 Codex。新 Codex 不应只凭本文件修改代码；应先核对仓库、配置和运行数据，因为 `runs/` 与外部数据不在 Git 中。

## 给新电脑 Codex 的首条指令

建议在新电脑打开 `Topo_Nader` 项目后发送：

> 请完整阅读 `CODEX_HANDOFF.md`、`README.md`、`configs/default.json` 和 `configs/formal_original_50iterations_elite_local.json`。先核对 Git 提交、配置解析结果、运行目录及数据是否齐全，再汇报当前状态。不要直接改代码。请区分已核实事实、推测和建议，并特别检查外层实验轮数与 EPSO 内部迭代数是否被混淆。

## 1. 用户的研究目标

TopoNader 是一个以代理模型、EPSO、COMSOL 有限元反馈和可选架构代理组成的拓扑优化框架。用户希望：

- 杆件数 `num_bars`、材料覆盖率限制、随机种子和实验名均可通过 JSON 配置；
- `experiment_name` 决定独立结果目录，避免不同实验混写；
- `latent_dim` 在启用架构自适应时可由架构代理修改；
- 覆盖率上限保留，覆盖率下限先有开关但默认关闭；
- 正式实验每轮选择 10 个 exploitation 样本和 10 个 exploration 样本；
- 比较不同杆件数、覆盖率和随机种子，以论证框架有效性；
- 改进 EPSO 对真实优秀样本的利用，但尽量不改变原 EPSO 的核心更新机制；
- 记录覆盖率修复、去重和修复前后排序变化，判断“PSO 认为好”和“最终送入 COMSOL 的结构”是否发生了实质偏移。

## 2. 仓库状态（已核实）

- 本机仓库：`D:\Nader\Topo_Nader`
- Git 远端：`git@github.com:SuperChef1/TopoNader.git`
- 分支：`main`
- 交接文件创建前 HEAD：`fa292aa Add elite-local EPSO search and diagnostics`
- 该提交已在 `origin/main`（此前已核实）。
- 创建本文件前工作树无其他改动；本文件创建后会成为新的未提交文件，除非之后另行提交。
- Python 要求：3.10 以上；原电脑实际使用 Python 3.11 环境。

不要在新电脑直接复制 `.venv`，应重新建立虚拟环境。

## 3. 关键配置语义

默认配置位于 `configs/default.json`：

- `experiment_name`: `toponader-four-bar-beam-elite-local`
- `random_seed`: `3`
- `data.num_bars`: `4`
- `data.parameters_per_bar`: `4`
- 因而当前默认输入维数为 `4 * 4 = 16`
- `training.epochs`: `300`
- `training.quick_epochs`: `30`
- `optimization.particles`: `300`
- `optimization.iterations`: `500`
- `optimization.fe_batch_size`: `20`
- `optimization.exploitation_count`: `10`
- `optimization.exploration_count`: `10`
- 覆盖率上限启用：`upper_bound = 0.30`
- 上限容差：`upper_tolerance = 0.02`
- 覆盖率下限开关存在，但默认 `lower_bound_enabled = false`
- `elite_local_search.enabled = true`

重要：`optimization.iterations = 500` 是每次 EPSO 搜索的内部迭代数，不是整个 COMSOL 实验的外层轮数。外层 50 轮由 MATLAB 调用中的 `max_iteration=50` 控制，例如：

```matlab
TopoNader_main(initial_dataset, 1, 50, run_dir, config_file)
```

正式固定原始架构实验配置为：

```text
configs/formal_original_50iterations_elite_local.json
```

它继承 `default.json`，并设置：

- `experiment_name = formal-original-50iterations-elite-local-10x10`
- `architecture_adaptation = false`
- `agents.enabled = false`
- uncertainty 和 diversity sampling 保持启用

因此这次正式运行是“固定原始代理架构 + elite-local exploitation + global UCB exploration”，不是架构代理自适应实验。`audits_performed = 0` 和 `architecture_changes = 0` 在这个配置下是预期行为，不代表触发器失效。

## 4. 架构、维数及重复架构判断

当前基线架构位于 `topo_nader/architecture/spec.py`：

- 默认 `input_dim = 16`
- 默认 `latent_dim = 12`
- encoder 为 MLP：`1024 -> 512 -> 256 -> 12`
- predictor 默认为 Transformer

“16 维一定压缩到 12 维”只对当前固定基线成立，不是框架硬编码的不变量：

- 输入维数由 `num_bars * parameters_per_bar` 解析；Python 管线不限于 4 根杆；
- 启用架构自适应后，代理可在 `architecture.min_latent_dim=2` 到 `max_latent_dim=512` 内提出 `latent_dim`；
- 当前正式配置关闭架构自适应，所以该运行会一直使用 `latent_dim=12`。

架构 ID 已做语义化规范：

- `name` 和 `metadata` 不参与架构 ID；
- 当 `predictor_type == "mlp"` 时，不起作用的 `predictor_depth` 和 `attention_heads` 不参与 ID；
- 因此仅这些无效字段不同的同一 MLP 不会再被当作不同架构。

但这不是一般意义上的神经网络图同构判定。例如，通过隐藏神经元置换得到的函数等价网络仍可能拥有不同描述；当前去重目标是配置级语义重复，而不是证明函数完全等价。

## 5. 杆件数接口的边界

Python 侧可以通过 `data.num_bars` 使用任意正整数杆件数，输入维数随之变化。MATLAB/COMSOL 侧有额外前提：`.mph` 模型必须含有 `r1` 到 `rN` 几何特征。

因此“配置可改杆件数”不等于“任意杆件数都能直接运行”。新实验前必须确认：

- COMSOL 模型已有对应数量的杆件特征；
- `func_inputs.m`、`TopoNader_fe_batch.m` 和 `.mph` 的参数约定一致；
- 初始 MAT 数据的输入列数等于 `num_bars * parameters_per_bar`。

## 6. 当前 EPSO 改动

用户此前认为仅用少量初始粒子、进行单一全局搜索，未充分利用真实优秀样本。当前实现把 exploitation 改为 elite-local 搜索：

- 从真实 FE 数据中选前 5 个 elite；
- 围绕 5 个 elite 分别运行局部 EPSO；
- 300 个粒子按 `120/60/48/36/36` 分配；
- 最终 10 个 exploitation 样本按 `4/2/2/1/1` 分配；
- 局部初始化优先采用单参数扰动、同杆双参数扰动及较小/中等半径扰动；
- exploration 仍是独立的全局 UCB-EPSO。

这次修改改变了初始化和搜索边界，但没有改变 EPSO 的惯性、认知项、社会项、personal best、global best、变异及停滞重置方程。因此称为在原 EPSO 上增加 elite-local 搜索策略更准确，而不是换成另一套底层优化算法。

## 7. 覆盖率修复与适应度的当前决定

当前没有把完整的“离散化 + 覆盖率修复 + 重新预测”嵌入每个 PSO 粒子的每次适应度计算。原因是用户认为计算和工程复杂度较高，希望先观察正常优化是否逐步进入可行区域。

当前流程是：

```text
EPSO 连续空间搜索
  -> 候选池
  -> 离散化
  -> 覆盖率修复
  -> 去重
  -> 对最终结构重新预测
  -> 根据最终结构重新排序/选择
  -> COMSOL
```

这样避免日志预测值与 COMSOL 实际结构不一致，但并不消除“PSO 原始最优点经修复后变成另一个结构”的语义问题。为此已增加统计，而不是假装问题不存在：

- 覆盖率修复比例；
- 修复前后宽度变化；
- 归一化投影距离；
- 历史重复和修复后重复；
- 唯一候选产出率；
- 修复前后 Spearman、pairwise agreement、top-10 overlap；
- 修复前后 priority 变化。

如果这些指标长期很差，下一步应优先考虑轻量可行域参数化、软惩罚或只对候选池执行约束感知重排，而不是立刻在所有粒子每代执行昂贵的完整投影。

## 8. 当前正式实验状态（已核实）

运行目录：

```text
runs/formal-original-50iterations-elite-local-10x10
```

截至 2026-10-12：

- 已完成 feedback 的轮数：39
- `state.next_iteration`: 40
- 第 40 轮已经完成 `prepare`
- `iteration_0040/fe_request.mat`、manifest 和模型 checkpoint 已存在
- 第 40 轮尚无 `fe_result.mat`，也尚未写入 cumulative summary
- 初始 FE 样本数：300
- 新增 FE 调用：780（39 × 20）
- 累计 FE 调用：1080
- 新增有效 FE 结果：780
- 新增失败 FE 结果：0
- 当前报表 best performance：`0.8486117156245405`
- 累计覆盖率最小值：`0.0012800819252432156`
- 累计覆盖率最大值：`0.2997951868919611`
- 累计覆盖率均值：`0.2872690929142543`
- 累计覆盖率标准差：`0.04515198779552137`

需要纠正一个来自较早实验阶段的旧判断：当前正式运行不能再概括为“始终没得到好样本”。`iterations.csv` 中第 1 轮的 `best_performance_so_far` 为约 `0.352859`，第 37 轮达到 `0.848612`。这说明 elite-local 版本至少在当前运行中找到了显著更高的结果。不过，它是否构成严谨的框架有效性证据仍需多随机种子、不同杆件数/覆盖率和基线对照，不能凭单次运行下结论。

## 9. 当前仍然明显的风险

第 40 轮 manifest 显示搜索仍有严重的“先出界、再修复”倾向：

- 5 个局部 swarm 的候选覆盖率修复比例约为 60.5% 到 84.6%；
- 全局 exploration 候选修复比例约为 96.5%；
- 局部候选修复后的结构重复比例约为 50% 到 74%；
- 全局候选修复前后排序的 Spearman 约为 `0.0073`，top-10 overlap 为 `0.0`；
- 第 40 轮 batch duplicate 拒绝计数为 581。

这表明当前改动提高了真实 elite 的利用率，但搜索空间与可行域仍不匹配。不能因为最终覆盖率都接近 0.30 就认为优化方向天然正确；大量粒子先跑到上限外，再被压回边界。

第 40 轮固定代理模型的验证指标也并不理想：

- R2 约 `-0.0995`
- pairwise accuracy 约 `0.6606`
- Kendall tau 约 `0.3213`
- top-k recall 为 `0`
- top-k regret 约 `0.1422`

因此高 FE 最佳值与代理模型整体精度差可以同时成立。一个可能解释是局部搜索偶然/局部有效，但代理对全局和 top-k 排序仍不稳定。后续评估必须同时看 FE 改进、排序指标、覆盖率修复比例和重复率。

## 10. 迁移到新电脑必须额外复制的内容

Git 仓库只迁移代码，不会自动迁移以下内容：

1. `runs/`：被 `.gitignore` 忽略。若要续跑第 40 轮，必须完整复制上述正式运行目录，包括 datasets、state、iteration 目录和 checkpoint。
2. 初始外部数据：

   ```text
   D:\Nader\test37_4Beam_Best\test37_4Beam_Best\data_g\300ntrain_train_step1.mat
   ```

3. COMSOL 安装、许可证和 MATLAB/COMSOL LiveLink 环境。
4. 本地环境变量和 API 密钥。不要把密钥写进本文件或 Git。
5. 若新电脑需要独立复现实验，应保留原运行目录不动，使用新的 `experiment_name` 创建新目录。

`matlab/fuban_Beam.mph` 当前在 Git 中，但仍应在新电脑确认它与安装的 COMSOL 版本兼容。

## 11. 新电脑初始化步骤

```powershell
git clone git@github.com:SuperChef1/TopoNader.git
cd TopoNader
git checkout main

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\run_TopoNader.ps1 --config configs\formal_original_50iterations_elite_local.json validate-config
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

如 MATLAB 调用的不是项目 `.venv`，设置：

```powershell
$env:TOPONADER_PYTHON = '<新电脑路径>\TopoNader\.venv\Scripts\python.exe'
```

长期使用时可改成用户环境变量，但先用进程级变量验证，避免污染其他 Python 项目。

## 12. 续跑与重跑的区别

### 续跑现有正式实验

前提是完整复制 `runs/formal-original-50iterations-elite-local-10x10`。当前第 40 轮处于“已 prepare、未 feedback”状态。最保守的处理是：

1. 使用现有 `iteration_0040/fe_request.mat` 完成 COMSOL 计算；
2. 将结果保存为 `iteration_0040/fe_result.mat`；
3. 对第 40 轮执行 feedback；
4. 再从第 41 轮运行至第 50 轮。

不要在未检查状态前直接从第 1 轮或第 40 轮重新调用整个 MATLAB 循环，否则可能覆盖已准备的第 40 轮文件或造成重复 FE 计算。

### 在新电脑重新做独立实验

修改配置中的：

- `experiment_name`
- `random_seed`
- `data.num_bars`
- `optimization.coverage.upper_bound`
- 必要时启用/设置覆盖率下限

每一组参数应有唯一实验名。不要复用已有运行目录，因为初始化逻辑会拒绝不一致的解析配置，强行混用也会破坏实验可追溯性。

## 13. 建议的实验矩阵

正式结论至少应包含：

- 多个 `random_seed`，避免把一次幸运搜索当成方法效果；
- 多种杆件数，但必须先准备对应 COMSOL 模型与数据；
- 多个覆盖率上限；
- 固定原始代理 + mean EPSO；
- 固定原始代理 + uncertainty/diversity；
- 固定原始代理 + elite-local；
- 完整的 event-triggered 架构自适应方法。

比较时应固定 FE 预算，而不是只固定外层轮数。杆件数变化会改变搜索维度，直接比较相同粒子数/内部迭代数可能对高维实验不公平，需要同时记录代理训练时间、EPSO 时间、COMSOL 调用数和失败数。

## 14. 新 Codex 接手时应先回答的问题

在任何新修改前，先核实并向用户汇报：

1. 当前 Git HEAD 是否仍为预期提交，是否有未提交改动？
2. 正式运行目录是否完整迁移，`state.next_iteration` 是多少？
3. 第 40 轮是否已有 `fe_result.mat`，是否已 feedback？
4. 新电脑的 COMSOL、MATLAB、Python、PyTorch 和 GPU 是否可用？
5. 外部初始数据路径是否已更新？
6. 用户要“续跑原实验”还是“新建独立复现实验”？
7. 下一步目标是诊断现有 50 轮结果，还是开启架构自适应/多参数对照？

## 15. 不应遗忘的原则

- 不要把代理预测好等同于 FE 真实好。
- 不要把最终覆盖率合规等同于 PSO 已学会在可行域内搜索。
- 不要把一次实验的最好值当成框架有效性证明。
- 不要混淆外层 COMSOL 轮数和 EPSO 内部迭代数。
- 不要在未复制 `runs/` 的情况下声称可以无缝续跑。
- 不要在固定架构配置下期待 `latent_dim` 或网络结构自动变化。
- 所有结论优先从 `summary.json`、`iterations.csv` 和每轮 `manifest.json` 重新计算验证。

