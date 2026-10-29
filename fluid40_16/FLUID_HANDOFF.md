# fluid40_16 流道版 TopoNader：跨电脑交接与实验指南

整理日期：2026-10-26（Asia/Shanghai）。

本文是本次对话的决策、实现和验证摘要，不是逐字聊天记录。目的：在另一台 Windows 电脑继续开发和运行流道实验。文中 `D:\Topo_Nader` 是当前电脑路径，新电脑应替换成实际路径。

## 1. 先给接手者看的状态

- 已实现独立的 `fluid40_16` 版 TopoNader，复制腹板框架后作流道专用修改，暂不建设完全通用的任务接口。
- 用户明确要求：**保持原始连通性规则，不修正入口范围或首个开放点选择逻辑。**
- 用户明确要求：从 **MATLAB 启动整个框架**，不需要 PowerShell 启动脚本。
- `web_plate` 是独立腹板任务，**不得为适配流道而修改其内容**。其他任务可能在运行，不要停止或改动相关进程。
- 原始流道案例保留在 `fluid40_16/reference/`；新版本不调用原 WAE 分支。
- Python 自动测试最终 9 项通过；MATLAB 轻量检查通过 122 个连通性样例和 5 个单元编号映射；5 个 MATLAB 文件静态检查无报告问题。
- **尚未运行真实 COMSOL 验证或正式流道优化。** Python 闭环测试使用模拟 oracle，不能当成真实流体结果。
- 本次整理时 `fluid40_16/runs/` 没有实际实验目录。
- 流道实现已于 2026-10-26 提交并推送，提交号 `0a86604`。新电脑可通过 git clone/pull 获取已跟踪的源码、配置和初始数据；忽略的权重及运行产物仍需另行同步。

## 2. 用户的协作偏好与已确认决定

1. 先检查错误前提、逻辑跳跃和缺失信息；不迎合用户，区分事实、推测和观点。
2. 数字和结论尽量核实，发现分歧要指出依据；主动说明影响实验的变量和代价。
3. 每个任务独立子目录，整体程序相似，针对任务修改。当前不做通用接口抽象。
4. 保持腹板源码、配置、数据和结果不变；此前目录调整已经完成。
5. 原连通性即便存在误拒绝也暂时保留，保证实验规则一致。
6. 没有单独授权前，不自动启动长时间实验、消耗真实 COMSOL 预算或运行外部 LLM 调用。

目录调整时曾新增根目录 `run_web_plate.ps1` 和 `directory_migration_manifest.json`，之后用户认为不需要；当前根目录已没有这两个文件。不要重新增加同类入口作为 MATLAB 流程的前提。

## 3. 当前项目布局

```text
D:\Topo_Nader\
├─ .git\
├─ .venv\                       # 当前电脑共用环境，新电脑需重建
├─ README.md
├─ web_plate\                   # 腹板任务，不改动
└─ fluid40_16\
   ├─ FLUID_HANDOFF.md           # 本文
   ├─ README.md                  # 流道版使用说明
   ├─ pyproject.toml
   ├─ reference\                # 原始脚本、数据、权重、模型与历史资料
   │  ├─ utils.py               # 原始连通性判断
   │  ├─ fluid.mph
   │  ├─ 10ntrain_train_step.mat
   │  └─ migration_provenance.json
   ├─ topo_nader\               # 流道独立源码
   ├─ matlab\
   │  ├─ fluid.mph
   │  ├─ TopoNader_fluid_main.m
   │  ├─ TopoNader_fluid_fe_batch.m
   │  ├─ TopoNader_fluid_legacy_isvalid.m
   │  ├─ TopoNader_fluid_check.m
   │  └─ TopoNader_fluid_validate_reference.m
   ├─ configs\
   │  ├─ default.json
   │  ├─ baseline_fixed.json
   │  ├─ smoke.json
   │  └─ smoke_audit.json
   ├─ data\
   │  ├─ initial_641.mat
   │  └─ initial_641.provenance.json
   ├─ tests\
   │  ├─ test_fluid.py
   │  ├─ legacy_connectivity_fixture.mat
   │  └─ matlab_check.log
   └─ runs\
```

不必拷贝 `.venv`、`__pycache__` 等机器相关缓存。应拷贝完整流道目录，特别是模型、MAT 数据和测试 fixture；不要只拷贝 `.py` 和 `.m`。

原始参考资料中的 `.pth` 权重受 Git ignore 规则影响，若选择通过 Git 转移，要额外核实资料是否完整；新主流程不依赖这些旧权重。

## 4. 流道问题的定义

### 4.1 设计表示

- 40×16 是设计单元数量，不是物理尺寸；共 640 个二值变量。
- `0`：流体空间；`1`：从流体域扣除的固体障碍。
- MAT 输入形状为 `(N,40,16)`，不能随意转换成其他轴顺序。
- `sq1` 位于左上角，`sq40` 是第一行右端，`sq41` 是第二行左端，`sq640` 位于右下角。
- MATLAB 单样本 `design(:)` 按列展平，直接对应 `sq1` 到 `sq640`；Python 用 C-order 展平供代理输入。两者通过原始数组轴对齐，不能直接把 Python C-order 扁平向量当作 COMSOL 方块编号。

### 4.2 模型信息与目标

以下来自此前对保存模型 XML 的解析，尚未通过此次真实求解验证：

| 项目 | 保存配置 |
|---|---|
| 设计区尺寸 | 宽 2 m，高 0.8 m |
| 单元边长 `dx` | 0.05 m |
| 两侧延伸参数 `excess` | 0.4 m |
| 外框范围 | x 从 -0.4 到 2.4，y 从 0 到 0.8 |
| 固定扣除区域 `r2` | 左下角 0.4×0.4 m，不参与优化 |
| 物理模型 | 二维、稳态、不可压缩层流，保留惯性项，关闭重力 |
| 材料密度 | 1 kg/m³ |
| 动力黏度 | 0.01 Pa·s |
| 入口 | 速度边界，保存的法向速度参数 `U0in=0.01` m/s |
| 出口 | 零表压，回流抑制开启 |
| 壁面 | 无滑移 |
| 原模型版本标记 | COMSOL 5.4.0.225 |

不应默认把材料解释成水。迁移保留模型中的物理设置。

目标：最大化 `f = 1 / 入口平均压力`。原脚本硬编码边界 1，新求解器读取入口物理特征的当前边界集合，并检查其坐标；出口也作坐标核验。预期入口在 x=-0.4、出口在 x=2.4。如果改变几何或模型，这些检查也需要复核。

不施加体积分数、固体数量或最小通道宽度约束。固体比例只记录，不作为约束。兼容字段 `coverage_ratios` 和 `coverage_actual_*` 在本任务中指固体面积比例。

### 4.3 连通性：必须保持的旧规则

1. 在 16 行×40 列的物理网格中，检查左边前 4 行，选第一个值为 0 的单元。
2. 检查右边前 8 行，选第一个值为 0 的单元。
3. 用上下左右四邻接检查所选两个点是否连通。
4. 端口检查范围完全堵塞或所选点无路径则拒绝。

Python 文件 `topo_nader/optimization/legacy_connectivity.py` 与 `reference/utils.py` 内容完全一致。MATLAB 文件是对二值网格相同规则的实现，已经用 Python 生成的 fixture 交叉验证。

已知局限：几何上可能允许但旧检查器拒绝的路径不会进入搜索可行域，例如仅第 5 行贯通；其他端口之间存在路径也不一定通过所选首对点检查。用户知道后明确要求暂不修正。**后续助手不可自行将其升级成多源多汇检查。**

## 5. 已实现的框架改动

| 模块 | 流道版行为 |
|---|---|
| `config.py` | 专用 40×16 数据约定，移除杆件参数，禁止启用覆盖率约束，支持初始架构配置 |
| `data/dataset.py` | 严格 `(N,40,16)` 二值输入；过滤无效目标；追加和保存时间顺序字段 |
| `optimization/geometry.py` | 二值化、缓存旧连通性结果、精确拓扑去重、固体比例 |
| `optimization/fluid_search.py` | 多精英利用和全局 UCB 探索；位翻转/局部块初始化及有界补样 |
| `optimization/pso.py` | 沿用连续 EPSO 更新公式；有界二值去重档案代替全部轨迹，修正记录与评分的对应时机 |
| `optimization/acquisition.py` | 可行性过滤、精确去重、Hamming 多样性和最终结构重预测 |
| `architecture/` | 640 维输入；MLP/residual MLP 编码器，首版只允许 MLP predictor |
| `agents/` | 说明流道任务和 MLP 限制，保留 Reflector/Proposer/Modifier 与变更范围检查 |
| `pipeline/engine.py` | 二值设计评分、任务专用搜索、模型冻结、请求/反馈数据匹配和流道诊断 |
| `reporting/summary.py` | 压力、固体比例、候选拒绝/重复和求解成功率 |
| MATLAB 主循环 | 固定任务目录、核验导入路径、prepare → FE → feedback、续跑 |
| MATLAB 求解器 | 改 `dif1` 障碍选择、重建几何/网格、求解、逐样本 checkpoint 与错误记录 |

评分前统一按 `>=0.5` 转为二值设计；旧连通性不通过给低分。相同二值粒子可合并推理。最终提交 COMSOL 的设计重新预测，不使用灰度设计的预测冒充二值结果。

局部搜索默认初始化翻转 1～16 个单元，部分采用最大边长 3 的块翻转；局部 Hamming 距离上限 64。探索采样的障碍密度默认混合 0.01、0.03、0.08、0.15、0.25，并结合已有精英变异。最多 4 次补样，每次默认 500 个候选。堵塞只拒绝，不开挖通道修复。候选不足规定 FE 批量时在调用 COMSOL 前报错。

保留训练、早停、MC-dropout、架构筛选、继承/回滚、断点恢复和报告主体。没有引入 WAE，也没有新增 CNN/网格 Transformer。每次训练从头开始，不继承代理权重。

## 6. 数据与配置

初始数据 `data/initial_641.mat` 是原始 `10ntrain_train_step.mat` 的逐字节副本：

- 形状：`(641,40,16)`。
- 641 个唯一设计，标签均有限且大于零，均通过旧连通性检查。
- 包含全流体设计与 640 个单障碍设计。
- 标签范围：约 0.74196697～1.94567312。
- SHA-256：`492f321ff43177c962a62aae1cee57ec8c96c120ca4c5d310fd41bd51fd6db31`。
- 标签尚未重新求解，来源顺序不能视为已知的采样时间顺序。

原 `data.mat` 有 11000 样本，`train.mat` 有 2641 样本，保留在 reference，不默认混入初始集。不要在不说明真实评价预算的情况下拿大数据集启动并与 641 样本启动的方法直接比较。

| 配置 | 作用 |
|---|---|
| `smoke.json` | 固定小网络，CPU，2 个训练 epoch，10 粒子，2 次 PSO 内部迭代，每轮 4 个 FE 设计：2 利用+2 探索，无 LLM |
| `smoke_audit.json` | 同级小预算，随机提案触发架构调整，供流程检查，不调用 LLM |
| `baseline_fixed.json` | 固定初始架构，保留不确定性和多样性，用作正式对照的起点 |
| `default.json` | 事件触发架构调整，300 粒子、500 次内部迭代，每轮 20 FE：10 利用+10 探索 |

默认初始架构：`640 → 512 → 256 → 128 → latent 30 → MLP predictor(width=128)`。默认训练 300 epoch、早停 patience 40，MC-dropout 20 次，目标做 log10 后标准化。这些都是尚待流道 pilot 验证的工程初值。

500 是每轮内部搜索迭代数，**不是外层真实求解轮数**。外层上限由 MATLAB 主入口的 `max_iteration` 指定。

## 7. 新电脑准备

### 7.1 拷贝和环境

推荐保留 `Topo_Nader/fluid40_16` 的两层结构，根目录重建 `.venv`。不要直接搬运旧 `.venv`，其中有旧路径和机器相关依赖。Python 至少 3.10，当前电脑用 Python 3.11。

在新电脑根目录运行，例如：

```powershell
cd D:\Topo_Nader
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy scipy requests
```

另外安装适合新电脑 GPU/驱动或 CPU 的 PyTorch。这里不固定 CUDA wheel：请在新电脑按实际硬件核对官方安装方式，再执行安装。`pyproject.toml` 给出了依赖下限，但没有锁定本机的全部版本。

**不要同时 editable 安装两个提供 `topo_nader` 包的任务。** 正常使用无需 pip install -e；从任务目录用 `python -m topo_nader`，MATLAB 入口会固定目录并检查导入位置。

新电脑需要 MATLAB、COMSOL 及可用 LiveLink for MATLAB。当前电脑 MATLAB 为 R2023b；新电脑版本和许可证尚未确认，原 `.mph` 是旧版模型，必须真实验证兼容性。

### 7.2 Python 检查

```powershell
cd D:\Topo_Nader\fluid40_16
..\.venv\Scripts\python.exe -B -c "import topo_nader; print(topo_nader.__file__)"
..\.venv\Scripts\python.exe -B -m topo_nader --config configs\smoke.json validate-config
..\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

导入必须来自当前电脑的 `fluid40_16/topo_nader`，不能来自 `web_plate` 或旧磁盘路径。

### 7.3 LLM 凭据

闭环 smoke 不需要密钥。启用 `default.json` 并触发架构调整时，客户端读取 `TOPONADER_LLM_API_KEY`（兼容 `OPENAI_API_KEY`），可用 `TOPONADER_LLM_MODEL_NAME`、`TOPONADER_LLM_BASE_URL` 覆盖模型和地址。

不要把密钥写入源码、Markdown 或聊天。新电脑需要单独设置，不能假定旧电脑环境变量会自动迁移。

## 8. 按顺序验证真实闭环

本节命令供用户在新电脑显式运行。本次对话没有执行这些真实 FE 实验。

### 第一步：准备 MATLAB

在已经连接 COMSOL LiveLink 的 MATLAB 会话执行：

```matlab
task = 'D:\Topo_Nader\fluid40_16';
addpath(fullfile(task,'matlab'),'-begin');
setenv('TOPONADER_PYTHON', ...
    'D:\Topo_Nader\.venv\Scripts\python.exe');

which mphload
which TopoNader_fluid_main
TopoNader_fluid_check();
```

预期：`Passed 122 legacy connectivity cases and 5 sq mappings.`。这是无 COMSOL 求解的轻量检查。

### 第二步：重算 5 个参考样本

```matlab
checkDir = fullfile(task,'runs','reference_check_001');
report = TopoNader_fluid_validate_reference( ...
    checkDir, [1,2,41,240,641], 0.05);
```

目录必须不存在；函数不会覆盖原始数据。预期 5 次真实求解均成功，目标值相对旧标签误差 ≤5%，`report.passed=true`。

5% 是初步核对阈值，不是精度认证。失败时检查 `fe_result.mat` 中的 `status`、`errors`、`inlet_pressure`，以及 `comparison.mat`；先排查模型、边界、单位、网格和求解版本差异，不要直接放宽容差。未通过时先不要启动优化。

### 第三步：第一轮闭环

```matlab
cfg = fullfile(task,'configs','smoke.json');
runDir = fullfile(task,'runs','closed_loop_check_001');
TopoNader_fluid_main('', 1, 1, runDir, cfg);
```

顺序为：训练小网络 → 搜索4个新设计 → 真实求解 → 反馈 → 追加数据。

```matlab
state = jsondecode(fileread(fullfile(runDir,'state.json')));
result = load(fullfile(runDir,'iteration_0001','fe_result.mat'));
nextData = load(fullfile(runDir,'datasets','dataset_0002.mat'));

state.next_iteration
result.status
result.outputs
size(nextData.inputsX,1)
```

预期 `next_iteration=2`；4 个结果均有限且大于零；全部成功时样本数从 641 到 645。部分失败只追加成功样本，先检查错误原因，不应把失败样本伪造成优解。整批失败会报错而不完成正常反馈。

### 第四步：第二轮续跑

```matlab
TopoNader_fluid_main( ...
    fullfile(runDir,'datasets','dataset_0002.mat'), ...
    2, 2, runDir, cfg);

state = jsondecode(fileread(fullfile(runDir,'state.json')));
data3 = load(fullfile(runDir,'datasets','dataset_0003.mat'));
summary = jsondecode(fileread(fullfile(runDir,'summary.json')));

state.next_iteration
size(data3.inputsX,1)
summary.iterations_completed
```

两轮全部成功时应得到 `3`、`649`、`2`。这验证新样本确实进入下一轮。

预算：5 次参考求解 + 两轮各4次，共 13 次真实求解尝试；失败和重试也会消耗计算时间。smoke 不要求性能提升，它仅验证真实闭环，不足以证明优化有效。

## 9. 结果文件与续跑原则

每个 run 包含冻结模型 `runtime/fluid.mph`、配置快照、`state.json`、累计报告和逐轮目录。

| 文件 | 内容 |
|---|---|
| `iteration_NNNN/fe_request.mat` | 最终二值设计、预测均值、不确定性、来源及比例 |
| `iteration_NNNN/fe_result.mat` | 真实目标、平均压力、状态、错误和耗时 |
| `fe_result.mat.checkpoint.mat` | 逐样本保存的进度 |
| `datasets/dataset_NNNN.mat` | 排序展示字段及时间顺序 `inputsX/outputsX` |
| `manifest.json` | 数据哈希、架构、搜索与候选诊断 |
| `feedback.json` | 反馈指标与继承/回滚决策 |
| `fluid_diagnostics.json` | 流道压力、比例、失败信息和搜索拒绝统计 |
| `iteration_summary.json` | 单轮报告 |
| `iterations.csv/jsonl`、`summary.json` | 累计报告 |

候选拒绝统计针对最终候选池，不是所有 PSO 粒子评价次数。真实预算主要按 FE 请求累计。

续跑必须以 `state.next_iteration` 为准，传入对应轮次的数据。未完成 FE 的同一轮会复用请求并从 checkpoint 继续；完整 FE 结果存在时直接反馈，不重新求解。不要同时写同一 run，不要同 run 换配置或换模型。

跨电脑恢复已有 run 还要额外检查绝对路径：配置快照中的服务设置、manifest、模型 provenance、FE checkpoint 的 `model_file` 等可能带旧机器路径；checkpoint 会比较模型路径。**目前跨电脑迁移整个旧 run 的自动路径重映射尚未验证。** 本次推荐先在新电脑建立全新验证 run，不要批量改写历史记录。

## 10. 本机验证记录与剩余限制

已完成：

- Python 9 项测试最终通过，覆盖固定架构与随机架构调整流程、模拟反馈与续跑、数据/结果匹配保护、精确去重、批量不足、配置限制和档案评分对应关系。
- MATLAB 实际批处理轻量检查：122 个 Python fixture 的连通性结果一致、5 个 sq 编号一致。
- 5 个 MATLAB 文件通过本机 mlint 静态检查。
- 641 个初始样本检查通过，原模型和初始数据副本字节一致。
- 96 个腹板源码/配置/文档文件哈希对照无差异；本次没有修改腹板任务内容。

尚未完成：

- 新求解器在真实 COMSOL 中运行及旧标签重算。
- 真实 FE 驱动的两轮 smoke。
- 真实 LLM 提案流程和正式预算下的优化效果。
- 网格收敛、正式超参数校准和多随机种子对比。

本机测试中曾出现 Windows 内存/页面文件不足导致 PyTorch CUDA DLL 加载失败，随后完整测试通过；期间未停止已有 MATLAB/COMSOL 进程。新电脑若出现同类错误，先核查内存、页面文件和 PyTorch 安装，不要直接归因于流道逻辑，也不要随意停止其他任务。

真正进入正式实验前，需固定初始数据、连通性规则、FE 预算和随机种子，再比较固定架构与自适应方法；不要把同时更改约束或数据量造成的差异归因于 TopoNader。

## 11. 在新对话中可直接粘贴的续接说明

> 请先阅读项目中的 `fluid40_16/FLUID_HANDOFF.md` 和 `fluid40_16/README.md`，并只读检查新电脑环境和当前代码。我们已经把 TopoNader 分成独立的 web_plate 和 fluid40_16 两个任务。不要修改 web_plate，不要做通用接口重构。fluid40_16 已有独立实现，从 MATLAB 启动，原始资料在 reference。必须保留原 utils.py 连通性规则（左4行、右8行，各取首个开放点），不能自行修正或增加通道修复。当前通过的是9项Python测试和MATLAB轻量检查，尚未验证真实COMSOL。请先确认文件转移完整、Python导入位置、MATLAB/COMSOL LiveLink及模型兼容性，然后按交接文档带我完成5个参考样本重算和两轮smoke闭环。不要自动启动长时间正式实验或外部LLM调用；遇到问题先基于日志定位，并区分实际证据与推测。路径请根据新电脑实际位置调整。

## 12. 版本溯源提示

整理时根仓库 HEAD：`16e5f460de94ed6cafd8225de4634bb0202ff33c`。这个提交号本身不包含尚未跟踪的流道实现。框架复制时的提交和腹板源文件哈希保存在 `reference/migration_provenance.json`，用于了解来源，不表示流道代码已经提交。

拷贝 Markdown 只会转移上下文，不会转移程序、模型或数据。新电脑必须同时取得完整 `fluid40_16` 目录。
