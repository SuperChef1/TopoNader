# fluid40_16：独立流道版 TopoNader

这是从 `web_plate` 框架独立复制并适配的版本，从 MATLAB 启动。两套任务不共享源码；共用根目录的 Python 环境，不需要安装新的 editable 包或 PowerShell 启动脚本。

## 目录

- `reference/`：原始流道脚本、数据、网络权重及历史结果，原文件内容不变。WAE 和旧 PSO 只作参考，不会被新主流程导入。
- `topo_nader/`：流道框架，保留训练、MC-dropout、架构调整、反馈、继承/回滚及报告流程。
- `matlab/`：新的 MATLAB 入口、独立 COMSOL 求解器、原模型的工作副本。
- `data/initial_641.mat`：原始 641 样本的副本，未重算标签；来源与检查见同目录 provenance JSON。
- `configs/`：正式方法默认值、固定架构对照、小规模检查及离线架构调整检查。
- `runs/`：由用户启动实验后产生的请求、结果、模型、数据与日志。
- `tests/`：Python 自动检查及 MATLAB 连通性对照数据。

## 固定的问题定义

输入为 `(N,40,16)`，0 是流体，1 是扣除的固体障碍。MATLAB 单个设计 `design(:)` 的第 j 项对应 `sqj`：`sq1` 是左上角，`sq40` 是第一行右端，`sq41` 是第二行左端。Python 的 C-order 展平用于代理输入；MATLAB 按数组轴读取，再用 column-major 展平选择几何，二者不应混用。

目标最大化 `1 / 入口平均压力`，出口零表压。材料与边界条件沿用 `fluid.mph`；不施加体积比例约束，固体比例只作统计。兼容字段 `coverage_ratios` / `coverage_actual_*` 在本任务中表示固体面积比例。

**连通性规则保持原样**：左边前 4 行、右边前 8 行，各取第一个开放单元，再检查四邻接路径；不扩展端口，不修复堵塞。Python 直接使用逐字复制的 `reference/utils.py`，MATLAB 提供等价二值网格实现。该旧规则存在已知误拒绝，属于本次保留的实验设定。

## 搜索与架构

采用已有连续 EPSO 更新公式，评分前先以 `>=0.5` 二值化并检查旧连通性；不合法候选给予固定低分。最终候选重新预测，按二值拓扑精确去重，以 Hamming 距离作多样性选择。采用有界、去重的搜索档案，避免保存全部 640 维粒子历史占用大量内存。

利用通道从真实最佳样本进行多精英搜索：初始化采用位翻转/局部块翻转，并以 `local_hamming_limit` 限制每个局部搜索。探索通道采用全局 UCB。堵塞和重复只拒绝、不修复；初始化及补样混合多个障碍密度和已有样本变异。补样次数有限，凑不齐准确批量则在 FE 前报错。

初始代理为 `640 -> 512 -> 256 -> 128 -> latent 30 -> MLP`，可通过 `initial_architecture` 配置。首版架构空间支持 MLP/residual MLP 编码器和 MLP predictor，不使用杆件 Transformer。架构调整保留 Reflector/Proposer/Modifier 与原有变更范围限制；网络从当前数据重新训练，不继承权重。超参数是迁移初值，尚未经过流道优化效果验证。

## 从 MATLAB 启动

在连接 COMSOL LiveLink 的 MATLAB 会话执行：

```matlab
task = 'D:\Topo_Nader\fluid40_16';
addpath(fullfile(task,'matlab'),'-begin');
setenv('TOPONADER_PYTHON','D:\Topo_Nader\.venv\Scripts\python.exe');

% 轻量检查，不训练、不调用 COMSOL。
TopoNader_fluid_check();

% 单独验证旧标签与新求解器，需要 COMSOL，消耗 5 次真实求解。
% 结果目录必须不存在；5% 仅为初步核对阈值，不是精度认证。
TopoNader_fluid_validate_reference(fullfile(task,'runs','reference_check_001'));

% 小规模闭环：每轮 4 个设计，最多 2 轮。需要 COMSOL。
TopoNader_fluid_main('',1,2,'',fullfile(task,'configs','smoke.json'));
```

入口固定工作目录并核验 Python 导入来自 `fluid40_16/topo_nader`，避免误用 `web_plate`。不需要把其他任务目录加入 Python 路径。

配置含义：

| 配置 | 用途 |
|---|---|
| `smoke.json` | 固定小网络、2 个训练 epoch、2 次 PSO 迭代、每轮 4 个 FE 设计，检查流程 |
| `smoke_audit.json` | 与 smoke 同预算，用随机架构提案检查调整/反馈流程，不调用 LLM |
| `baseline_fixed.json` | 固定初始架构，保留不确定性与多样性，用于对照 |
| `default.json` | 启用事件触发架构调整，默认每轮 20 个 FE 设计；触发时需要配置现有框架使用的 LLM 凭据 |

正式运行轮数由 `TopoNader_fluid_main` 的 `max_iteration` 显式给出。`smoke` 配置用于程序检查，不适合评价优化性能。不要在同一 run 目录混用配置或模型。

## 恢复与诊断

开始 run 时冻结 COMSOL 模型，源模型改变会拒绝复用 run。MATLAB 逐样本保存 `fe_result.mat.checkpoint.mat`；中断后重新调用同一轮，可复用已有请求并从已保存进度继续求解，不自动重试已完成的失败样本。完整结果已存在时直接进入 feedback。

第 k 轮已完成 feedback 后，续跑使用：

```matlab
runDir = fullfile(task,'runs','fluid40-16-smoke');
TopoNader_fluid_main(fullfile(runDir,'datasets','dataset_0003.mat'), ...
    3,4,runDir,fullfile(task,'configs','smoke.json'));
```

例子仅适用于 state 中 `next_iteration=3`。未完成 feedback 的轮次使用原轮次数据和编号。不要同时运行两个进程写入同一个 run。

每轮结果包含原始平均压力、倒数目标、状态、错误堆栈和耗时；模型入口/出口选择会重新读取并核验坐标。失败返回 NaN，反馈不会当成优解加入训练集。`fluid_diagnostics.json`、`iteration_summary.json` 和累计 CSV/JSON 记录固体比例、候选拒绝/重复、FE 成功率、压力及架构决策。若整批失败，反馈明确报错，先检查原始 FE 结果。

## 验证范围

Python 测试可从本目录运行：

```powershell
..\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

模拟 oracle 仅验证流程，不能作为真实流体求解结果。真实 COMSOL 兼容性、参考标签重算和优化效果，需要上述用户显式启动的验证；现有 641 个标签的物理正确性不因接口迁移自动得到证明。
