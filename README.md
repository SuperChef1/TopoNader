# TopoNader 任务目录

- `web_plate/`：现有腹板项目，保留原有程序内容、配置、数据、实验结果和备份。
- `fluid40_16/`：原 `fluid_40_16/` 流道案例。此次只整理目录，尚未适配为流道版 TopoNader。
- `.venv/`：现有共用 Python 环境。
- `.git/`：共用版本库。

## 腹板启动

在 `D:\Topo_Nader` 执行，先进入腹板项目目录：

```powershell
Set-Location .\web_plate
..\.venv\Scripts\python.exe -m topo_nader --config configs\default.json validate-config
```

配置路径相对于 `web_plate/`。在该目录运行会优先加载腹板项目源码。

腹板原有 `run_TopoNader.ps1` 内容未改动；若使用它，先将 `TOPONADER_PYTHON` 设为 `D:\Topo_Nader\.venv\Scripts\python.exe`。

MATLAB 使用 `web_plate/matlab/` 下的入口，并传入整理后的数据、配置路径。历史实验记录中原有的绝对路径保留不变；恢复历史实验前需检查相关路径。此次未启动或恢复任何优化实验。

目录迁移后，已有环境的 editable 安装可能仍记录旧源码位置；请使用上述方式启动，避免在根目录直接调用 `toponader`。
