# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PointToCAD: 工业级点云逆向工程与质检流水线 (Python, 纯 CPU)。输入 3D 扫描点云 (.ply/.pcd)，可选输入 STL 名义模型；输出物理尺寸的融合点云、JSON 几何报表、ICP 配准/偏差/GD&T 公差报告。代码、日志、提交信息均为中文，保持一致。

## 环境与常用命令

依赖见 `requirements.txt` (numpy / open3d / scipy / trimesh)。**工作虚拟环境是 `.venv310`** (open3d 0.19.0)；根目录的 `.venv` 未装依赖。

```bash
# 运行完整流水线 (batch 模式无交互，推荐用于验证改动)
.venv310/Scripts/python.exe main.py -i point_cloud.ply --stl 60_80_100Cuboid.STL --icp --batch -o outputs_test

# 仅几何流水线 (无 STL/ICP)
.venv310/Scripts/python.exe main.py -i point_cloud.ply --batch --scale_factor 1.0

# 全部测试 (unittest, 无 pytest 配置)
.venv310/Scripts/python.exe -m unittest discover -s tests -v

# 单个测试文件 / 单个用例
.venv310/Scripts/python.exe -m unittest tests.test_icp_solve -v
.venv310/Scripts/python.exe -m unittest tests.test_icp_solve.TestRankDeficientSolve.test_min_norm_solution -v
```

注意: tests/ 无 `__init__.py`；`discover -s tests` 能跑通是因为每个测试文件都先 `import _bootstrap`。

## 架构 (main.py 是唯一编排者)

`IndustrialPipeline.run()` 按编号阶段串行执行，每个阶段委托给一个独立模块。阶段编号在日志中以 `[N/7]` 打印，改动时保持编号一致 (参见提交 "统一流水线阶段编号"):

1. **预处理** `preprocess.PointCloudPreprocessor` → 降采样/去噪/法向
1.5. (可选) **CAD 引导裁剪** `cad_cropper.crop_scan_to_cad_region` (需 `--cad_crop`)
2. **RANSAC 平面提取** `detector.RansacDetector` → `List[Plane]` + 残余点云
3. **共面碎片融合** `merger.PlaneMerger`；3.5 **桌面平面剔除** (缺失面防护，可用 `--keep_table_plane` 关闭)
4. **拓扑建图 + 角点求解** `graph.PlaneGraphBuilder` → `geometry.GeometryAnalyzer`
4.5. **比例尺标定** `calibration.ScaleCalibrator` / `--scale_factor` / Auto-Scale (Sim3 ICP, `--stl + --icp` 时默认启用)
4.8. **3-2-1 基准对齐** `aligner.DatumAligner` (主基准面→Z=0，次基准面→X=0，ID 由 `--primary_plane_id/--secondary_plane_id` 指定)
5. **导出** `report.ReportGenerator` → fused.ply + report.json (+ 可选 STEP)
5.5. (需 `--stl --icp`) **ICP 配准** `registration.ICPRegistrar` → `deviation.DeviationAnalyzer` → `features.FeatureExtractor` + `tolerance.ToleranceAnalyzer` (GD&T)
6. **可视化** `visualization.Visualizer` (非 batch)
7. **综合检测报告** `ReportGenerator.export_inspection_summary`

核心数据对象是 `plane.Plane`：封装单个面片的方程 model[a,b,c,d]、法向、质心、OBB、面积、点云，并提供 `scale()` 做整体单位切换。

## 三条不可破坏的隐性契约

1. **尺度因子单点施加**：虚拟单位→物理 mm 的唯一入口是 `IndustrialPipeline._to_physical_units()`，它同时缩放 Plane 内部所有字段 (cloud/model/centroid/area)、rest_pcd 和 vertices。绝不在下游模块手工补乘 scale_factor (历史上 6 处补乘漏改的教训，见提交 c9ab8e6)。Auto-Scale 挂起时阶段 5 必须跳过缩放，由阶段 5.5 统一施加。

2. **可复现性**：
   - `OMP_NUM_THREADS=1` 必须在 `import open3d` **之前**设置 (Open3D 0.19 RANSAC 的 OpenMP 竞态会让同种子结果漂移)。main.py 顶部和 tests/_bootstrap.py 各有一份，新入口脚本和新测试文件都必须照做——**每个新测试文件的第一行业务 import 必须是 `import _bootstrap`**。
   - 所有随机源经 `_seed_everything()` 固定 (`--seed`，默认 42)。引入新随机算法时必须接入种子体系 (o3d.utility.random.seed / np.random.seed / 显式 seed 参数)，禁止用裸 `random` 或未播种的生成器。

3. **报告不谎报**：`ReportGenerator` 用 `_record_written()` 追踪本次实际写出的文件，综合报告的文件清单只列本次产物。新增导出函数时必须走 `_record_written()`，导出失败返回 None 并告警，不得假装成功。

## 其它约定

- 距离/尺寸类阈值不使用硬编码 mm 值，而是用 `utils.resolve_threshold` 按场景对角线比例自适应 (提交 5edeb95)。
- 公差阈值分长度 (`--tolerance`, mm) 与角度 (`--angle_tolerance`, deg) 两纲；`ToleranceResult` 的 `threshold`/`algorithm` 字段必须填，使 tolerance.json 自描述。
- 提交信息：中文 Conventional Commits (如 `fix(aligner): ... (#5)`)。仓库还沿用"启发/否决"式备注记录经实测否决的方案——否决的方案也要在代码注释或提交信息里说明原因，防止后人重试。
- 流水线产物 (outputs*/, logs/, *.ply) 均已 gitignore，不要提交。
