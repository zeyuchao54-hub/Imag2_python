# PointToCAD

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

工业级点云逆向工程与几何质检流水线（Industrial Point Cloud Reverse Engineering & Quality Inspection Pipeline）。

输入 3D 扫描点云（`.ply` / `.pcd`），可选输入 STL 名义 CAD 模型；输出物理尺寸的融合点云、结构化 JSON 检验报告、ICP 配准结果、偏差色谱及 GD&T 几何公差分析。

---

## 功能特性

- **完整闭环**：预处理 → RANSAC 平面提取 → 碎片融合 → 拓扑建图 → 比例尺标定 → 3-2-1 基准对齐 → ICP 配准 → 偏差分析 → GD&T 公差判定
- **Auto-Scale**：基于 Sim3 (Weighted Umeyama) 的全局比例尺自动估计，无需人工标定即可对齐扫描与 CAD
- **Datum-Weighted ICP**：在 Fine ICP 阶段对 3-2-1 基准面区域施加高权重，提升对齐精度
- **缺失面防护**：自动检测并剔除疑似桌面/夹具的超大平行平面，防止缺失面扫描导致基准面绑定错误
- **可复现性保障**：固定 OpenMP 线程数规避 Open3D 0.19 RANSAC 竞态，全随机源播种，同输入同输出
- **自适应阈值**：距离类容差按场景尺度自动计算，避免硬编码 mm 值在不同尺寸零件间失效

## 安装

```bash
git clone https://github.com/zeyuchao54-hub/Imag2_python.git
cd Imag2_python
python -m venv .venv310
.venv310\Scripts\python.exe -m pip install -r requirements.txt
```

依赖：Python ≥3.10, NumPy, Open3D ≥0.18, SciPy, Trimesh ≥4.0

## 快速开始

### 仅几何流水线（无 CAD）

```bash
.venv310\Scripts\python.exe main.py -i point_cloud.ply --batch --scale_factor 1.0 -o outputs
```

### 带 CAD 的完整质检流水线

```bash
.venv310\Scripts\python.exe main.py -i point_cloud.ply \
  --stl 60_80_100Cuboid.STL --icp --batch \
  --tolerance 2.0 --angle_tolerance 2.0 \
  -o outputs
```

输出产物示例：

```
outputs/
├── fused.ply              # 融合后的物理单位点云
├── report.json            # 平面/顶点几何尺寸报表
├── registration.json      # ICP 配准结果（变换矩阵、fitness、RMSE）
├── deviation.json         # 偏差统计量
├── deviation_map.ply      # 扫描侧偏差色谱点云
├── tolerance.json         # GD&T 公差判定结果
├── aligned_cad.ply        # 对齐后的 CAD 点云
└── inspection_summary.json # 综合检测报告
```

## 流水线架构

`main.py` 中的 `IndustrialPipeline` 按以下阶段串行执行：

| 阶段 | 模块 | 说明 |
|------|------|------|
| 1 | `preprocess` | 降采样、统计去噪、法向量估计 |
| 1.5 | `cad_cropper` | （可选）CAD 引导的扫描区域裁剪 |
| 2 | `detector` | RANSAC 多平面几何图元提取 |
| 3 | `merger` | 共面碎片融合 + 桌面平面剔除 |
| 4 | `graph` / `geometry` | 拓扑关系图构建与 CAD 角点求解 |
| 4.5 | `calibration` | 物理比例尺标定（交互式 / CLI 覆盖 / Auto-Scale） |
| 4.8 | `aligner` | 3-2-1 工业基准坐标系对齐 |
| 5 | `report` | 导出 PLY / JSON /（可选）STEP |
| 5.5 | `registration` / `deviation` / `tolerance` | ICP 配准、偏差分析、GD&T 公差计算 |
| 6 | `visualization` | Open3D 可视化（非 batch 模式） |
| 7 | `report` | 综合检测报告 |

## 测试

```bash
# 全部测试
.venv310\Scripts\python.exe -m unittest discover -s tests -v

# 单个测试
.venv310\Scripts\python.exe -m unittest tests.test_icp_solve -v
```

> 注意：每个测试文件首行必须先 `import _bootstrap`，以在 `import open3d` 之前固定 `OMP_NUM_THREADS=1`，保证可复现性。

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | 必填 | 输入点云文件 `.ply` 或 `.pcd` |
| `--stl` | `None` | 输入 STL 名义模型（启用 ICP 分支） |
| `--icp` | `False` | 执行 CAD → Scan 的 ICP 配准与偏差分析 |
| `--scale_factor` | `None` | 物理尺度缩放因子（batch 无 STL 时建议提供） |
| `--primary_plane_id` | `1` | 3-2-1 对齐主基准面 ID（对齐到 Z=0） |
| `--secondary_plane_id` | `4` | 3-2-1 对齐次基准面 ID（对齐到 X=0） |
| `--tolerance` | `2.0` | 长度类公差判定阈值（mm） |
| `--angle_tolerance` | `2.0` | 角度类公差判定阈值（°） |
| `--seed` | `42` | 全局随机种子，固定后同输入必同输出 |
| `--batch` | `False` | 批处理/静默模式（跳过 3D 可视化与交互标定） |

完整参数列表请运行 `.venv310\Scripts\python.exe main.py --help`。

## 许可

[MIT](LICENSE)
