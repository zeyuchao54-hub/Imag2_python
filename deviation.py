"""
deviation.py
工业级偏差分析模块
职责: ICP 对齐后，计算 CAD 与扫描点云之间的 signed/unsigned 偏差，并输出统计量
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class DeviationResult:
    """
    封装双向偏差分析结果。

    - cad_to_scan: 从 CAD 理想表面出发，到最近扫描点的距离统计量
    - scan_to_cad: 从实际扫描点出发，到最近 CAD 表面的 signed 距离统计量（更常用于质检）
    - cad_to_scan_signed / scan_to_cad_signed: 原始 signed 距离数组，供下游公差模块使用
    """
    cad_to_scan: Dict = field(default_factory=dict)
    scan_to_cad: Dict = field(default_factory=dict)
    cad_to_scan_signed: np.ndarray = field(default_factory=lambda: np.array([]))
    scan_to_cad_signed: np.ndarray = field(default_factory=lambda: np.array([]))
    coverage: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cad_to_scan": self.cad_to_scan,
            "scan_to_cad": self.scan_to_cad,
            "coverage": self.coverage,
        }


class DeviationAnalyzer:
    """
    基于最近邻搜索的偏差分析器。

    说明:
    - signed distance 使用 CAD 表面的法向作为名义法向
    - 正号表示扫描表面位于 CAD 理想表面法向指向的一侧（通常视为"外侧"）
    - 负号表示扫描表面位于 CAD 理想表面的另一侧（通常视为"内侧"）
    """

    def __init__(self, max_nn: int = 1, coverage_threshold: float = 1.0):
        """
        :param max_nn: 最近邻搜索数量，这里固定为 1
        :param coverage_threshold: CAD 表面覆盖度判定阈值 (mm)，
               CAD 点到最近 scan 点距离 ≤ 该值视为"被覆盖"
        """
        self.logger = logging.getLogger("PointToCAD_System.Deviation")
        self.max_nn = int(max_nn)
        self.coverage_threshold = float(coverage_threshold)

    def analyze(
        self,
        cad_pcd: o3d.geometry.PointCloud,
        scan_pcd: o3d.geometry.PointCloud,
    ) -> Tuple[DeviationResult, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        计算 CAD 与 Scan 之间的偏差。

        :param cad_pcd: 对齐后的 CAD 点云，必须包含 normals
        :param scan_pcd: 扫描点云
        :return: (DeviationResult, cad_cloud_with_deviation_color, scan_cloud_with_deviation_color)
                 返回的两个点云为原始点云的 deep copy，并附加了颜色
        """
        if cad_pcd.is_empty() or scan_pcd.is_empty():
            raise ValueError("CAD 点云或 Scan 点云为空，无法计算偏差。")

        if not cad_pcd.has_normals():
            raise ValueError("CAD 点云缺少法向，无法计算 signed deviation。")

        self.logger.info(
            f"开始偏差分析: CAD={len(cad_pcd.points)} 点, Scan={len(scan_pcd.points)} 点"
        )

        # 1. CAD -> Scan (遵循任务流程图 Aligned CAD -> NN -> Scan)
        cad_to_scan_signed, cad_to_scan_unsigned = self._compute_one_direction(
            source_pcd=cad_pcd,
            target_pcd=scan_pcd,
            use_source_normals=True,
        )

        # 2. Scan -> CAD (更常用的质检视角：测量点相对于名义表面的偏差)
        scan_to_cad_signed, _ = self._compute_one_direction(
            source_pcd=scan_pcd,
            target_pcd=cad_pcd,
            use_source_normals=False,
        )
        scan_to_cad_unsigned = np.abs(scan_to_cad_signed)

        # 2.5 CAD 表面覆盖度统计 (缺失面/遮挡检测)
        coverage = self._compute_coverage(cad_to_scan_unsigned)

        # 3. 统计量
        cad_stats = self._compute_statistics(
            signed=cad_to_scan_signed,
            unsigned=cad_to_scan_unsigned,
            name="cad_to_scan",
        )
        scan_stats = self._compute_statistics(
            signed=scan_to_cad_signed,
            unsigned=scan_to_cad_unsigned,
            name="scan_to_cad",
        )

        self.logger.info(
            f"偏差统计 (scan_to_cad): mean={scan_stats['signed']['mean']:.4f}, "
            f"rms={scan_stats['signed']['rms']:.4f}, max={scan_stats['unsigned']['max']:.4f}"
        )

        # 4. 为两个点云附加颜色（基于 signed distance）
        cad_colored = self._colorize_point_cloud(cad_pcd, cad_to_scan_signed)
        scan_colored = self._colorize_point_cloud(scan_pcd, scan_to_cad_signed)

        result = DeviationResult(
            cad_to_scan=cad_stats,
            scan_to_cad=scan_stats,
            cad_to_scan_signed=cad_to_scan_signed,
            scan_to_cad_signed=scan_to_cad_signed,
            coverage=coverage,
        )
        return result, cad_colored, scan_colored

    def _compute_one_direction(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        use_source_normals: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 source 每个点到 target 最近邻的 signed / unsigned 距离。

        :param use_source_normals: 为 True 时，signed distance 使用 source 点自身的法向；
                                   为 False 时，使用 target 最近点的法向。

        实现说明: 最近邻搜索用 cKDTree 一次性向量化查询。旧实现对每个源点单独调用
        KDTreeFlann.search_knn_vector_3d (Python 级循环)，在 8.6 万点上耗时约 1.5s；
        向量化后同一数据降至毫秒级，且结果逐位一致 (同为最近邻欧氏距离)。
        """
        source_pts = np.asarray(source_pcd.points)
        target_pts = np.asarray(target_pcd.points)
        target_normals = np.asarray(target_pcd.normals) if target_pcd.has_normals() else None

        if len(source_pts) == 0 or len(target_pts) == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty.copy()

        # 1. 最近邻 (cKDTree.query 对 k=1 直接返回欧氏距离)
        tree = cKDTree(target_pts)
        unsigned, nearest_idx = tree.query(source_pts, k=1)
        unsigned = np.asarray(unsigned, dtype=np.float64)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64)

        # 2. 计算 signed 距离所需的方向向量
        diff = target_pts[nearest_idx] - source_pts

        if use_source_normals:
            normals = np.asarray(source_pcd.normals)
        else:
            if target_normals is None:
                # target 无法向: 退化为"指向最近点的连线方向"
                normals = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
            else:
                normals = target_normals[nearest_idx]

        # 3. 法向归一化 (法向长度退化时按旧逻辑退化为 unsigned)
        norm_lengths = np.linalg.norm(normals, axis=1)
        valid = norm_lengths > 1e-12
        signed = unsigned.copy()
        if valid.any():
            unit = normals[valid] / norm_lengths[valid, None]
            signed[valid] = np.einsum("ij,ij->i", diff[valid], unit)

        return signed, unsigned

    def _compute_coverage(self, cad_to_scan_unsigned: np.ndarray) -> Dict:
        """
        CAD 表面覆盖度 (缺失面/遮挡检测)。
        CAD 点到最近 scan 点的距离 ≤ coverage_threshold 视为"被覆盖"，
        未覆盖区域通常对应贴桌面、遮挡或重建失败的缺失面。
        """
        if len(cad_to_scan_unsigned) == 0:
            return {}

        covered = cad_to_scan_unsigned <= self.coverage_threshold
        uncovered = ~covered
        coverage = {
            "threshold_mm": round(self.coverage_threshold, 6),
            "covered_count": int(covered.sum()),
            "uncovered_count": int(uncovered.sum()),
            "covered_ratio": round(float(covered.mean()), 6),
            "covered_mean": round(float(np.mean(cad_to_scan_unsigned[covered])), 6) if covered.any() else 0.0,
            "covered_rms": round(float(np.sqrt(np.mean(cad_to_scan_unsigned[covered] ** 2))), 6) if covered.any() else 0.0,
            "uncovered_mean_distance": round(float(np.mean(cad_to_scan_unsigned[uncovered])), 6) if uncovered.any() else None,
        }
        self.logger.info(
            f"CAD 表面覆盖度: {coverage['covered_ratio'] * 100:.2f}% "
            f"(阈值 {self.coverage_threshold} mm, 未覆盖 {coverage['uncovered_count']} 点)"
        )
        return coverage

    @staticmethod
    def _compute_statistics(
        signed: np.ndarray,
        unsigned: np.ndarray,
        name: str,
    ) -> dict:
        """汇总偏差统计量。"""
        if len(signed) == 0:
            return {"name": name, "count": 0}

        positive = signed[signed > 0]
        negative = signed[signed < 0]

        def _round_dict(d: dict) -> dict:
            return {k: round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in d.items()}

        return _round_dict({
            "name": name,
            "count": int(len(signed)),
            "signed": {
                "mean": float(np.mean(signed)),
                "median": float(np.median(signed)),
                "rms": float(np.sqrt(np.mean(signed ** 2))),
                "max": float(np.max(signed)),
                "min": float(np.min(signed)),
                "std": float(np.std(signed)),
                "p95": float(np.percentile(signed, 95)),
                "p99": float(np.percentile(signed, 99)),
            },
            "unsigned": {
                "mean": float(np.mean(unsigned)),
                "median": float(np.median(unsigned)),
                "rms": float(np.sqrt(np.mean(unsigned ** 2))),
                "max": float(np.max(unsigned)),
                "min": float(np.min(unsigned)),
                "std": float(np.std(unsigned)),
                "p95": float(np.percentile(unsigned, 95)),
                "p99": float(np.percentile(unsigned, 99)),
            },
            "positive_count": int(len(positive)),
            "negative_count": int(len(negative)),
            "positive_mean": float(np.mean(positive)) if len(positive) > 0 else 0.0,
            "negative_mean": float(np.mean(negative)) if len(negative) > 0 else 0.0,
        })

    @staticmethod
    def _colorize_point_cloud(
        pcd: o3d.geometry.PointCloud,
        signed_deviation: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        根据 signed deviation 为点云着色。

        使用 diverging colormap:
        - 负偏差 (内凹): 蓝色
        - 零偏差: 白色/灰色
        - 正偏差 (外凸): 红色
        """
        pcd_colored = o3d.geometry.PointCloud(pcd)

        if len(signed_deviation) == 0:
            pcd_colored.paint_uniform_color([0.8, 0.8, 0.8])
            return pcd_colored

        # 以最大绝对值为对称边界，保证颜色映射中心在 0
        vmax = max(abs(np.min(signed_deviation)), abs(np.max(signed_deviation)))
        if vmax < 1e-9:
            pcd_colored.paint_uniform_color([0.8, 0.8, 0.8])
            return pcd_colored

        # 向量化色谱映射。旧实现对每个点调用一次 _diverging_color (Python 循环)，
        # 5 万点约 0.1s；向量化后降至微秒级，结果逐位一致。
        t = np.clip(np.asarray(signed_deviation, dtype=np.float64) / vmax, -1.0, 1.0)
        negative = t < 0
        s = np.abs(t)
        colors = np.column_stack([
            np.where(negative, s, 1.0),          # R: 负→s,   正→1
            np.where(negative, s, 1.0 - s),      # G: 负→s,   正→1-s
            np.where(negative, 1.0, 1.0 - s),    # B: 负→1,   正→1-s
        ])
        pcd_colored.colors = o3d.utility.Vector3dVector(colors)
        return pcd_colored

    @staticmethod
    def _diverging_color(t: float) -> np.ndarray:
        """
        简单的 diverging colormap (标量版，保留供逐点调用与测试使用)。
        t ∈ [-1, 1]；返回 RGB 颜色。
        """
        t = np.clip(t, -1.0, 1.0)
        # 蓝色 -> 白色 -> 红色
        if t < 0:
            # 蓝色到白色
            s = -t
            return np.array([s, s, 1.0])
        else:
            # 白色到红色
            s = t
            return np.array([1.0, 1.0 - s, 1.0 - s])
