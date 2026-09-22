"""#4: 距离类阈值必须随数据尺度自适应，而非硬编码绝对值。"""

import _bootstrap  # noqa: F401  # 必须先于 open3d 导入，固定 OpenMP 线程数
import os
import sys

import unittest  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from utils import diagonal_of, resolve_threshold  # noqa: E402
from geometry import GeometryAnalyzer  # noqa: E402
from graph import PlaneGraph  # noqa: E402
from merger import PlaneMerger  # noqa: E402
from plane import Plane  # noqa: E402
from preprocess import PointCloudPreprocessor  # noqa: E402

def _cloud(points):
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return c

def _box_planes(scale=1.0, jitter=0.0):
    """构造一个长方体的 3 个互相垂直的面 (单位边长 scale)。"""
    planes = []
    specs = [
        ("z", 0.0),   # z = 0
        ("y", 0.0),   # y = 0
        ("x", scale),  # x = scale
    ]
    for pid, (axis, offset) in enumerate(specs, start=1):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        other = [i for i in range(3) if i != idx]
        g = np.linspace(0, scale, 25)
        aa, bb = np.meshgrid(g, g)
        pts = np.zeros((aa.size, 3))
        pts[:, other[0]] = aa.ravel()
        pts[:, other[1]] = bb.ravel()
        pts[:, idx] = offset
        pts += jitter
        normal = np.zeros(3)
        normal[idx] = 1.0
        planes.append(Plane(plane_id=pid, model=[*normal, -offset], cloud=_cloud(pts)))
    return planes

class TestDiagonalOf(unittest.TestCase):
    def test_matches_manual_computation(self):
        pts = np.array([[0.0, 0, 0], [3.0, 4, 0], [0, 0, 12.0]])
        self.assertAlmostEqual(diagonal_of(pts), 13.0, places=9)  # 3-4-12 → 13

    def test_accepts_point_cloud(self):
        self.assertGreater(diagonal_of(_cloud(np.random.rand(50, 3) * 10)), 0.0)

    def test_degenerate_inputs_return_zero(self):
        self.assertEqual(diagonal_of(None), 0.0)
        self.assertEqual(diagonal_of(np.zeros((0, 3))), 0.0)
        self.assertEqual(diagonal_of(_cloud(np.zeros((0, 3)))), 0.0)

    def test_non_finite_returns_zero(self):
        pts = np.array([[0.0, 0, 0], [np.nan, 1, 1]])
        self.assertEqual(diagonal_of(pts), 0.0)

class TestResolveThreshold(unittest.TestCase):
    def test_explicit_wins(self):
        self.assertEqual(resolve_threshold(0.05, 100.0, 0.01), 0.05)

    def test_ratio_scales_with_reference(self):
        self.assertAlmostEqual(resolve_threshold(None, 100.0, 0.01), 1.0, places=9)
        self.assertAlmostEqual(resolve_threshold(None, 200.0, 0.01), 2.0, places=9)

    def test_degenerate_reference_uses_floor(self):
        self.assertEqual(resolve_threshold(None, 0.0, 0.01, floor=1e-9), 1e-9)

    def test_rejects_invalid_explicit(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError, msg=f"explicit={bad} 应被拒绝"):
                resolve_threshold(bad, 100.0, 0.01)

class TestMergerThresholdIsScaleRelative(unittest.TestCase):
    """共面融合的判定阈值必须随场景尺度等比缩放。"""

    def _two_fragments(self, scale, gap):
        """
        构造同一物理平面的两个碎片 (detector 输出的字典格式):
        一个在 z=0，一个沿法向平移 gap。gap 小于阈值时应被融合。
        """
        g = np.linspace(0, scale, 20)
        aa, bb = np.meshgrid(g, g)
        a = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
        b = a + np.array([0.0, 0.0, gap])
        return [
            {"id": 1, "model": [0.0, 0.0, 1.0, 0.0], "cloud": _cloud(a)},
            {"id": 2, "model": [0.0, 0.0, 1.0, -gap], "cloud": _cloud(b)},
        ]

    def test_same_relative_gap_merges_at_any_scale(self):
        """
        关键性质: 同一"相对间距"在不同绝对尺度下判定结果必须一致。
        旧硬编码 0.05 在 scale=1 时不融合、在 scale=100 时会错误融合。
        """
        merger = PlaneMerger()

        # 小尺度场景: 间距 = 对角线的 0.5% (远小于 1.34% 阈值) → 应融合
        small = self._two_fragments(scale=1.0, gap=0.005)
        merged_small = merger.merge(small)
        self.assertEqual(len(merged_small), 1, "小尺度下相对间距极小却未融合")

        # 大尺度场景: 同样的"相对间距"必须同样融合
        big = self._two_fragments(scale=100.0, gap=0.5)
        merged_big = merger.merge(big)
        self.assertEqual(len(merged_big), 1, "大尺度下同样的相对间距未融合")

    def test_relative_gap_above_threshold_does_not_merge(self):
        merger = PlaneMerger()
        for scale in (1.0, 10.0, 100.0):
            # 间距 = 对角线的 10% (远大于 1.34% 阈值) → 不应融合
            gap = scale * 0.1
            planes = self._two_fragments(scale=scale, gap=gap)
            self.assertEqual(len(merger.merge(planes)), 2,
                             f"scale={scale} 下相对间距 10% 被错误融合")

    def test_explicit_absolute_threshold_still_supported(self):
        merger = PlaneMerger(dist_threshold=0.01)
        planes = self._two_fragments(scale=1.0, gap=0.005)
        self.assertEqual(len(merger.merge(planes)), 1)

    def test_seam_voxel_is_spacing_relative(self):
        """接缝去重体素必须随点密度自适应: 密 → 小，疏 → 大，但都远小于点间距。"""
        for scale, n in ((1.0, 400), (10.0, 400), (100.0, 4000)):
            g = np.linspace(0, scale, n)
            aa, bb = np.meshgrid(g, g)
            pts = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
            cloud = _cloud(pts)
            voxel = PlaneMerger._seam_voxel_size(cloud)
            spacing = diagonal_of(cloud) / np.sqrt(len(cloud.points))
            self.assertLess(voxel, spacing,
                            f"scale={scale}: 体素 {voxel} 不小于点间距 {spacing}，会误并采样点")
            self.assertGreater(voxel, 0.0)

    def test_seam_voxel_rejects_invalid_divisor(self):
        PlaneMerger(seam_voxel_divisor=0)

class TestGeometryThresholdIsScaleRelative(unittest.TestCase):
    def _build_graph(self, scale):
        """构造长方体三面 (两两垂直) 的拓扑图。"""
        planes = _box_planes(scale=scale)
        graph = PlaneGraph(planes)
        ids = [p.id for p in planes]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                # compute_intersection_vertices 优先走 find_perpendicular_triplets，
                # 只有标为 "perpendicular" 的三面组才会被取用
                graph.add_edge(ids[i], ids[j], "perpendicular", 90.0)
        return graph

    def test_cluster_tolerance_scales_with_scene(self):
        """聚类容差必须随场景尺度等比缩放，否则大零件上角点无法合并。"""
        results = {}
        for scale in (1.0, 100.0):
            analyzer = GeometryAnalyzer()
            vertices = analyzer.compute_intersection_vertices(self._build_graph(scale))
            results[scale] = vertices

        # 两个尺度都应解出同一个角点并聚成 1 个
        for scale, verts in results.items():
            self.assertEqual(len(verts), 1,
                             f"scale={scale} 下顶点数 {len(verts)}，聚类容差未随尺度自适应")

        # 且顶点坐标应与尺度成正比
        v_small = results[1.0][0]
        v_big = results[100.0][0]
        self.assertTrue(np.allclose(v_big, v_small * 100.0, rtol=1e-6),
                        f"顶点未随尺度等比放大: {v_small} vs {v_big}")

    def test_explicit_overrides_still_work(self):
        analyzer = GeometryAnalyzer(cluster_tolerance=0.001, max_bound=5.0)
        verts = analyzer.compute_intersection_vertices(self._build_graph(1.0))
        self.assertEqual(len(verts), 1)

class TestNormalRadiusIsScaleRelative(unittest.TestCase):
    def test_radius_scales_with_cloud(self):
        ratios = []
        for scale in (1.0, 100.0):
            pre = PointCloudPreprocessor()
            g = np.linspace(0, scale, 30)
            aa, bb = np.meshgrid(g, g)
            pts = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
            cloud = _cloud(pts)
            diag = diagonal_of(cloud)
            radius = resolve_threshold(
                pre.normal_radius, diag, pre.normal_radius_ratio,
                floor=1e-9, name="r",
            )
            ratios.append(radius / diag)
        for r in ratios:
            self.assertAlmostEqual(r, PointCloudPreprocessor.NORMAL_RADIUS_RATIO, places=6)

    def test_radius_positive_for_normal_cloud(self):
        pre = PointCloudPreprocessor()
        radius = resolve_threshold(
            None, diagonal_of(_cloud(np.random.rand(100, 3) * 5)),
            pre.normal_radius_ratio, floor=1e-9, name="r",
        )
        self.assertGreater(radius, 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
