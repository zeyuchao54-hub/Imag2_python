"""#3: 尺度链路不变量——Plane.scale 后所有派生量同步，且导出层不再二次补乘。"""

import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tempfile  # noqa: E402
import unittest  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from plane import Plane  # noqa: E402
from report import ReportGenerator  # noqa: E402


def _make_plane(plane_id, normal, offset, extent=30.0, n=600):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    rng = np.random.default_rng(plane_id)
    a = rng.uniform(-extent / 2, extent / 2, n)
    b = rng.uniform(-extent / 2, extent / 2, n)
    pts = np.outer(a, u) + np.outer(b, v) + normal * offset

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    # 点位于 +normal*offset 一侧，故方程为 n.x + d = 0 且 d = -offset
    return Plane(plane_id=plane_id, model=[*normal, -offset], cloud=cloud)


class TestPlaneScale(unittest.TestCase):
    def setUp(self):
        self.plane = _make_plane(1, [0.2, 0.1, 1.0], offset=-3.0)

    def test_points_satisfy_scaled_equation(self):
        """缩放后每个内点仍必须满足新的平面方程 (d → factor·d)。"""
        factor = 42.19231
        before = np.asarray(self.plane.cloud.points).copy()
        a, b, c, d0 = self.plane.model

        self.plane.scale(factor)

        after = np.asarray(self.plane.cloud.points)
        self.assertTrue(np.allclose(after, before * factor, atol=1e-9))

        a1, b1, c1, d1 = self.plane.model
        residual = after @ np.array([a1, b1, c1]) + d1
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-6),
                        f"缩放后方程残差过大: max|r|={np.max(np.abs(residual))}")

    def test_centroid_and_area_scale_correctly(self):
        centroid_before = np.asarray(self.plane.centroid).copy()
        area_before = float(self.plane.area)
        factor = 7.5

        self.plane.scale(factor)

        self.assertTrue(np.allclose(self.plane.centroid, centroid_before * factor, atol=1e-9))
        self.assertAlmostEqual(float(self.plane.area), area_before * factor ** 2, places=6)

    def test_normal_is_unit_and_unchanged(self):
        normal_before = np.asarray(self.plane.normal).copy()
        self.plane.scale(13.0)
        self.assertTrue(np.allclose(self.plane.normal, normal_before, atol=1e-12))
        self.assertAlmostEqual(float(np.linalg.norm(self.plane.normal)), 1.0, places=12)

    def test_double_scale_is_idempotent_to_single(self):
        """分两次 scale(2) 必须等价于一次 scale(4) —— 旧实现只缩放点云，此性质必然破坏。"""
        p_a = _make_plane(2, [1, 1, 1], offset=2.0)
        p_b = _make_plane(2, [1, 1, 1], offset=2.0)

        p_a.scale(2.0).scale(2.0)
        p_b.scale(4.0)

        self.assertTrue(np.allclose(p_a.model, p_b.model, atol=1e-9))
        self.assertTrue(np.allclose(p_a.centroid, p_b.centroid, atol=1e-9))
        self.assertAlmostEqual(float(p_a.area), float(p_b.area), places=6)

    def test_rejects_non_positive_factor(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"factor={bad} 应当被拒绝"):
                _make_plane(3, [0, 0, 1], 1.0).scale(bad)

    def test_factor_one_is_noop(self):
        before_model = self.plane.model.copy()
        before_pts = np.asarray(self.plane.cloud.points).copy()
        self.plane.scale(1.0)
        self.assertTrue(np.allclose(self.plane.model, before_model))
        self.assertTrue(np.allclose(np.asarray(self.plane.cloud.points), before_pts))


class TestExportJsonHasNoHiddenScaling(unittest.TestCase):
    """export_json 必须原样输出 Plane 的物理值，不得再乘任何 factor。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reporter = ReportGenerator(out_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_matches_plane_object_exactly(self):
        plane = _make_plane(1, [0, 0, 1], offset=-5.0)
        vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        path = self.reporter.export_json([plane], vertices, filename="report.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        eq = data["planes"][0]["equation"]
        # 关键: JSON 里的 d 必须与 Plane.model[3] 完全一致 (没有任何补乘)
        self.assertAlmostEqual(eq["d"], round(float(plane.model[3]), 6), places=9)
        self.assertAlmostEqual(eq["a"], round(float(plane.model[0]), 6), places=9)
        self.assertAlmostEqual(
            data["planes"][0]["estimated_area_mm2"], round(float(plane.area), 6), places=6
        )
        for json_c, plane_c in zip(
            data["planes"][0]["centroid"], np.asarray(plane.centroid).tolist()
        ):
            self.assertAlmostEqual(json_c, round(plane_c, 6), places=9)

        # 顶点同样原样输出
        self.assertEqual(len(data["vertices"]), 2)
        self.assertAlmostEqual(data["vertices"][0]["x"], 1.0, places=9)
        self.assertAlmostEqual(data["vertices"][1]["z"], 6.0, places=9)

    def test_scaled_scene_exports_consistent_units(self):
        """
        端到端: 先 scale 再导出，JSON 中的方程必须与 fused.ply 里的点自洽。
        这正是旧实现的失效模式——点云是物理单位、报表却是虚拟单位。
        """
        plane = _make_plane(1, [0.3, -0.2, 1.0], offset=1.5)
        factor = 42.19231
        plane.scale(factor)

        json_path = self.reporter.export_json([plane], np.zeros((0, 3)), filename="report.json")
        ply_path = self.reporter.export_fused_ply([plane], filename="fused.ply")

        with open(json_path, encoding="utf-8") as f:
            eq = json.load(f)["planes"][0]["equation"]

        fused = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(fused.points)
        residual = pts @ np.array([eq["a"], eq["b"], eq["c"]]) + eq["d"]

        # 用相对判据: PLY 以 float32 落盘、JSON 把 a/b/c 舍入到 6 位小数，
        # 在坐标量级 ~600mm 上会有约 1e-4 的绝对残差，这是存储精度而非单位错配。
        # 旧实现 (点云物理单位 / 报表虚拟单位) 的残差约为量级本身，相对误差 ~1e0，
        # 因此 1e-5 的相对阈值足以锁定单位一致性。
        coord_scale = max(float(np.abs(pts).max()), 1.0)
        rel_err = float(np.max(np.abs(residual))) / coord_scale
        self.assertLess(rel_err, 1e-5,
                        f"报表方程与 fused.ply 点云单位不一致: max|r|={np.max(np.abs(residual)):.3e}, "
                        f"相对误差={rel_err:.3e} (坐标量级 {coord_scale:.1f})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
