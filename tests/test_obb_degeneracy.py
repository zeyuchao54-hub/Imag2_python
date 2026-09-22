"""
Plane._compute_obb 的退化修复: 用解析"第五顶点"替代随机抖动。

旧实现对完美共面的 CAD 点云加随机抖动 (np.random.normal, std=1e-6)，实测会把
30x30 共面正方形的 OBB 算成 42.22x42.22 (≈30·√2) —— 抖动让"最扁方向"变得模棱两可，
qhull 的最小体积盒搜索退到 45° 旋转的次优解，面内尺寸误差 40%。
新实现用 SVD 找出退化方向并补一个解析支撑点，面内尺寸误差为 0。"""

import _bootstrap  # noqa: F401  # 必须先于 open3d 导入，固定 OpenMP 线程数
import os
import sys

import unittest  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from plane import Plane  # noqa: E402

def _cloud(points):
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return c

def _coplanar_grid(n=30, half=15.0, z=0.0):
    g = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])

class TestObbDegenerateRepair(unittest.TestCase):
    """核心: 完美共面点云不能再报错，且面内尺寸必须精确。"""

    def test_coplanar_cloud_does_not_raise(self):
        """回归: 旧实现在此路径上靠抖动勉强成功，但不能保证。"""
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(_coplanar_grid()))
        obb = plane.obb
        self.assertIsNotNone(obb)

    def test_coplanar_in_plane_extent_is_exact(self):
        """
        关键回归: 30x30 共面正方形的 OBB，两个面内 extent 必须都是 30.0。
        旧抖动法给出 42.22 (≈30√2, 45° 旋转的次优盒)，本测试锁定该缺陷不再复现。
        """
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(_coplanar_grid()))
        ext = np.sort(np.asarray(plane.obb.extent))
        self.assertAlmostEqual(float(ext[1]), 30.0, places=6,
                               msg=f"面内 extent 应为 30.0，实际 {ext[1]}")
        self.assertAlmostEqual(float(ext[2]), 30.0, places=6,
                               msg=f"面内 extent 应为 30.0，实际 {ext[2]}")

    def test_normal_direction_extent_is_negligible(self):
        """法向 extent 应远小于面内尺寸 (支撑点的扰动不可感知)。"""
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(_coplanar_grid()))
        ext = np.sort(np.asarray(plane.obb.extent))
        in_plane = ext[1]
        self.assertLess(float(ext[0]) / in_plane, 1e-4,
                        f"法向 extent 占比过大: {ext[0] / in_plane:.2e}")

    def test_obb_orientation_matches_true_plane_normal(self):
        """OBB 的轴向必须与真实平面对齐，而不是被扰动带偏。"""
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(_coplanar_grid()))
        R = np.asarray(plane.obb.R)
        # OBB 三个轴中应有一个接近世界 Z 轴
        alignments = [abs(float(np.dot(R[:, i], [0, 0, 1]))) for i in range(3)]
        self.assertGreater(max(alignments), 0.9999,
                           f"OBB 未与平面对齐: 轴向-Z 对齐度 {alignments}")

    def test_deterministic_across_instances(self):
        """零随机性: 同样的点云，OBB 必须逐位一致。"""
        pts = _coplanar_grid()
        results = [Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts)).obb
                   for _ in range(5)]
        for obb in results[1:]:
            self.assertTrue(np.array_equal(np.asarray(obb.R), np.asarray(results[0].R)))
            self.assertTrue(np.array_equal(np.asarray(obb.extent),
                                           np.asarray(results[0].extent)))
            self.assertTrue(np.array_equal(np.asarray(obb.center),
                                           np.asarray(results[0].center)))

class TestOtherDegenerateShapes(unittest.TestCase):
    def test_collinear_cloud(self):
        """共线点云 (两个维度退化) 也应能修复。"""
        pts = np.column_stack([
            np.linspace(0, 40, 200),
            np.zeros(200),
            np.zeros(200),
        ])
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts))
        self.assertIsNotNone(plane.obb)
        # 主方向 extent 应接近 40
        self.assertAlmostEqual(float(np.max(np.asarray(plane.obb.extent))), 40.0, places=3)

    def test_tilted_coplanar_cloud_does_not_raise(self):
        """倾斜共面点云 (法向非坐标轴) 也必须能处理且不抛异常。"""
        g = np.linspace(-10, 10, 25)
        aa, bb = np.meshgrid(g, g)
        base = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
        angle = np.radians(35)
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(base @ R.T))
        self.assertIsNotNone(plane.obb)
        # 第三轴必须是平面法向 (解析 OBB 的核心保证)
        Rm = np.asarray(plane.obb.R)
        self.assertGreater(max(abs(float(np.dot(Rm[:, i], [0, 0, 1])))
                               for i in range(3)), 0.9999)

    def test_update_cloud_on_coplanar_is_safe(self):
        """
        回归: update_cloud 旧实现直接调用 get_oriented_bounding_box()，
        融合后的点云若完美共面会抛 RuntimeError (该路径此前没有保护)。
        """
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(_coplanar_grid()))
        try:
            plane.update_cloud(_cloud(_coplanar_grid(n=20, half=8.0)))
        except RuntimeError as e:
            self.fail(f"update_cloud 对共面点云抛出了 RuntimeError: {e}")
        ext = np.sort(np.asarray(plane.obb.extent))
        self.assertAlmostEqual(float(ext[2]), 16.0, places=6)

class TestAnalyticObbPrecision(unittest.TestCase):
    """解析 OBB 的精度: 对各向异性面片精确，对正方形有 PCA 固有歧义。"""

    def test_tilted_rectangle_extents_exact(self):
        """
        倾斜 20x40 矩形的 OBB 必须精确为 20 与 40。
        Open3D 对共面点云只会给轴对齐盒，此处会得到放大的错误值。
        """
        nrm = np.array([0.2, 0.1, 1.0])
        nrm /= np.linalg.norm(nrm)
        u0 = np.cross(nrm, [0, 0, 1.0])
        u0 /= np.linalg.norm(u0)
        v0 = np.cross(nrm, u0)
        g1 = np.linspace(-10, 10, 25)
        g2 = np.linspace(-20, 20, 31)
        q, w = np.meshgrid(g1, g2)
        pts = (np.outer(q.ravel(), u0) + np.outer(w.ravel(), v0) + nrm * 3.0)

        plane = Plane(plane_id=1, model=[*nrm, -3.0], cloud=_cloud(pts))
        ext = np.sort(np.asarray(plane.obb.extent))
        self.assertAlmostEqual(float(ext[1]), 20.0, places=6,
                               msg=f"面内 extent 应为 20.0，实际 {ext[1]}")
        self.assertAlmostEqual(float(ext[2]), 40.0, places=6,
                               msg=f"面内 extent 应为 40.0，实际 {ext[2]}")
        # 第三轴应为平面法向
        R = np.asarray(plane.obb.R)
        self.assertGreater(max(abs(float(np.dot(R[:, i], nrm))) for i in range(3)), 0.999)

    def test_obb_feeds_area_fallback_correctly(self):
        """
        OBB 面内尺寸正确 -> _estimate_area 的 OBB 兜底公式
        (sorted_extents[1] * [2]) 也应给出正确面积 20*40=800。
        旧的轴对齐/随机扰动 OBB 在此会给出被放大的错误面积。
        """
        nrm = np.array([0.2, 0.1, 1.0])
        nrm /= np.linalg.norm(nrm)
        u0 = np.cross(nrm, [0, 0, 1.0])
        u0 /= np.linalg.norm(u0)
        v0 = np.cross(nrm, u0)
        g1 = np.linspace(-10, 10, 25)
        g2 = np.linspace(-20, 20, 31)
        q, w = np.meshgrid(g1, g2)
        pts = np.outer(q.ravel(), u0) + np.outer(w.ravel(), v0)

        plane = Plane(plane_id=1, model=[*nrm, 0.0], cloud=_cloud(pts))
        ext = np.sort(np.asarray(plane.obb.extent))
        fallback_area = float(ext[1] * ext[2])
        self.assertAlmostEqual(fallback_area, 800.0, delta=1e-6,
                               msg=f"OBB 兜底面积应为 800，实际 {fallback_area}")

class TestSquareAmbiguity(unittest.TestCase):
    """
    正方形点阵的面内协方差是各向同性的 (σ²I)，面内主方向不唯一，
    PCA-OBB 在此固有歧义 —— 这不是缺陷，但必须记录在案，
    避免后来者以为此处还能再"修精确"。
    """

    def test_axis_aligned_square_is_exact(self):
        """轴对齐正方形无歧义，仍应精确。"""
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(_coplanar_grid()))
        ext = np.sort(np.asarray(plane.obb.extent))
        self.assertAlmostEqual(float(ext[1]), 30.0, places=6)
        self.assertAlmostEqual(float(ext[2]), 30.0, places=6)

    def test_rotated_square_is_bounded_and_deterministic(self):
        """旋转正方形: 只需保证不爆表、可复现、法向 span 可忽略。"""
        g = np.linspace(-10, 10, 25)
        aa, bb = np.meshgrid(g, g)
        base = np.column_stack([aa.ravel(), bb.ravel(), np.zeros(aa.size)])
        a = np.radians(35)
        Rz = np.array([[np.cos(a), -np.sin(a), 0],
                       [np.sin(a), np.cos(a), 0],
                       [0, 0, 1]])
        pts = base @ Rz.T

        planes = [Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts))
                  for _ in range(4)]
        ext0 = np.sort(np.asarray(planes[0].obb.extent))
        for p in planes[1:]:
            e = np.sort(np.asarray(p.obb.extent))
            self.assertTrue(np.array_equal(e, ext0), "同一输入多次结果不一致")

        # 上界: 不超过 20·√2 (45° 的理论最坏情况)
        self.assertLessEqual(float(ext0[2]), 20.0 * np.sqrt(2) + 1e-9)
        # 法向 span 可忽略
        self.assertLess(float(ext0[0]) / 20.0, 1e-4)

class TestNoRandomnessLeft(unittest.TestCase):
    """本模块至此不应再使用任何随机数 (用 AST 检查真实调用，不受注释影响)。"""

    def test_no_random_calls_in_plane_module(self):
        import ast
        import inspect
        import plane as plane_mod

        src = inspect.getsource(plane_mod)
        tree = ast.parse(src)
        random_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = (f"{ast.unparse(f.value)}.{f.attr}"
                        if isinstance(f, ast.Attribute) else ast.unparse(f))
                if "random" in name.lower():
                    random_calls.append(name)
        self.assertEqual(random_calls, [],
                         f"plane.py 仍存在随机调用: {random_calls}")

    def test_result_independent_of_global_seed(self):
        """改变全局随机种子不得影响 OBB 结果。"""
        pts = _coplanar_grid()
        np.random.seed(1)
        a = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts)).obb
        np.random.seed(999999)
        b = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts)).obb
        self.assertTrue(np.array_equal(np.asarray(a.R), np.asarray(b.R)))
        self.assertTrue(np.array_equal(np.asarray(a.extent), np.asarray(b.extent)))

class TestNormalCaseUnaffected(unittest.TestCase):
    """非退化点云必须走原路径，结果与 Open3D 原生一致。"""

    def test_3d_cloud_matches_native(self):
        rng = np.random.default_rng(7)
        pts = rng.uniform(-10, 10, (800, 3))
        native = _cloud(pts).get_oriented_bounding_box()
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts))
        self.assertTrue(np.allclose(np.asarray(plane.obb.R),
                                    np.asarray(native.R), atol=1e-12))
        self.assertTrue(np.allclose(np.asarray(plane.obb.extent),
                                    np.asarray(native.extent), atol=1e-12))

    def test_no_extra_points_introduced(self):
        """非退化路径不得篡改点云。"""
        pts = np.random.default_rng(3).uniform(-5, 5, (300, 3))
        plane = Plane(plane_id=1, model=[0, 0, 1, 0.0], cloud=_cloud(pts))
        self.assertEqual(len(plane.cloud.points), 300)

if __name__ == "__main__":
    unittest.main(verbosity=2)
