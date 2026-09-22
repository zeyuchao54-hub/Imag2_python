"""
PointToCAD 测试套件（标准库 unittest，无需安装 pytest）

运行方式（项目根目录）:
    .venv310\\Scripts\\python.exe -m unittest discover -s tests -v"""

import _bootstrap  # noqa: F401  # 必须先于 open3d 导入，固定 OpenMP 线程数
import os
import sys

# 保证从任意工作目录都能 import 项目模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import unittest  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from aligner import DatumAligner  # noqa: E402
from plane import Plane  # noqa: E402

def _make_plane(plane_id, normal, offset, extent=40.0, n=800, jitter=0.01):
    """构造一个带噪声的理想平面点云。"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    rng = np.random.default_rng(plane_id)
    a = rng.uniform(-extent / 2, extent / 2, n)
    b = rng.uniform(-extent / 2, extent / 2, n)
    pts = (np.outer(a, u) + np.outer(b, v)) + normal * offset
    pts += rng.normal(0.0, jitter, pts.shape)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    # 点位于 +normal*offset 一侧，故方程为 n.x + d = 0 且 d = -offset
    return Plane(plane_id=plane_id, model=[*normal, -offset], cloud=cloud)

class TestRotationBetweenVectors(unittest.TestCase):
    """#1: _get_rotation_matrix_between_vectors 必须恒为正规旋转 (det=+1)。"""

    def setUp(self):
        self.aligner = DatumAligner()

    def _assert_maps_and_is_rotation(self, v1, v2):
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        R = self.aligner._get_rotation_matrix_between_vectors(v1, v2)

        # 1) 必须是正规旋转，不是反射
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9,
                               msg=f"det(R)={np.linalg.det(R)}，出现反射矩阵")
        # 2) 必须是正交矩阵
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-9))
        # 3) 必须真的把 v1 转到 v2 方向
        mapped = R @ (v1 / np.linalg.norm(v1))
        self.assertTrue(np.allclose(mapped, v2 / np.linalg.norm(v2), atol=1e-9))

    def test_antiparallel_downward_normal_is_not_a_reflection(self):
        """回归测试: 法向朝下 (0,0,-1) 时旧代码返回 -I (det=-1)。"""
        R = self.aligner._get_rotation_matrix_between_vectors(
            np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
        )
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
        self.assertFalse(np.allclose(R, -np.eye(3)), "仍在返回 -I 反射矩阵")

    def test_antiparallel_random_axes(self):
        for v1 in ([1, 0, 0], [0, 1, 0], [0.3, -0.5, 0.81], [-1, 2, -2.5]):
            self._assert_maps_and_is_rotation(v1, -np.asarray(v1, dtype=float))

    def test_nearly_antiparallel_is_stable(self):
        """
        极点附近的病态输入也必须给出正规旋转。

        注意 v2 与 v1 夹角略小于 180°（指向 +Z 且带极小 X 分量），
        此时 π 旋转 trick 无法精确命中目标，只能保证 det=+1 且显著改善对齐。
        """
        v1 = np.array([0.0, 0.0, -1.0])
        v2 = np.array([1e-7, 0.0, 1.0])
        R = self.aligner._get_rotation_matrix_between_vectors(v1, v2)

        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-9))

        v1h = v1 / np.linalg.norm(v1)
        v2h = v2 / np.linalg.norm(v2)
        mapped = R @ v1h
        before = float(np.dot(v1h, v2h))
        after = float(np.dot(mapped, v2h))
        self.assertGreater(after, before)
        self.assertGreater(after, 0.999999)

    def test_parallel_and_general_cases(self):
        self._assert_maps_and_is_rotation([0, 0, 1], [0, 0, 1])
        self._assert_maps_and_is_rotation([1, 0, 0], [0, 1, 0])
        self._assert_maps_and_is_rotation([1, 2, 3], [-4, 5, 6])

    def test_zero_vector_degrades_to_identity(self):
        R = self.aligner._get_rotation_matrix_between_vectors([0, 0, 0], [0, 0, 1])
        self.assertTrue(np.allclose(R, np.eye(3)))

class TestDatumAlignment(unittest.TestCase):
    """3-2-1 对齐端到端: 旋转必须正规，主基准面必须落到 Z=0。"""

    def test_downward_facing_primary_plane_keeps_right_handed_frame(self):
        aligner = DatumAligner()
        # jitter=0.0 → 理想共面几何，SVD 重新拟合应精确复现平面方程
        planes = [
            _make_plane(1, [0, 0, -1], offset=-30.0, jitter=0.0),   # 主基准面法向朝下 → 旧代码必然反射
            _make_plane(2, [1, 0, 0], offset=-20.0, jitter=0.0),
            _make_plane(3, [0, 1, 0], offset=-10.0, jitter=0.0),
            _make_plane(4, [0, 0, 1], offset=25.0, jitter=0.0),
        ]
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

        aligned_planes, aligned_vertices, T = aligner.align_to_datum(
            planes=planes, vertices=vertices, primary_plane_id=1, secondary_plane_id=2
        )

        self.assertAlmostEqual(float(np.linalg.det(T[:3, :3])), 1.0, places=9)
        self.assertAlmostEqual(float(np.linalg.det(T)), 1.0, places=9)

        # 主基准面 (P1) 对齐后其法向应平行于 +Z
        n_z = abs(float(np.dot(aligned_planes[0].normal, [0, 0, 1])))
        self.assertGreater(n_z, 0.999)
        # 次基准面 (P2) 对齐后其法向应平行于 +X
        n_x = abs(float(np.dot(aligned_planes[1].normal, [1, 0, 0])))
        self.assertGreater(n_x, 0.999)

        # 3-2-1 的完整语义: 原点取三基准面交点后，主基准面必须恰好落在 Z=0。
        # (#5 修复项——旧实现取 vertices[0]=[0,0,0]，主基准面对齐后位于 Z=+30)
        self.assertAlmostEqual(float(aligned_planes[0].model[3]), 0.0, places=9,
                               msg="主基准面对齐后未落在 Z=0，3-2-1 原点定位不正确")
        self.assertAlmostEqual(float(aligned_planes[1].model[3]), 0.0, places=9,
                               msg="次基准面对齐后未落在 X=0")

    def test_datum_planes_land_on_zero_with_realistic_noise(self):
        """带真实噪声时残差应远小于平面本身的偏移量 (修复前残差≈30)。"""
        aligner = DatumAligner()
        planes = [
            _make_plane(1, [0, 0, -1], offset=-30.0, jitter=0.01),
            _make_plane(2, [1, 0, 0], offset=-20.0, jitter=0.01),
            _make_plane(3, [0, 1, 0], offset=-10.0, jitter=0.01),
            _make_plane(4, [0, 0, 1], offset=25.0, jitter=0.01),
        ]
        aligned, _, _ = aligner.align_to_datum(
            planes=planes, vertices=np.array([[0.0, 0.0, 0.0]]),
            primary_plane_id=1, secondary_plane_id=2,
        )
        # 平面偏移量为 20~30，噪声引入的残差应在 1e-3 量级 (比修复前小 4 个数量级)
        self.assertLess(abs(float(aligned[0].model[3])), 5e-3)
        self.assertLess(abs(float(aligned[1].model[3])), 5e-3)

    def test_secondary_normal_pointing_minus_x(self):
        """次基准法向旋转后指向 -X 的路径。"""
        aligner = DatumAligner()
        planes = [
            _make_plane(1, [0, 1, 0], offset=5.0),
            _make_plane(2, [-1, 0, 0], offset=5.0),
            _make_plane(3, [0, 0, 1], offset=5.0),
        ]
        _, _, T = aligner.align_to_datum(
            planes=planes, vertices=np.array([[0.0, 0.0, 0.0]]),
            primary_plane_id=1, secondary_plane_id=2
        )
        self.assertAlmostEqual(float(np.linalg.det(T[:3, :3])), 1.0, places=9)
        self.assertGreater(abs(float(np.dot(np.asarray(planes[1].normal), [1, 0, 0]))), 0.999)

    def test_transform_is_volume_preserving(self):
        """刚体变换必须保体积: |det T| == 1。"""
        aligner = DatumAligner()
        planes = [
            _make_plane(1, [0.2, 0.1, 1.0], offset=3.0),
            _make_plane(2, [1.0, 0.3, 0.2], offset=-4.0),
            _make_plane(3, [0.1, 1.0, -0.2], offset=7.0),
        ]
        _, _, T = aligner.align_to_datum(
            planes=planes, vertices=np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]),
            primary_plane_id=1, secondary_plane_id=2
        )
        self.assertAlmostEqual(abs(float(np.linalg.det(T))), 1.0, places=9)

class TestDatumOriginResolution(unittest.TestCase):
    """#5: 3-2-1 原点必须取三基准面交点，并有明确回退链。"""

    def setUp(self):
        self.aligner = DatumAligner()

    def test_origin_is_the_three_plane_intersection(self):
        planes = [
            _make_plane(1, [0, 0, 1], offset=0.0, extent=40.0),
            _make_plane(2, [1, 0, 0], offset=-20.0, extent=40.0),
            _make_plane(3, [0, 1, 0], offset=-10.0, extent=40.0),
        ]
        vertices = np.array([[999.0, 999.0, 999.0]])  # 显然不是正确原点

        origin, source = self.aligner._resolve_datum_origin(planes, planes[0], planes[1], vertices)

        # _make_plane 的约定: 点位于 +normal·offset，故方程为 n.x - offset = 0
        #   P1: z = 0, P2: x = -20, P3: y = -10
        self.assertTrue(np.allclose(origin, [-20.0, -10.0, 0.0], atol=1e-6),
                        f"原点应为三面交点 (-20,-10,0)，实际 {origin}")
        self.assertIn("A∩B∩C", source)

    def test_origin_lies_on_both_datum_planes(self):
        """原点必须同时满足主/次基准面方程——这是 3-2-1 正确性的核心。"""
        planes = [
            _make_plane(1, [0, 0, 1], offset=7.0, extent=30.0),
            _make_plane(2, [1, 0, 0], offset=-13.0, extent=30.0),
            _make_plane(3, [0, 1, 0], offset=-5.0, extent=30.0),
        ]
        origin, _ = self.aligner._resolve_datum_origin(planes, planes[0], planes[1], np.zeros((0, 3)))
        for p in (planes[0], planes[1]):
            residual = abs(float(np.dot(origin, p.model[:3]) + p.model[3]))
            self.assertLess(residual, 1e-9, f"原点不在 P{p.id} 上 (残差 {residual})")

    def test_falls_back_to_first_vertex_when_no_orthogonal_tertiary(self):
        """所有其他面都与主/次基准面近平行时，退化为首个推导角点。"""
        planes = [
            _make_plane(1, [0, 0, 1], offset=0.0, extent=30.0),
            _make_plane(2, [1, 0, 0], offset=-5.0, extent=30.0),
            _make_plane(3, [0.02, 0, 1], offset=0.01, extent=30.0),   # 与 P1 近平行
        ]
        vertices = np.array([[3.0, 4.0, 5.0]])
        origin, source = self.aligner._resolve_datum_origin(planes, planes[0], planes[1], vertices)
        self.assertTrue(np.allclose(origin, [3.0, 4.0, 5.0]))
        self.assertIn("fallback", source)

    def test_falls_back_to_primary_centroid_when_no_vertices(self):
        planes = [
            _make_plane(1, [0, 0, 1], offset=0.0, extent=30.0),
            _make_plane(2, [1, 0, 0], offset=-5.0, extent=30.0),
            _make_plane(3, [0.02, 0, 1], offset=0.01, extent=30.0),
        ]
        origin, source = self.aligner._resolve_datum_origin(
            planes, planes[0], planes[1], np.zeros((0, 3))
        )
        self.assertTrue(np.allclose(origin, np.asarray(planes[0].centroid), atol=1e-9))
        self.assertIn("fallback", source)

    def test_old_behaviour_would_leave_primary_off_z0(self):
        """
        回归锚点: 若沿用旧实现 (取一个不在主基准面上的点作原点)，
        主基准面对齐后会被留在 Z≠0；新实现取三面交点则恰好为 0。

        推导: 旋转把主基准面法向 n 转到 +Z 时平面截距 d_A 不变 (绕原点旋转
        保持平面到原点距离)，随后按 t = -R·o 平移，新截距变为 d_A + n·o。
        故原点 o 落在平面上 (n·o = -d_A) 时新截距恰为 0。
        """
        planes = [
            _make_plane(1, [0, 0, 1], offset=-30.0, extent=40.0, jitter=0.0),
            _make_plane(2, [1, 0, 0], offset=-20.0, extent=40.0, jitter=0.0),
            _make_plane(3, [0, 1, 0], offset=-10.0, extent=40.0, jitter=0.0),
        ]
        origin_new, _ = self.aligner._resolve_datum_origin(
            planes, planes[0], planes[1], np.zeros((0, 3))
        )
        origin_old = np.array([0.0, 0.0, 0.0])  # 远离主基准面 (z=-30) 的点

        n_a = np.asarray(planes[0].normal, dtype=float)
        d_a = float(planes[0].model[3])

        # 新原点必须落在主基准面上 → 新截距为 0
        self.assertLess(abs(float(np.dot(n_a, origin_new)) + d_a), 1e-9,
                        "新原点不在主基准面上")
        self.assertAlmostEqual(d_a + float(np.dot(n_a, origin_new)), 0.0, places=9,
                               msg="新原点应使主基准面对齐后落在 Z=0")

        # 旧原点不在主基准面上 → 新截距 = d_A = 30，明显偏离 Z=0
        self.assertGreater(abs(d_a + float(np.dot(n_a, origin_old))), 1.0,
                           "旧原点下主基准面应明显偏离 Z=0")

if __name__ == "__main__":
    unittest.main(verbosity=2)
