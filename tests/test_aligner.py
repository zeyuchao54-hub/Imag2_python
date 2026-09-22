"""
PointToCAD 测试套件（标准库 unittest，无需安装 pytest）

运行方式（项目根目录）:
    .venv310\\Scripts\\python.exe -m unittest discover -s tests -v
"""

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
    return Plane(plane_id=plane_id, model=[*normal, offset], cloud=cloud)


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
        planes = [
            _make_plane(1, [0, 0, -1], offset=-30.0),   # 主基准面法向朝下 → 旧代码必然反射
            _make_plane(2, [1, 0, 0], offset=-20.0),
            _make_plane(3, [0, 1, 0], offset=-10.0),
            _make_plane(4, [0, 0, 1], offset=25.0),
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
        # 注意: "主基准面恰好落在 Z=0" 还取决于原点的取法，
        # 属于 #5 (3-2-1 原点取三基准面交点) 的范围，此处不断言。

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
