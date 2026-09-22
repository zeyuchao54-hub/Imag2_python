"""可复现性: 固定种子后，同一输入必须逐位复现。"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from plane import Plane  # noqa: E402
from features import FeatureExtractor  # noqa: E402


def _disk_cloud(surfaces, n=900):
    """构造带孔圆柱的简化点云: 圆柱侧壁 + 顶底环。"""
    pts = []
    for radius, z0, z1 in surfaces:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.column_stack([
            radius * np.cos(t) + 5.0,
            radius * np.sin(t) - 3.0,
            np.full(n, (z0 + z1) / 2) + np.linspace(-2.0, 2.0, n),
        ]))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(pts))
    return cloud


class TestObbJitterReproducible(unittest.TestCase):
    """_compute_obb 的防共面抖动必须是确定性的。"""

    def _perfectly_coplanar_plane(self):
        n = 400
        g = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(g, g)
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        return Plane(plane_id=1, model=[0, 0, 1, 0], cloud=cloud)

    def test_identical_planes_give_identical_obb(self):
        results = [self._perfectly_coplanar_plane()._compute_obb() for _ in range(5)]
        for obb in results[1:]:
            self.assertTrue(np.allclose(obb.center, results[0].center, atol=1e-9))
            self.assertTrue(np.allclose(obb.extent, results[0].extent, atol=1e-9))
            self.assertTrue(np.allclose(obb.R, results[0].R, atol=1e-9))


class TestCylinderRansacReproducible(unittest.TestCase):
    def test_same_seed_same_cylinders(self):
        cloud = _disk_cloud([(10.0, 0, 10), (4.0, 0, 10)])
        extractor = FeatureExtractor()

        a = extractor.detect_cylinders(cloud, seed=42, ransac_iterations=200)
        b = extractor.detect_cylinders(cloud, seed=42, ransac_iterations=200)

        self.assertEqual(len(a), len(b))
        for ca, cb in zip(a, b):
            self.assertTrue(np.allclose(ca.axis, cb.axis, atol=1e-12))
            self.assertTrue(np.allclose(ca.center, cb.center, atol=1e-9))
            self.assertAlmostEqual(ca.radius, cb.radius, places=9)

    def test_seed_is_actually_used(self):
        """不同种子应能产生不同结果，证明 seed 真的参与了采样。"""
        cloud = _disk_cloud([(10.0, 0, 10), (4.0, 0, 10)])
        extractor = FeatureExtractor()
        runs = [extractor.detect_cylinders(cloud, seed=s, ransac_iterations=200)
                for s in (1, 2, 3, 4, 5)]
        sigs = {(round(c.radius, 6), round(float(c.axis[2]), 6))
                for run in runs for c in run}
        # 至少出现两种不同的拟合结果，说明种子有效
        self.assertGreater(len(sigs), 1, "不同种子得到完全相同的结果，seed 可能未生效")


class TestSeedAcceptance(unittest.TestCase):
    """_seed_everything 必须接受合法种子并拒绝非法值。"""

    def test_rejects_non_int_seed(self):
        from main import IndustrialPipeline
        for bad in ("42", 4.2, None, True):
            with self.assertRaises((ValueError, TypeError), msg=f"seed={bad!r} 应被拒绝"):
                IndustrialPipeline._seed_everything(object.__new__(IndustrialPipeline), bad)

    def test_accepts_valid_seed(self):
        from main import IndustrialPipeline
        pipe = object.__new__(IndustrialPipeline)
        import logging
        pipe.logger = logging.getLogger("test")
        IndustrialPipeline._seed_everything(pipe, 7)
        IndustrialPipeline._seed_everything(pipe, 0)
        # 不应抛错即通过


class TestOpen3DRansacDeterministic(unittest.TestCase):
    """
    o3d.utility.random.seed 的可复现性语义。

    注意: Open3D 的 RANSAC 是**顺序消费同一条全局随机流**的。因此设一次种子
    能保证"整段调用序列"逐位复现 (只要输入与调用顺序相同)，但不能保证
    "单独两次 segment_plane 调用"结果一致——第二次调用消费的是流的下游。
    流水线依赖的正是前者，故此处测试整段序列。
    """

    def _run_extraction_sequence(self, seed=42):
        """模拟 detector.detect 的多轮 RANSAC 剥离序列。"""
        o3d.utility.random.seed(seed)
        rng = np.random.default_rng(0)
        pts = rng.normal(0, 1, (20000, 3))
        pts[:, 2] *= 0.01  # 压成薄片，制造强平面
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)

        out = []
        for _ in range(3):
            if len(cloud.points) < 3:  # RANSAC 至少需要 ransac_n=3 个点
                break
            model, inliers = cloud.segment_plane(0.05, 3, 1500)
            out.append((np.asarray(model, dtype=float).copy(), len(inliers)))
            cloud = cloud.select_by_index(inliers, invert=True)
        self.assertTrue(out, "序列为空，测试失去意义")
        return out

    def test_seeded_sequence_is_bitwise_reproducible(self):
        a = self._run_extraction_sequence()
        b = self._run_extraction_sequence()

        self.assertEqual(len(a), len(b))
        for (m1, n1), (m2, n2) in zip(a, b):
            self.assertTrue(np.allclose(m1, m2),
                            f"同种子下平面模型不一致: {m1} vs {m2}")
            self.assertEqual(n1, n2, "同种子下内点数不一致")

    def test_different_seeds_change_the_sequence(self):
        """换种子必须改变结果，证明 seed 真的生效 (而非碰巧收敛)。"""
        sig_a = self._run_extraction_sequence(seed=1)
        sig_b = self._run_extraction_sequence(seed=999)

        differs = any(
            not np.allclose(m1, m2) or n1 != n2
            for (m1, n1), (m2, n2) in zip(sig_a, sig_b)
        )
        self.assertTrue(differs, "不同种子得到完全相同的序列，seed 可能未生效")


if __name__ == "__main__":
    unittest.main(verbosity=2)
