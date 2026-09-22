"""#6/#7: 公差阈值必须分纲 (长度 vs 角度)，每项须记录所用阈值与算法性质。"""

import _bootstrap  # noqa: F401  # 必须先于 open3d 导入，固定 OpenMP 线程数
import os
import sys

import unittest  # noqa: E402

import numpy as np  # noqa: E402

from features import PlaneFeature  # noqa: E402
from tolerance import ToleranceAnalyzer, ToleranceResult  # noqa: E402

def _plane(pid, normal, centroid, n=200, spread=0.3):
    """构造一个带噪声的平面特征。"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rng = np.random.default_rng(pid)
    a = rng.uniform(-spread, spread, n)
    b = rng.uniform(-spread, spread, n)
    pts = np.outer(a, u) + np.outer(b, v) + np.asarray(centroid, dtype=float)
    pts += rng.normal(0, 0.05, pts.shape)
    return PlaneFeature(
        id=pid, normal=normal, centroid=np.asarray(centroid, dtype=float),
        d=0.0, points=pts, source="scan",
    )

class TestThresholdSeparation(unittest.TestCase):
    def test_defaults_are_independent(self):
        a = ToleranceAnalyzer(tolerance_threshold_mm=2.0, angle_threshold_deg=5.0)
        self.assertEqual(a.tolerance_threshold_mm, 2.0)
        self.assertEqual(a.angle_threshold_deg, 5.0)

    def test_angle_default_preserves_old_behaviour(self):
        """未指定角度阈值时默认 2.0°，与旧硬编码值一致。"""
        self.assertEqual(ToleranceAnalyzer().angle_threshold_deg, 2.0)

    def test_rejects_invalid_thresholds(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"tolerance={bad} 应被拒绝"):
                ToleranceAnalyzer(tolerance_threshold_mm=bad)
            with self.assertRaises(ValueError, msg=f"angle={bad} 应被拒绝"):
                ToleranceAnalyzer(angle_threshold_deg=bad)

class TestAngularTolerancesUseAngleThreshold(unittest.TestCase):
    """方向公差必须按角度阈值判定，与长度阈值无关。"""

    def test_perpendicularity_flips_with_angle_threshold_only(self):
        """两面夹角 87° (偏离 90° 3°) → 垂直度偏差 3°。"""
        p1 = _plane(1, [0, 0, 1], [0, 0, 0])
        p2 = _plane(2, [np.sin(np.radians(87)), 0, np.cos(np.radians(87))], [5, 0, 0])

        loose = ToleranceAnalyzer(tolerance_threshold_mm=100.0, angle_threshold_deg=5.0)
        strict = ToleranceAnalyzer(tolerance_threshold_mm=100.0, angle_threshold_deg=1.0)

        r_loose = loose._perpendicularity(p1, p1, p2, p2)
        r_strict = strict._perpendicularity(p1, p1, p2, p2)

        self.assertAlmostEqual(r_loose.value, 3.0, places=3)
        self.assertEqual(r_loose.status, "PASS")
        self.assertEqual(r_strict.status, "FAIL")
        self.assertEqual(r_loose.unit, "deg")
        # 阈值必须被记录下来
        self.assertEqual(r_loose.threshold, 5.0)
        self.assertEqual(r_strict.threshold, 1.0)

    def test_parallelism_uses_angle_threshold(self):
        p1 = _plane(1, [0, 0, 1], [0, 0, 0])
        p2 = _plane(2, [np.sin(np.radians(1.5)), 0, np.cos(np.radians(1.5))], [0, 0, 10])

        a = ToleranceAnalyzer(angle_threshold_deg=2.0)
        b = ToleranceAnalyzer(angle_threshold_deg=0.5)
        self.assertEqual(a._parallelism(p1, p1, p2, p2).status, "PASS")
        self.assertEqual(b._parallelism(p1, p1, p2, p2).status, "FAIL")
        self.assertEqual(a._parallelism(p1, p1, p2, p2).unit, "deg")

    def test_angularity_uses_angle_threshold(self):
        """
        倾斜度 = |实测夹角 − CAD 标称夹角|。
        scan 两面夹角 30°，CAD 两面夹角 32° → 偏差 2°。
        """
        sp1 = _plane(1, [0, 0, 1], [0, 0, 0])
        sp2 = _plane(2, [np.sin(np.radians(30)), 0, np.cos(np.radians(30))], [0, 0, 10])
        cp1 = _plane(3, [0, 0, 1], [0, 0, 0])
        cp2 = _plane(4, [np.sin(np.radians(32)), 0, np.cos(np.radians(32))], [0, 0, 10])

        a = ToleranceAnalyzer(angle_threshold_deg=5.0)
        b = ToleranceAnalyzer(angle_threshold_deg=1.0)

        r_a = a._angularity(sp1, cp1, sp2, cp2)
        r_b = b._angularity(sp1, cp1, sp2, cp2)
        self.assertAlmostEqual(r_a.value, 2.0, places=3)
        self.assertEqual(r_a.status, "PASS")
        self.assertEqual(r_b.status, "FAIL")
        self.assertEqual(r_a.unit, "deg")
        self.assertEqual(r_a.threshold, 5.0)

class TestLinearTolerancesUseMmThreshold(unittest.TestCase):
    def test_flatness_uses_mm_threshold(self):
        noisy = _plane(1, [0, 0, 1], [0, 0, 0], spread=1.0)
        flat = _plane(1, [0, 0, 1], [0, 0, 0], n=200, spread=1.0)
        flat.points = flat.points @ np.array(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]]
        ) + np.array([0, 0, 3.0])  # 压成严格共面

        loose = ToleranceAnalyzer(tolerance_threshold_mm=100.0)
        strict = ToleranceAnalyzer(tolerance_threshold_mm=1e-6)

        self.assertEqual(loose._flatness(noisy, noisy).status, "PASS")
        self.assertEqual(strict._flatness(noisy, noisy).status, "FAIL")
        self.assertEqual(loose._flatness(noisy, noisy).unit, "mm")
        self.assertEqual(loose._flatness(noisy, noisy).threshold, 100.0)
        # 严格共面的点云，平面度应接近 0
        self.assertLess(strict._flatness(flat, flat).value, 1e-6)

    def test_profile_uses_mm_threshold(self):
        dev = np.linspace(-1.0, 1.0, 100)
        loose = ToleranceAnalyzer(tolerance_threshold_mm=10.0)
        strict = ToleranceAnalyzer(tolerance_threshold_mm=0.5)
        self.assertEqual(loose._profile_of_surface(dev).status, "PASS")
        self.assertEqual(strict._profile_of_surface(dev).status, "FAIL")
        self.assertEqual(loose._profile_of_surface(dev).threshold, 10.0)

class TestSelfDescribingResults(unittest.TestCase):
    """#7: 每项结果必须自描述阈值与算法性质，避免误导下游读者。"""

    def test_to_dict_includes_threshold_and_algorithm(self):
        r = ToleranceResult(
            name="X", value=1.0, unit="mm", status="PASS",
            threshold=2.0, algorithm="simplified max-min range",
        )
        d = r.to_dict()
        self.assertEqual(d["threshold"], 2.0)
        self.assertEqual(d["algorithm"], "simplified max-min range")

    def test_omitted_fields_are_absent_from_json(self):
        d = ToleranceResult(name="X", value=1.0).to_dict()
        self.assertNotIn("threshold", d)
        self.assertNotIn("algorithm", d)

    def test_every_computed_tolerance_records_threshold_and_status(self):
        """
        端到端: 任何算出来的项都必须带 threshold，且必须有明确的 PASS/FAIL 判定。

        这条断言是回归守卫——开发过程中曾因脚本失误把各方法的 status= 行覆盖掉，
        导致所有项静默退化为 dataclass 默认的 "N/A"，报告看起来正常却毫无判定能力。
        """
        analyzer = ToleranceAnalyzer()
        p1 = _plane(1, [0, 0, 1], [0, 0, 0])
        p2 = _plane(2, [1, 0, 0], [3, 0, 0])
        p3 = _plane(3, [0, 1, 0], [0, 4, 0])

        from features import PointFeature
        results = analyzer.analyze(
            scan_planes=[p1, p2, p3],
            cad_planes=[p1, p2, p3],
            scan_cylinders=[], cad_cylinders=[],
            scan_lines=[],
            scan_points=[PointFeature(id=1, coord=np.zeros(3), source="scan")],
            cad_points=[PointFeature(id=1, coord=np.zeros(3), source="cad")],
            deviation_signed=np.linspace(-0.5, 0.5, 50),
        )
        total = 0
        for category, items in results.items():
            for item in items:
                total += 1
                self.assertIsNotNone(item.threshold,
                                     f"{category}/{item.name} 缺少 threshold")
                self.assertIn(item.status, ("PASS", "FAIL"),
                              f"{category}/{item.name} 的状态应为 PASS/FAIL，实际 {item.status!r}")
                self.assertIn("threshold", item.to_dict())
        self.assertGreater(total, 0, "没有任何公差被计算出来，测试失去意义")

    def test_simplified_algorithms_are_labelled(self):
        """简化算法必须显式标注，不能伪装成 ISO 严格解。"""
        analyzer = ToleranceAnalyzer()
        p = _plane(1, [0, 0, 1], [0, 0, 0])
        results = analyzer.analyze(
            scan_planes=[p], cad_planes=[p],
            scan_cylinders=[], cad_cylinders=[], scan_lines=[],
            scan_points=[], cad_points=[],
            deviation_signed=np.linspace(-1, 1, 20),
        )
        labelled = [r for r in results["shape"] + results["profile"] if r.algorithm]
        self.assertTrue(labelled, "没有任何项标注算法性质")
        for r in labelled:
            self.assertIn("simplified", r.algorithm.lower(),
                          f"{r.name} 的算法标注未说明是简化实现")

if __name__ == "__main__":
    unittest.main(verbosity=2)
