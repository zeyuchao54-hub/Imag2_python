"""
软对应 / 鲁棒加权 (借鉴 CAD-Deform 的 p2p soft indicator)。

CAD-Deform 用核加权 + 精确匹配门控来处理"没有精确对应"的点。移植到本项目的
部分: 对应点距离不再二值化 (门限内全 1 / 门限外全 0)，而是按 Tukey 双权重衰减。
硬门限下"刚好卡在门限内"的点与可靠点同权，而这类点最可能是错误对应
(CAD 缺失面/遮挡区域)，会把 Sim3 尺度估计和 Datum-Weighted ICP 拉偏。
"""

import os

import _bootstrap  # noqa: F401  # 必须先于 open3d 导入，固定 OpenMP 线程数

import unittest

import numpy as np

from registration import ICPRegistrar


class TestBiweightKernel(unittest.TestCase):
    """Tukey 双权重的基本数学性质。"""

    def setUp(self):
        self.k = ICPRegistrar._biweight

    def test_zero_residual_gets_full_weight(self):
        self.assertAlmostEqual(float(self.k(np.array([0.0]), 5.0)[0]), 1.0, places=12)

    def test_at_scale_is_zero(self):
        w = self.k(np.array([5.0, -5.0]), 5.0)
        self.assertTrue(np.allclose(w, 0.0))

    def test_beyond_scale_is_zero(self):
        w = self.k(np.array([5.5, 100.0, -1e6]), 5.0)
        self.assertTrue(np.allclose(w, 0.0))

    def test_monotonically_decreasing(self):
        r = np.linspace(0, 5, 200)
        w = self.k(r, 5.0)
        self.assertTrue(np.all(np.diff(w) <= 1e-15), "权重必须随残差单调不增")

    def test_matches_closed_form(self):
        r = np.array([0.0, 1.0, 2.5, 4.0])
        w = self.k(r, 5.0)
        t = r / 5.0
        self.assertTrue(np.allclose(w, (1 - t * t) ** 2))

    def test_symmetric(self):
        r = np.linspace(-5, 5, 51)
        w = self.k(r, 5.0)
        self.assertTrue(np.allclose(w, w[::-1]))

    def test_strictly_descending_penalty_vs_hard_gate(self):
        """关键性质: 门限内的点必须被降权 (而非全部等价)。"""
        w = self.k(np.array([0.1, 3.0, 4.9]), 5.0)
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])
        self.assertGreater(w[2], 0.0)

    def test_invalid_scale_gives_zero_weight(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            w = self.k(np.array([0.0, 1.0]), bad)
            self.assertTrue(np.all(w == 0.0), f"scale={bad} 应给出零权重")

    def test_empty_input(self):
        self.assertEqual(len(self.k(np.zeros(0), 5.0)), 0)


class TestWeightedUmeyamaRobustness(unittest.TestCase):
    """核心收益: 污染对应下，鲁棒加权显著降低尺度估计误差。"""

    def _make_correspondences(self, n=3000, true_scale=42.0, seed=0):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(seed)
        p = rng.normal(0, 10, (n, 3))
        R = Rotation.from_euler("z", 17, degrees=True).as_matrix()
        t = np.array([5.0, -3.0, 8.0])
        q = true_scale * (R @ p.T).T + t
        return p, q, true_scale

    def test_no_contamination_is_exact(self):
        p, q, true_s = self._make_correspondences()
        s, _, _ = ICPRegistrar.weighted_umeyama(p, q)
        self.assertAlmostEqual(s, true_s, places=8)

    def _contaminate_random(self, q, k, rng, gate):
        """随机方向污染: 位移方向随机 (零均值，两种估计都会被部分抵消)。"""
        mag = rng.uniform(0.6, 0.95, k) * gate
        dirs = rng.normal(0, 1, (k, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        q[:k] += dirs * mag[:, None]
        return q

    def _contaminate_systematic(self, q, k, rng, gate):
        """
        系统性同向污染: 全部朝 +X 偏移。

        这才是物理上最危险的情形 —— CAD 缺失面/遮挡会把那一侧的对应点
        一致地拉偏，偏差不抵消，会直接把尺度估计带跑。
        """
        mag = rng.uniform(0.6, 0.95, k) * gate
        q[:k, 0] += mag
        return q

    def _mean_errors(self, contam_fn, frac, seeds, gate=6.0):
        """对多个种子取平均误差，避免单一种子的随机性主导结论。"""
        errs_hard, errs_robust = [], []
        for sd in seeds:
            rng = np.random.default_rng(100 + sd)
            p, q, true_s = self._make_correspondences(seed=sd)
            k = int(len(p) * frac)
            q_bad = contam_fn(q.copy(), k, rng, gate)
            err = np.linalg.norm(q_bad - q, axis=1)
            inl = err <= gate
            hard, _, _ = ICPRegistrar.weighted_umeyama(p[inl], q_bad[inl])
            robust, _, _ = ICPRegistrar.weighted_umeyama(
                p[inl], q_bad[inl],
                weights=ICPRegistrar._biweight(err[inl], gate),
            )
            errs_hard.append(abs(hard - true_s))
            errs_robust.append(abs(robust - true_s))
        return float(np.mean(errs_hard)), float(np.mean(errs_robust))

    SEEDS = (0, 1, 2, 3, 4)

    def test_robust_beats_hard_gate_systematic_outliers(self):
        """
        系统性同向外点 (物理上最危险): 鲁棒加权必须一致优于硬门限。
        多 seed 平均，避免单种子噪声翻转结论。
        """
        for frac in (0.2, 0.3, 0.4):
            e_hard, e_robust = self._mean_errors(
                self._contaminate_systematic, frac, self.SEEDS
            )
            self.assertLess(e_robust, e_hard,
                            f"系统性污染 {frac:.0%}: 鲁棒 {e_robust:.6f} 未优于硬门限 {e_hard:.6f}")

    def test_robust_beats_hard_gate_random_outliers(self):
        for frac in (0.2, 0.3, 0.4):
            e_hard, e_robust = self._mean_errors(
                self._contaminate_random, frac, self.SEEDS
            )
            self.assertLess(e_robust, e_hard,
                            f"随机污染 {frac:.0%}: 鲁棒 {e_robust:.6f} 未优于硬门限 {e_hard:.6f}")

    def test_improvement_is_substantial_systematic(self):
        """系统性污染下改善应至少 50%，否则说明加权没有真正起作用。"""
        gains = []
        for frac in (0.2, 0.3, 0.4):
            e_hard, e_robust = self._mean_errors(
                self._contaminate_systematic, frac, self.SEEDS
            )
            gains.append((e_hard - e_robust) / max(e_hard, 1e-12))
        for frac, gain in zip((0.2, 0.3, 0.4), gains):
            self.assertGreater(gain, 0.5,
                               f"系统性污染 {frac:.0%} 时改善仅 {gain:.1%}")

    def test_improvement_is_substantial_random(self):
        """随机方向污染下改善应至少 30% (零均值抵消会削弱差距)。"""
        for frac in (0.2, 0.3, 0.4):
            e_hard, e_robust = self._mean_errors(
                self._contaminate_random, frac, self.SEEDS
            )
            gain = (e_hard - e_robust) / max(e_hard, 1e-12)
            self.assertGreater(gain, 0.3,
                               f"随机污染 {frac:.0%} 时改善仅 {gain:.1%}")

    def test_zero_weights_give_unit_scale_not_crash(self):
        """全零权重 (门限内但 biweight 归零) 不得崩溃。"""
        p, q, _ = self._make_correspondences(n=200)
        w = np.zeros(len(p))
        s, R, t = ICPRegistrar.weighted_umeyama(p, q, weights=w)
        self.assertEqual(s, 1.0)
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertTrue(np.allclose(t, np.zeros(3)))


class TestWeightedUmeyamaEmptyInput(unittest.TestCase):
    """
    回归: 空输入此前会 ZeroDivisionError (np.full(0, 1.0/0))。
    门限把所有对应点滤光时会走到这里。
    """

    def test_empty_input_returns_identity(self):
        s, R, t = ICPRegistrar.weighted_umeyama(np.zeros((0, 3)), np.zeros((0, 3)))
        self.assertEqual(s, 1.0)
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertTrue(np.allclose(t, np.zeros(3)))

    def test_empty_with_weights(self):
        s, R, t = ICPRegistrar.weighted_umeyama(
            np.zeros((0, 3)), np.zeros((0, 3)), weights=np.zeros(0)
        )
        self.assertEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
