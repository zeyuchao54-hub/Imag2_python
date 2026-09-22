"""
Datum-Weighted ICP 求解器的秩亏处理。

原计划"把正规方程改成直接 SVD 最小二乘"，实测后否决 (详见实现注释):
本轮真实系统 cond(A)≈8、严格秩亏时 solve 本就会抛错、且直接 lstsq 慢 439 倍。
因此这里只锁定两个**真实成立**的性质:
  1) 秩亏系统 (只扫到单个面/细长件) 下，兜底解必须是正确的最小范数解;
  2) solve 抛 LinAlgError 已被现有 except 捕获，解必须有限。
"""

import os

import _bootstrap  # noqa: F401

import unittest

import numpy as np


def _build_planar_system(n=4000):
    """
    构造秩亏的点面 ICP 系统: 所有源点位于 y=0.03 平面、法向恒为 +Y。
    此时 6 自由度中只有 Y 平移可观测，秩 = 3/6。
    """
    g1 = np.linspace(-100, 100, 400)
    g2 = np.linspace(-1.5, 1.5, 10)
    a, b = np.meshgrid(g1, g2)
    src = np.column_stack([a.ravel(), np.full(a.size, 0.03), b.ravel()])
    normal = np.tile([0.0, 1.0, 0.0], (len(src), 1))
    return src, normal


class TestRankDeficientSolve(unittest.TestCase):
    """秩亏系统的正确行为。"""

    def test_planar_system_is_actually_rank_deficient(self):
        """先确认测试构造真的秩亏，否则后续断言没有意义。"""
        src, nrm = _build_planar_system()
        src_tf = src + np.array([0.5, 0.2, -0.1])
        r = np.einsum("ij,ij->i", src_tf - src, nrm)
        A = np.hstack([np.cross(src_tf, nrm), nrm])
        AtA = (A.T @ A)
        self.assertEqual(np.linalg.matrix_rank(AtA), 3,
                         "该系统本应秩亏 (秩 3/6)")

    def test_solve_raises_on_exactly_singular(self):
        """记录真实行为: 恰好奇异时 solve 抛 LinAlgError (可被现有 except 捕获)。"""
        src, nrm = _build_planar_system()
        t = np.array([0.5, 0.2, -0.1])
        src_tf = src + t
        r = np.einsum("ij,ij->i", src_tf - src, nrm)
        A = np.hstack([np.cross(src_tf, nrm), nrm])
        AtA = A.T @ A
        Atb = A.T @ (-r)
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.solve(AtA, Atb)

    def test_lstsq_fallback_gives_minimum_norm_solution(self):
        """
        秩亏兜底: 只有 Y 平移可观测，解必须是 [0,0,0 | 0,-0.2,0]。
        切向与旋转分量处于零空间，最小范数解把它们置 0 而非给出任意大值。
        """
        src, nrm = _build_planar_system()
        t = np.array([0.5, 0.2, -0.1])
        src_tf = src + t
        r = np.einsum("ij,ij->i", src_tf - src, nrm)
        A = np.hstack([np.cross(src_tf, nrm), nrm])
        AtA = A.T @ A
        Atb = A.T @ (-r)

        try:
            x = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

        self.assertTrue(np.all(np.isfinite(x)), "兜底解必须有限")
        expected = np.array([0.0, 0.0, 0.0, 0.0, -0.2, 0.0])
        self.assertTrue(np.allclose(x, expected, atol=1e-9),
                        f"秩亏兜底解错误: {x}, 期望 {expected}")

    def test_well_conditioned_system_recovers_increment(self):
        """对照组: 满秩系统应精确恢复位姿增量。"""
        rng = np.random.default_rng(0)
        n = 3000
        src = rng.normal(0, 5, (n, 3))
        nrm = rng.normal(0, 1, (n, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        t = np.array([0.1, -0.2, 0.3])
        src_tf = src + t
        r = np.einsum("ij,ij->i", src_tf - src, nrm)
        A = np.hstack([np.cross(src_tf, nrm), nrm])
        AtA = A.T @ A
        Atb = A.T @ (-r)
        x = np.linalg.solve(AtA, Atb)
        self.assertTrue(np.allclose(x[3:], -t, atol=1e-8))


class TestSolveStrategyComparison(unittest.TestCase):
    """
    记录"为何不改用直接最小二乘"的量化依据。
    这些断言锁定的是"当前实现已足够好"这一结论，防止后来者误以为必须改。
    """

    def test_realistic_system_is_well_conditioned(self):
        """真实点面系统 (权重跨 6 个数量级) 的 cond(A) 仍很小。"""
        rng = np.random.default_rng(3)
        n = 4000
        src = rng.normal(0, 5, (n, 3))
        nrm = rng.normal(0, 1, (n, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        w = 10.0 ** rng.uniform(-6, 0, n)
        A = np.hstack([np.cross(src, nrm), nrm])
        Aw = A * np.sqrt(w)[:, None]
        cond_a = np.linalg.cond(Aw)
        self.assertLess(cond_a, 1e3,
                        f"cond(A)={cond_a:.2e} 已很高，需要重新评估是否改用直接最小二乘")

    def test_direct_lstsq_is_far_slower_at_scale(self):
        """直接 lstsq(Aw) 在大规模下显著变慢，这是否决该改动的代价依据。"""
        import time

        rng = np.random.default_rng(1)
        n = 20000
        src = rng.normal(0, 5, (n, 3))
        nrm = rng.normal(0, 1, (n, 3))
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        A = np.hstack([np.cross(src, nrm), nrm])
        AtA = A.T @ A
        bh = -rng.normal(0, 0.5, n)
        Atb = A.T @ bh
        b2 = -rng.normal(0, 0.5, n)

        # 预热，避免首次调用开销污染计时
        np.linalg.solve(AtA, Atb)
        np.linalg.lstsq(A, b2, rcond=None)

        t0 = time.perf_counter()
        for _ in range(20):
            np.linalg.solve(AtA, Atb)
        t_solve = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        for _ in range(5):
            np.linalg.lstsq(A, b2, rcond=None)
        t_direct = (time.perf_counter() - t0) / 5

        self.assertGreater(t_direct, t_solve * 5,
                           f"直接 lstsq 仅慢 {t_direct / max(t_solve, 1e-9):.1f}x，"
                           "代价可能已可接受，需重新评估")


if __name__ == "__main__":
    unittest.main(verbosity=2)
