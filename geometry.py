import logging
import numpy as np
from typing import List
from plane import Plane
from graph import PlaneGraph
from utils import diagonal_of, resolve_threshold


class GeometryAnalyzer:
    """
    工业级几何分析与计算模块
    职责: 解析拓扑图，通过线性代数求解 CAD 角点，并进行空间聚类去重
    """

    #: 顶点聚类容差相对于"全部平面点云对角线"的比例。
    #: 旧实现硬编码 cluster_tolerance=0.02，隐含"点云尺度≈1"假设；
    #: 该比例在当前参考数据集上复现原值，但换尺度/换零件时自适应。
    CLUSTER_TOLERANCE_RATIO = 0.005

    #: "幽灵交点"过滤边界 = 场景对角线 × 该倍数。
    #: 三平面近于平行时会解出数十倍于零件的伪交点，需按场景尺度成比例剔除。
    MAX_BOUND_RATIO = 13.0

    def __init__(self, cluster_tolerance=None, max_bound=None,
                 cluster_tolerance_ratio=None, max_bound_ratio=None):
        """
        :param cluster_tolerance: 顶点聚类容差 (绝对单位)。默认 None → 按场景对角线比例自适应
        :param max_bound: 异常点过滤边界 (绝对单位)。默认 None → 按场景对角线倍数自适应
        :param cluster_tolerance_ratio: 覆盖默认聚类容差比例
        :param max_bound_ratio: 覆盖默认边界倍数
        """
        self.logger = logging.getLogger("PointToCAD_System.Geometry")
        self.cluster_tolerance = cluster_tolerance
        self.max_bound = max_bound
        self.cluster_tolerance_ratio = (
            self.CLUSTER_TOLERANCE_RATIO if cluster_tolerance_ratio is None
            else float(cluster_tolerance_ratio)
        )
        self.max_bound_ratio = (
            self.MAX_BOUND_RATIO if max_bound_ratio is None
            else float(max_bound_ratio)
        )

    def compute_intersection_vertices(self, graph: PlaneGraph) -> np.ndarray:
        """
        根据拓扑图，计算所有有效的三面交点 (CAD 顶点)
        """
        self.logger.info("开始进行几何求交与顶点推导...")

        # 0. 解析与场景尺度挂钩的容差 (旧实现为硬编码绝对值)
        scene_diagonal = diagonal_of(
            np.vstack([np.asarray(p.cloud.points) for p in graph.nodes.values()])
            if graph.nodes else np.zeros((0, 3))
        )
        cluster_tol = resolve_threshold(
            self.cluster_tolerance, scene_diagonal,
            self.cluster_tolerance_ratio, floor=1e-12, name="geometry.cluster_tolerance",
        )
        max_bound = resolve_threshold(
            self.max_bound, scene_diagonal,
            self.max_bound_ratio, floor=1e-6, name="geometry.max_bound",
        )
        self.logger.debug(
            f"顶点容差: cluster={cluster_tol:.6g}, max_bound={max_bound:.6g} "
            f"(场景对角线 {scene_diagonal:.6g})"
        )

        # 1. 从拓扑图中提取所有“两两互相垂直”的三面组合
        triplets = graph.find_perpendicular_triplets()
        if not triplets:
            self.logger.warning("未在拓扑图中找到互相垂直的三面组合，无法计算直角顶点。")
            return np.array([])

        # 如果严格垂直的三面组太少（如 < 3），退而求“互不平行”的三面组，
        # 以应对点云噪声导致法向偏离 90° 的实际情况。
        if len(triplets) < 3:
            self.logger.info(
                f"严格垂直的三面组仅 {len(triplets)} 个，退而使用互不平行三面组以捕获更多顶点..."
            )
            triplets = graph.find_independent_triplets()

        self.logger.debug(f"找到 {len(triplets)} 组垂直三面体，准备求解空间坐标。")

        raw_vertices = []
        for p1, p2, p3 in triplets:
            vertex = self._solve_intersection(p1, p2, p3)

            if vertex is not None:
                # 过滤掉因为微小误差跑到“十万八千里”之外的非法坐标
                if np.all(np.abs(vertex) < max_bound):
                    raw_vertices.append(vertex)

        if not raw_vertices:
            self.logger.warning("所有求出的交点均超出合法物理边界。")
            return np.array([])

        # 2. 对求解出的坐标进行空间聚类与去重
        clean_vertices = self._cluster_vertices(raw_vertices, cluster_tol)

        self.logger.info(f"顶点计算完成！共推导出 {len(clean_vertices)} 个有效 CAD 顶点。")
        for idx, v in enumerate(clean_vertices):
            self.logger.debug(f"  -> 顶点 {idx + 1}: X={v[0]:8.4f}, Y={v[1]:8.4f}, Z={v[2]:8.4f}")

        return np.array(clean_vertices)

    def _solve_intersection(self, p1: Plane, p2: Plane, p3: Plane) -> np.ndarray:
        """
        利用线性代数求解三元一次方程组:
        A1*x + B1*y + C1*z = -D1
        A2*x + B2*y + C2*z = -D2
        A3*x + B3*y + C3*z = -D3
        """
        # 构建系数矩阵 A (3x3)
        A = np.array([
            p1.model[:3],
            p2.model[:3],
            p3.model[:3]
        ])

        # 构建常数项矩阵 B (3x1)
        B = np.array([
            -p1.model[3],
            -p2.model[3],
            -p3.model[3]
        ])

        try:
            # 检查行列式，防止奇异矩阵（例如存在两个几乎平行的面）
            if np.abs(np.linalg.det(A)) < 1e-5:
                return None

            # 求解 Ax = B
            point = np.linalg.solve(A, B)
            return point

        except np.linalg.LinAlgError:
            self.logger.debug("矩阵奇异，无法求解该三面组合的交点。")
            return None

    def _cluster_vertices(self, vertices: List[np.ndarray],
                          cluster_tolerance: float) -> List[np.ndarray]:
        """
        顶点空间聚类算法
        由于 RANSAC 平面并非 100% 完美的理论平面，多个相近组合算出的角点可能存在微小偏差。
        该算法将距离小于 tolerance 的点平均化，融合成一个唯一的精准角点。

        :param cluster_tolerance: 聚类容差，由 compute_intersection_vertices 按场景尺度解析
        """
        if not vertices:
            return []

        clustered = []

        for pt in vertices:
            found_cluster = False
            for i, center in enumerate(clustered):
                # 如果当前点距离某个已存在的角点非常近
                if np.linalg.norm(pt - center['coord']) < cluster_tolerance:
                    # 将它加入该聚类，并动态更新该聚类的平均坐标
                    center['points'].append(pt)
                    center['coord'] = np.mean(center['points'], axis=0)
                    found_cluster = True
                    break

            if not found_cluster:
                # 发现一个新的独立角点，创建一个新的聚类组
                clustered.append({
                    'coord': pt,
                    'points': [pt]
                })

        # 提取出聚类后的平均坐标作为最终顶点
        final_vertices = [cluster['coord'] for cluster in clustered]
        return final_vertices