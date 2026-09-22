import logging
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation
from plane import Plane
from utils import diagonal_of


class DatumAligner:
    """
    工业级 3-2-1 基准坐标系对齐引擎
    职责: 将随机的点云/平面/顶点坐标系变换为以物理基准面为 Z=0、主角点为 (0,0,0) 的标准 CAD 坐标系
    """

    def __init__(self):
        self.logger = logging.getLogger("PointToCAD_System.Aligner")

    #: 判定"同向/反向极点"的余弦容差。越接近 ±1，轴角分解越病态，必须走极点分支。
    _POLE_COS_TOL = 1.0 - 1e-9

    @staticmethod
    def _any_perpendicular(unit_vec: np.ndarray) -> np.ndarray:
        """
        返回任意一个与 unit_vec 正交的单位向量。

        选取与 unit_vec 最不平行的世界坐标轴做叉积，保证数值稳定性。
        """
        world_axes = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        ref = min(world_axes, key=lambda a: abs(float(np.dot(a, unit_vec))))
        axis = np.cross(unit_vec, ref)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            # 理论上不会发生 (min 已选中最不平行的轴)，保留兜底
            axis = np.cross(unit_vec, np.array([0.0, 0.0, 1.0]))
            norm = float(np.linalg.norm(axis))
        return axis / norm

    def _get_rotation_matrix_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        计算将向量 vec1 旋转到 vec2 的 3x3 旋转矩阵，保证 det(R) = +1 (纯旋转)。

        历史缺陷: 旧实现在两向量反向时直接返回 -np.eye(3)，其行列式为 -1，
        是一次镜像反射而非旋转，会把整个场景左右翻转 (基准面法向朝下、
        或次基准法向绕 Z 转完后指向 -X 时都会命中)。
        反向的正确旋转是绕任意一条垂直于 vec1 的轴旋转 π。
        """
        n1 = float(np.linalg.norm(vec1))
        n2 = float(np.linalg.norm(vec2))
        if n1 < 1e-12 or n2 < 1e-12:
            self.logger.warning("基准对齐收到零向量，跳过该步旋转")
            return np.eye(3)

        v1 = np.asarray(vec1, dtype=float) / n1
        v2 = np.asarray(vec2, dtype=float) / n2
        cos_angle = float(np.clip(np.dot(v1, v2), -1.0, 1.0))

        # 同向: 单位旋转
        if cos_angle >= self._POLE_COS_TOL:
            return np.eye(3)

        # 反向 (或轴角分解已病态): 绕垂直轴转 π，结果必为正规旋转
        if cos_angle <= -self._POLE_COS_TOL:
            axis = self._any_perpendicular(v1)
            return Rotation.from_rotvec(axis * np.pi).as_matrix()

        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(cos_angle)

        rot = Rotation.from_rotvec(axis * angle)
        return rot.as_matrix()

    @staticmethod
    def _solve_three_planes(pa: Plane, pb: Plane, pc: Plane) -> Optional[np.ndarray]:
        """
        求解三个平面的交点。矩阵奇异或结果非有限时返回 None。
        """
        A = np.array([pa.model[:3], pb.model[:3], pc.model[:3]], dtype=float)
        b = -np.array([pa.model[3], pb.model[3], pc.model[3]], dtype=float)
        try:
            if abs(float(np.linalg.det(A))) < 1e-6:
                return None
            corner = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(corner)):
            return None
        return corner

    def _resolve_datum_origin(
        self,
        planes: List[Plane],
        plane_a: Plane,
        plane_b: Plane,
        vertices: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """
        解析 3-2-1 对齐的原点。

        旧实现直接取 vertices[0] (推导出的第一个 CAD 角点)。问题在于该点的选取
        取决于 graph 里三元组的枚举顺序与聚类顺序，并不保证落在三个基准面上；
        结果是"原点被移走了，但主基准面并没有被推到 Z=0"——实测中主基准面对齐后
        位于 Z=+30 而非 Z=0，3-2-1 只满足了旋转部分。

        正确语义: 原点 = Datum A ∩ Datum B ∩ Datum C。只有同时落在主基准面上的点
        才能在对齐后把主基准面压到 Z=0。

        :return: (原点坐标, 来源说明)
        """
        # ---- 1. 优先: Datum A ∩ B ∩ C ----
        # 第三基准面 C 的选取准则: 法向与 A/B 都不接近平行 (|n·nA| 与 |n·nB| 都较小)，
        # 否则三平面近共线，求出的交点会跑到无穷远。
        scene_diagonal = diagonal_of(
            np.vstack([np.asarray(p.cloud.points) for p in planes])
            if planes else np.zeros((0, 3))
        )
        scene_centroid = (
            np.mean(np.vstack([np.asarray(p.cloud.points) for p in planes]), axis=0)
            if planes else np.zeros(3)
        )

        tertiary = None
        best_parallelism = float("inf")
        for p in planes:
            if p.id in (plane_a.id, plane_b.id):
                continue
            parallelism = max(
                abs(float(np.dot(p.normal, plane_a.normal))),
                abs(float(np.dot(p.normal, plane_b.normal))),
            )
            if parallelism < best_parallelism:
                best_parallelism = parallelism
                tertiary = p

        # |dot| < 0.7 ≈ 与 A/B 的夹角均大于 45°，可构成稳定角点
        if tertiary is not None and best_parallelism < 0.7:
            corner = self._solve_three_planes(plane_a, plane_b, tertiary)
            if corner is not None:
                # 兜底: 交点不应远离场景 (近共线时会解出伪交点)
                if np.linalg.norm(corner - scene_centroid) <= 5.0 * max(scene_diagonal, 1e-12):
                    return corner, (
                        f"Datum A∩B∩C (P{plane_a.id} ∩ P{plane_b.id} ∩ P{tertiary.id})"
                    )
                self.logger.warning(
                    f"基准面交点 P{plane_a.id}∩P{plane_b.id}∩P{tertiary.id} 远离场景，弃用"
                )
        elif tertiary is not None:
            self.logger.warning(
                f"找不到与主/次基准面都足够正交的第三基准面 (最小 |n·n| = {best_parallelism:.3f})"
            )

        # ---- 2. 退化: 退回复用推导出的第一个角点 ----
        if vertices is not None and len(vertices) > 0:
            self.logger.warning(
                "未能用三基准面交点定位原点，退化为使用首个推导角点 (主基准面可能不落在 Z=0)"
            )
            return np.asarray(vertices[0], dtype=float), "first derived vertex (fallback)"

        # ---- 3. 最后退化: 主基准面质心 ----
        self.logger.warning(
            "未能用三基准面交点定位原点且无可用角点，退化为使用主基准面质心 "
            "(主基准面可能不落在 Z=0)"
        )
        return np.asarray(plane_a.centroid, dtype=float), "primary plane centroid (fallback)"

    def align_to_datum(
            self,
            planes: List[Plane],
            vertices: np.ndarray,
            primary_plane_id: int = 1,
            secondary_plane_id: int = 4
    ) -> Tuple[List[Plane], np.ndarray, np.ndarray]:
        """
        根据指定的基准面和角点将整个场景对齐到标准坐标系
        :param primary_plane_id: 主基准面 ID (Datum A, 将被对齐到 Z=0)
        :param secondary_plane_id: 次基准面 ID (Datum B, 将被对齐到 X=0)
        :return: (aligned_planes, aligned_vertices, transform_matrix)
        """
        self.logger.info(
            f"开始进行 3-2-1 基准坐标系对齐 (主基准面 P{primary_plane_id}, 次基准面 P{secondary_plane_id})...")

        # 找主基准面 Plane A
        plane_a = next((p for p in planes if p.id == primary_plane_id), planes[0])
        plane_b = next((p for p in planes if p.id == secondary_plane_id), planes[1] if len(planes) > 1 else planes[0])

        # -------------------------------------------------------------
        # Step 1: 旋转对齐 - 主基准面法向对齐到 +Z 轴 (0, 0, 1)
        # -------------------------------------------------------------
        normal_a = plane_a.normal
        R_z = self._get_rotation_matrix_between_vectors(normal_a, np.array([0.0, 0.0, 1.0]))

        # -------------------------------------------------------------
        # Step 2: 旋转对齐 - 次基准面法向绕 Z 轴旋转对齐到 +X 轴 (1, 0, 0)
        # -------------------------------------------------------------
        normal_b_rot = R_z @ plane_b.normal
        # 忽略 Z 分量，仅在 XY 平面上计算与 X 轴的旋转夹角
        normal_b_2d = np.array([normal_b_rot[0], normal_b_rot[1], 0.0])
        if np.linalg.norm(normal_b_2d) > 1e-4:
            R_x = self._get_rotation_matrix_between_vectors(normal_b_2d, np.array([1.0, 0.0, 0.0]))
        else:
            R_x = np.eye(3)

        R_total = R_x @ R_z

        # 正规性校验: det(R) 必须为 +1。若为 -1 说明装配出了反射矩阵，
        # 后续所有点云/平面都会被镜像翻转，属于不可静默吞掉的程序性错误。
        det_R = float(np.linalg.det(R_total))
        if not np.isclose(det_R, 1.0, atol=1e-6):
            raise ValueError(
                f"3-2-1 基准对齐装配出非正规旋转矩阵 (det={det_R:.6f})，"
                f"该场景会被镜像反射，已中止。请检查主/次基准面 (P{primary_plane_id}/P{secondary_plane_id}) 的法向是否退化。"
            )

        # -------------------------------------------------------------
        # Step 3: 平移对齐 - 将三基准面交点平移至原点 (0, 0, 0)
        # 只有交点同时落在主基准面上，对齐后主基准面才真正位于 Z=0
        # -------------------------------------------------------------
        origin_ref, origin_source = self._resolve_datum_origin(
            planes, plane_a, plane_b, vertices
        )
        self.logger.info(f"3-2-1 原点定位方式: {origin_source}")

        origin_transformed = R_total @ origin_ref
        t_total = -origin_transformed

        # 构造 4x4 齐次变换矩阵 T
        T = np.eye(4)
        T[:3, :3] = R_total
        T[:3, 3] = t_total

        # -------------------------------------------------------------
        # Step 4: 应用变换矩阵更新点云、平面方程与顶点
        # -------------------------------------------------------------
        # 1. 变换顶点
        aligned_vertices = []
        if len(vertices) > 0:
            for v in vertices:
                v_trans = R_total @ v + t_total
                aligned_vertices.append(v_trans)
            aligned_vertices = np.array(aligned_vertices)

        # 2. 变换平面数据与点云
        for p in planes:
            p.cloud.transform(T)
            # 重新计算坐标系变换后的平面方程系数 [A, B, C, D]
            pts = np.asarray(p.cloud.points)
            centroid = np.mean(pts, axis=0)
            # 使用 SVD 重新计算精确平面模型
            cov = np.cov(pts.T)
            _, _, vh = np.linalg.svd(cov)
            normal = vh[2, :]
            d = -np.dot(normal, centroid)
            p.model = np.array([normal[0], normal[1], normal[2], d])
            p.normal = normal
            p.centroid = centroid

        self.logger.info(
            f"基准坐标系对齐完成！原点定位于 {origin_source}，主基准面 P{plane_a.id} 已对齐至 Z=0。"
        )
        return planes, aligned_vertices, T