import logging
import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation
from plane import Plane


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
        # Step 3: 平移对齐 - 将主角点或基准面中心平移至原点 (0, 0, 0)
        # -------------------------------------------------------------
        if len(vertices) > 0:
            # 以第 1 个推导出来的角点作为绝对原点 (0, 0, 0)
            origin_ref = vertices[0]
        else:
            origin_ref = plane_a.centroid

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

        self.logger.info("基准坐标系对齐完成！原点已定位于首个 CAD 角点，主平面已对齐至 Z=0。")
        return planes, aligned_vertices, T