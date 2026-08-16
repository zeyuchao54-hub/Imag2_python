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

    def _get_rotation_matrix_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """计算将向量 vec1 旋转到 vec2 的 3x3 旋转矩阵"""
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)

        # 如果两向量几乎同向
        if np.allclose(v1, v2, atol=1e-5):
            return np.eye(3)
        # 如果两向量反向
        if np.allclose(v1, -v2, atol=1e-5):
            return -np.eye(3)

        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

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