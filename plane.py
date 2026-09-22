import numpy as np
import open3d as o3d

#: _compute_obb 防退化抖动使用的固定种子，保证 OBB 可复现
_JITTER_SEED = 20240901


class Plane:
    """
    工业级平面数据对象 (Data Object)
    职责: 封装单一面片的物理与几何属性，提供基础的数学计算接口
    """

    def __init__(self, plane_id: int, model: list, cloud: o3d.geometry.PointCloud):
        """
        :param plane_id: 平面唯一标识符
        :param model: 平面方程系数 [a, b, c, d] (Ax + By + Cz + D = 0)
        :param cloud: 属于该平面的点云数据 (内点)
        """
        self.id = plane_id

        # 基础属性
        self.model = np.array(model)
        self.cloud = cloud

        # 几何特征 (初始化时自动计算)
        self.normal = self._compute_normal()
        self.centroid = self.cloud.get_center()

        # 物理边界特征
        self.obb = self._compute_obb()
        self.area = self._estimate_area()

    def _compute_obb(self) -> o3d.geometry.OrientedBoundingBox:
        """计算定向包围盒，对完美共面的 CAD 点云做抖动容错。"""
        try:
            return self.cloud.get_oriented_bounding_box()
        except RuntimeError:
            # CAD 点云可能完美共面，导致 qhull 失败。加入微小抖动后重试。
            # 抖动必须可复现: 使用固定种子的局部 Generator，避免每次 Import/
            # 每次运行的 OBB 都不同 (否则可视化与 report.json 无法稳定复现)。
            rng = np.random.default_rng(_JITTER_SEED)
            jittered = o3d.geometry.PointCloud()
            pts = np.asarray(self.cloud.points)
            noise = rng.normal(0, 1e-6, pts.shape)
            jittered.points = o3d.utility.Vector3dVector(pts + noise)
            return jittered.get_oriented_bounding_box()

    def _compute_normal(self) -> np.ndarray:
        """提取并标准化平面的法向量 (A, B, C)"""
        normal = self.model[:3]
        norm_length = np.linalg.norm(normal)
        if norm_length == 0:
            return np.array([0.0, 0.0, 0.0])
        return normal / norm_length

    def _estimate_area(self) -> float:
        """
        估算平面的物理面积。

        方法: 将点云投影到平面局部 2D 坐标系，计算 2D 凸包面积。
        相比 OBB（定向包围盒），凸包更紧致，且对非矩形边界适应更好。
        进一步对凸包顶点做 P95 边缘收缩，剔除最远端的飞点/相邻面混入点，
        避免面积被边缘噪点异常撑大。
        """
        if len(self.cloud.points) < 3:
            return 0.0

        points = np.asarray(self.cloud.points)
        normal = self.normal

        # 1. 构建平面局部坐标系 (centroid 为原点, normal 为 Z)
        if abs(normal[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        u = np.cross(normal, ref)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-8:
            ref = np.array([1.0, 0.0, 0.0])
            u = np.cross(normal, ref)
            u_norm = np.linalg.norm(u)
        u = u / u_norm

        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # 2. 投影到 2D
        centered = points - self.centroid
        coords_2d = np.column_stack((centered @ u, centered @ v))

        # 3. 计算 2D 凸包面积
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords_2d)
            area = hull.volume  # 2D 中 volume 即为面积
        except Exception:
            # fallback: OBB
            extents = self.obb.extent
            sorted_extents = np.sort(extents)
            return float(sorted_extents[1] * sorted_extents[2])

        # 4. P95 边缘收缩: 剔除最远的 5% 凸包顶点，抗飞点
        hull_vertices = coords_2d[hull.vertices]
        dists = np.linalg.norm(hull_vertices, axis=1)
        threshold = np.percentile(dists, 95)
        filtered_vertices = hull_vertices[dists <= threshold]

        if len(filtered_vertices) >= 3:
            try:
                hull_shrunk = ConvexHull(filtered_vertices)
                area = hull_shrunk.volume
            except Exception:
                pass  # 保持原凸包面积

        return float(area)

    def angle_with(self, other_plane: 'Plane') -> float:
        """
        计算与另一个平面的夹角 (单位: 度)
        用于后续 merger.py 判断两个面是否平行/共面
        """
        dot_product = np.dot(self.normal, other_plane.normal)
        # 避免浮点数精度溢出导致 arccos 报错 (如 dot_product = 1.0000001)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        # 平面法向量夹角可能是钝角，取锐角部分作为面面夹角
        if angle_deg > 90.0:
            angle_deg = 180.0 - angle_deg

        return angle_deg

    def distance_to_point(self, point: np.ndarray) -> float:
        """
        计算空间中某一点到该平面的垂直距离
        公式: d = |Ax + By + Cz + D| / sqrt(A^2 + B^2 + C^2)
        """
        a, b, c, d = self.model
        x, y, z = point
        numerator = np.abs(a * x + b * y + c * z + d)
        denominator = np.linalg.norm([a, b, c])
        return float(numerator / denominator)

    def update_cloud(self, new_cloud: o3d.geometry.PointCloud):
        """
        更新点云数据并重新计算几何特征
        (在 merger.py 缝合碎片平面后调用)
        """
        self.cloud = new_cloud
        self.centroid = self.cloud.get_center()
        self.obb = self.cloud.get_oriented_bounding_box()
        self.area = self._estimate_area()

    def scale(self, factor: float) -> 'Plane':
        """
        将本平面整体迁移到物理单位 (mm)。

        这是全流水线唯一的平面尺度施加入口。调用之后，本对象的**全部**派生量
        —— cloud / model 的截距 d / centroid / area / obb —— 都处于物理单位，
        下游 (报表导出、Datum 权重、GD&T) 无需再手工乘以 scale_factor。

        注意:
          - model 的前三列是单位法向，缩放下不变；
          - 平面方程关于原点的全局缩放满足 d → factor·d (点云整体缩放时,
            原内点仍满足 A·x' + B·y' + C·z' + factor·d = 0)；
          - 凸包面积是二阶量，故 area → factor²·area。

        :param factor: 线性比例尺因子，必须为正有限值
        :return: self (便于链式调用)
        """
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"非法的比例尺因子: {factor!r}，必须为正有限值")

        if factor == 1.0:
            return self

        self.cloud.scale(factor, center=(0, 0, 0))

        self.model = np.array([
            self.model[0], self.model[1], self.model[2],
            float(self.model[3]) * factor,
        ])
        self.centroid = np.asarray(self.centroid, dtype=float) * factor
        self.area = float(self.area) * (factor ** 2)

        try:
            self.obb.scale(factor, center=(0, 0, 0))
        except Exception:
            # 个别 Open3D 版本对退化 OBB 的 scale 不稳健，退回复算
            self.obb = self._compute_obb()

        return self

    def __repr__(self):
        """打印对象时的友好格式"""
        a, b, c, d = self.model
        return (f"<Plane ID:{self.id:02d} | "
                f"Eq:[{a:5.2f}x + {b:5.2f}y + {c:5.2f}z + {d:5.2f}=0] | "
                f"Pts:{len(self.cloud.points):5d} | "
                f"Area:{self.area:6.2f}>")