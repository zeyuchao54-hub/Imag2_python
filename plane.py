import logging
import numpy as np
import open3d as o3d


class Plane:
    """
    工业级平面数据对象 (Data Object)
    职责: 封装单一面片的物理与几何属性，提供基础的数学计算接口
    """

    #: OBB 退化修复时，合成支撑点沿退化方向的偏移量相对于点云对角线的比例。
    #: 取 1e-6: 足以让 qhull 摆脱共面退化，又小到不改变包围盒的物理含义。
    _SUPPORT_EPS_RATIO = 1e-6

    #: 判定某一维度"退化"的阈值: 奇异值小于最大奇异值的该倍数即视为无方差。
    _DEGENERATE_SV_RATIO = 1e-9

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

    @classmethod
    def _add_degenerate_supports(cls, points: np.ndarray) -> np.ndarray:
        """
        为几何退化的点云补齐"最小三维构型"，使 qhull 能求出 OBB。

        动机: 完美共面的 CAD 面片采样点云会让 get_oriented_bounding_box()
        因凸包退化而抛 RuntimeError。旧实现是对**所有点**加随机抖动
        (np.random.normal, std=1e-6)，需要固定种子才能复现。

        现改用 CAD-Deform 的"第五顶点"思路并推广到一般形式:
        用 SVD 分解点云，找出方差为 0 的方向，沿每个退化方向补**一个**支撑点。
        零随机性、完全可复现；原有顶点分毫未动，面内几何不被扰动；
        偏移量与点云尺度成比例，而非绝对常量。

        注意: 该方法只能让 qhull "不报错"，并不能得到正确的最小体积盒——
        实测 Open3D 对补齐后的共面点云返回的是**轴对齐**包围盒
        (旋转 35° 的 20x20 共面正方形得到 27.85 而非 20)。
        正确做法见 _analytic_coplanar_obb()。

        :param points: (N,3) 点云坐标
        :return: 补齐后的 (N+k,3) 点云坐标
        """
        centroid = points.mean(axis=0)
        diag = float(np.linalg.norm(np.ptp(points, axis=0)))
        eps = max(diag * cls._SUPPORT_EPS_RATIO, 1e-12)

        supports = []
        try:
            _, s, vh = np.linalg.svd(points - centroid, full_matrices=False)
        except np.linalg.LinAlgError:
            s, vh = None, None

        if s is not None and s[0] > 0:
            # s 降序; 奇异值可忽略的方向即无方差方向，沿其补一个支撑点
            tol = s[0] * cls._DEGENERATE_SV_RATIO
            for k in range(min(3, len(s))):
                if s[k] <= tol:
                    supports.append(centroid + vh[k] * eps)

        if not supports:
            # SVD 不可用或已满秩 (正常不会进入此处): 沿三坐标轴兜底补点
            for axis in (np.array([1.0, 0.0, 0.0]),
                         np.array([0.0, 1.0, 0.0]),
                         np.array([0.0, 0.0, 1.0])):
                supports.append(centroid + axis * eps)

        return np.vstack([points] + [sp[None, :] for sp in supports])

    def _analytic_coplanar_obb(self) -> o3d.geometry.OrientedBoundingBox:
        """
        为完美共面的点云**解析构造** OBB。

        为什么不用 Open3D: 它的 get_oriented_bounding_box() 靠凸包枚举候选朝向。
        共面点云的凸包退化为二维，该算法随之退化为返回**轴对齐**包围盒 ——
        实测旋转 35° 的 20x20 共面正方形得到 27.85 (= 20·(cos35°+sin35°))，
        而正确的最小体积盒是 20x20。旧实现在此处加随机抖动，得到的更糟:
        30x30 共面正方形被算成 42.22x42.22 (≈30·√2，面内尺寸误差 40%)。

        共面情形下我们本来就**知道**平面的法向 (来自 RANSAC 平面方程)，
        因此可以直接构造: 第三轴取平面法向，面内两轴取投影点 SVD 的主方向。
        零随机性、零对 Open3D 凸包内部行为的依赖，且对各向异性的真实面片精确。

        局限 (固有，非缺陷): 若面片在面内各向同性 (例如正方形点阵的协方差为 σ²I)，
        面内主方向不唯一，此时退化为 (u,v) 基下的轴对齐盒，与 Open3D 的降级行为一致。
        """
        points = np.asarray(self.cloud.points)
        centroid = points.mean(axis=0)
        diag = float(np.linalg.norm(np.ptp(points, axis=0)))

        # 第三轴: 平面法向 (退化时退回 SVD 最小方差方向)
        n = np.asarray(self.normal, dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
            n = vh[-1]
        else:
            n = n / n_norm

        # 面内正交基 (u, v)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-12:
            u = np.cross(n, np.array([1.0, 0.0, 0.0]))
            u_norm = float(np.linalg.norm(u))
        u = u / u_norm
        v = np.cross(n, u)  # n、u 均为单位正交向量，v 已是单位向量

        # 面内主方向
        centered = points - centroid
        coords = np.column_stack([centered @ u, centered @ v])
        _, _, vh2 = np.linalg.svd(coords, full_matrices=False)
        p1 = u * vh2[0, 0] + v * vh2[0, 1]
        p2 = u * vh2[1, 0] + v * vh2[1, 1]

        e1 = max(float(np.ptp(centered @ p1)), 0.0)
        e2 = max(float(np.ptp(centered @ p2)), 0.0)
        # 共面点云在法向上的跨度理论上为 0; 下限取尺度相关量，避免下游遇到精确 0
        e3 = max(float(np.ptp(centered @ n)), diag * self._SUPPORT_EPS_RATIO, 1e-12)

        R = np.column_stack([p1, p2, n])
        if float(np.linalg.det(R)) < 0:  # 保证右手系
            R[:, 1] = -R[:, 1]

        return o3d.geometry.OrientedBoundingBox(
            center=centroid, R=R, extent=np.array([e1, e2, e3])
        )

    def _compute_obb(self) -> o3d.geometry.OrientedBoundingBox:
        """计算定向包围盒，对完美共面的 CAD 点云做解析式退化容错。"""
        try:
            return self.cloud.get_oriented_bounding_box()
        except RuntimeError:
            pass

        # 退化路径: 共面点云改用解析构造 (见 _analytic_coplanar_obb 的说明)
        points = np.asarray(self.cloud.points)
        if len(points) >= 3:
            try:
                return self._analytic_coplanar_obb()
            except (np.linalg.LinAlgError, ValueError) as e:
                logging.getLogger("PointToCAD_System.Plane").warning(
                    f"平面 P{self.id} 解析构造 OBB 失败 ({e})，降级为轴对齐包围盒"
                )

        # 点数不足或解析失败: 降级为轴对齐包围盒并告警，
        # 而不是像旧实现那样静默给出一个被随机性污染的 OBB。
        logging.getLogger("PointToCAD_System.Plane").warning(
            f"平面 P{self.id} 无法计算定向包围盒 (点数 {len(points)})，降级为轴对齐包围盒"
        )
        return self.cloud.get_axis_aligned_bounding_box()

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
        # 复用 _compute_obb 而非直接调用 get_oriented_bounding_box():
        # 后者在完美共面点云上会抛 RuntimeError，而融合后的点云正是这种情况的高发场景
        self.obb = self._compute_obb()
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