"""
features.py
几何特征提取与匹配模块
职责: 从扫描点云和 CAD 点云中提取平面、圆柱、棱边、顶点等工业几何特征，
      并提供特征匹配接口，为几何公差计算提供输入。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import open3d as o3d

from plane import Plane


@dataclass
class PlaneFeature:
    """平面特征"""
    id: int
    normal: np.ndarray
    centroid: np.ndarray
    d: float
    points: np.ndarray
    source: str
    matched_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "normal": self.normal.tolist(),
            "centroid": self.centroid.tolist(),
            "d": float(self.d),
            "point_count": len(self.points),
            "matched_id": self.matched_id,
        }


@dataclass
class CylinderFeature:
    """圆柱特征 (孔、轴、销)"""
    id: int
    axis: np.ndarray
    center: np.ndarray
    radius: float
    height: float
    points: np.ndarray
    source: str
    matched_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "axis": self.axis.tolist(),
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "height": float(self.height),
            "point_count": len(self.points),
            "matched_id": self.matched_id,
        }


@dataclass
class LineFeature:
    """直线特征 (两平面交线 / 棱边)"""
    id: int
    direction: np.ndarray
    point: np.ndarray
    points: np.ndarray
    source: str
    plane_ids: Tuple[int, int]
    matched_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "direction": self.direction.tolist(),
            "point": self.point.tolist(),
            "point_count": len(self.points),
            "plane_ids": list(self.plane_ids),
            "matched_id": self.matched_id,
        }


@dataclass
class PointFeature:
    """点特征 (顶点 / 孔心)"""
    id: int
    coord: np.ndarray
    source: str
    matched_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "coord": self.coord.tolist(),
            "matched_id": self.matched_id,
        }


class FeatureExtractor:
    """几何特征提取器"""

    def __init__(self):
        self.logger = logging.getLogger("PointToCAD_System.Features")

    # ------------------------------------------------------------------
    # Plane features
    # ------------------------------------------------------------------
    def extract_planes_from_scan(self, planes: List[Plane], source: str = "scan") -> List[PlaneFeature]:
        """从已有的 Plane 对象列表中提取平面特征"""
        features = []
        for i, p in enumerate(planes):
            points = np.asarray(p.cloud.points) if p.cloud else np.zeros((0, 3))
            features.append(PlaneFeature(
                id=i + 1,
                normal=np.array(p.normal),
                centroid=np.array(p.centroid),
                d=float(p.model[3]),
                points=points,
                source=source,
            ))
        return features

    def extract_planes_from_pcd(self, pcd: o3d.geometry.PointCloud,
                                max_planes: int = 10,
                                distance_threshold: Optional[float] = None,
                                source: str = "cad") -> List[PlaneFeature]:
        """对点云 (如 CAD 点云) 执行 RANSAC 平面提取"""
        from detector import RansacDetector
        from merger import PlaneMerger

        if pcd is None or pcd.is_empty():
            return []

        detector = RansacDetector(max_planes=max_planes, distance_threshold=distance_threshold)
        raw_planes, _ = detector.detect(pcd)

        merger = PlaneMerger()
        merged = merger.merge(raw_planes)

        return self.extract_planes_from_scan(merged, source=source)

    # ------------------------------------------------------------------
    # Cylinder features
    # ------------------------------------------------------------------
    def detect_cylinders(self,
                         pcd: o3d.geometry.PointCloud,
                         max_cylinders: int = 5,
                         distance_threshold: float = 1.0,
                         radius_limits: Tuple[float, float] = (1.0, 100.0),
                         source: str = "scan",
                         ransac_iterations: int = 500,
                         seed: int = 42) -> List[CylinderFeature]:
        """
        使用自定义 RANSAC 从点云中检测圆柱特征。
        当前 Open3D 0.19.0 不包含 segment_cylinder，因此使用轴采样 + 圆拟合实现。

        :param seed: RANSAC 采样种子。固定种子保证同一输入得到同一组圆柱，
                     使 tolerance.json 可复现、可追溯。
        """
        if pcd is None or len(pcd.points) < 100:
            return []

        points = np.asarray(pcd.points)
        remaining_mask = np.ones(len(points), dtype=bool)
        cylinders = []
        rng = np.random.default_rng(seed)

        for cyl_id in range(max_cylinders):
            remaining_idx = np.where(remaining_mask)[0]
            if len(remaining_idx) < 100:
                break

            remaining_points = points[remaining_idx]
            best_model = self._ransac_cylinder(
                remaining_points, distance_threshold, radius_limits, ransac_iterations,
                rng=rng,
            )
            if best_model is None:
                break

            axis, center, radius, inlier_mask = best_model
            inlier_count = int(np.sum(inlier_mask))
            if inlier_count < 50:
                break

            inlier_points = remaining_points[inlier_mask]
            inlier_global_idx = remaining_idx[inlier_mask]
            remaining_mask[inlier_global_idx] = False

            projections = (inlier_points - center) @ axis
            h_min, h_max = float(np.min(projections)), float(np.max(projections))
            height = h_max - h_min
            center_adjusted = center + axis * (h_min + h_max) / 2.0

            cylinders.append(CylinderFeature(
                id=cyl_id + 1,
                axis=axis,
                center=center_adjusted,
                radius=radius,
                height=height,
                points=inlier_points,
                source=source,
            ))

        if cylinders:
            self.logger.info(f"{source} 点云检测到 {len(cylinders)} 个圆柱特征。")
        return cylinders

    def _ransac_cylinder(self,
                         points: np.ndarray,
                         distance_threshold: float,
                         radius_limits: Tuple[float, float],
                         iterations: int = 500,
                         rng: Optional[np.random.Generator] = None):
        """
        自定义 RANSAC 圆柱拟合。
        策略: 随机采样两点确定轴线方向，将点投影到垂直平面后用代数圆拟合，统计内点。

        :param rng: 随机数生成器。由 detect_cylinders 传入并跨多次检测复用，
                    保证可复现; 为 None 时退化为固定种子 (而非不定态)。
        """
        if len(points) < 50:
            return None

        best = None
        best_inliers = 0
        if rng is None:
            rng = np.random.default_rng(42)

        for _ in range(iterations):
            # 随机采样两个点定义轴线方向
            idx = rng.choice(len(points), 2, replace=False)
            p1, p2 = points[idx]
            axis = p2 - p1
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis = axis / axis_norm

            # 在垂直平面建立局部二维坐标系
            ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(axis, ref)
            u = u / np.linalg.norm(u)
            v = np.cross(axis, u)

            centered = points - p1
            perp = centered - np.outer(centered @ axis, axis)
            coords_2d = np.column_stack((perp @ u, perp @ v))

            # 代数圆拟合 (Kasa 方法)
            try:
                circle_center_2d, radius = self._fit_circle_2d(coords_2d)
            except Exception:
                continue

            if radius < radius_limits[0] or radius > radius_limits[1]:
                continue

            # 圆心在 3D 中的位置
            circle_center_3d = p1 + circle_center_2d[0] * u + circle_center_2d[1] * v

            # 计算每个点到圆柱表面的距离
            radial_vecs = (points - circle_center_3d) - np.outer((points - circle_center_3d) @ axis, axis)
            radii = np.linalg.norm(radial_vecs, axis=1)
            distances = np.abs(radii - radius)

            inliers = distances <= distance_threshold
            inlier_count = int(np.sum(inliers))

            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best = (axis, circle_center_3d, float(radius), inliers.copy())

        return best

    @staticmethod
    def _fit_circle_2d(points_2d: np.ndarray) -> Tuple[np.ndarray, float]:
        """代数圆拟合 (Kasa 方法)"""
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x ** 2 + y ** 2
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, c, d = sol
        center = np.array([a / 2.0, c / 2.0])
        radius = np.sqrt(d + a ** 2 / 4.0 + c ** 2 / 4.0)
        return center, float(radius)

    # ------------------------------------------------------------------
    # Line / Edge features
    # ------------------------------------------------------------------
    def extract_lines_from_planes(self,
                                  planes: List[PlaneFeature],
                                  source_pcd: Optional[o3d.geometry.PointCloud] = None,
                                  edge_width: float = 1.0) -> List[LineFeature]:
        """
        从垂直平面对中提取交线（棱边）。
        如果提供 source_pcd，则在交线附近采样实际边缘点用于后续公差计算。
        """
        lines = []
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                p1, p2 = planes[i], planes[j]

                # 只保留近似垂直的平面对
                dot = float(np.clip(np.abs(p1.normal @ p2.normal), -1.0, 1.0))
                angle = np.degrees(np.arccos(dot))
                if angle < 60.0 or angle > 120.0:
                    continue

                direction = np.cross(p1.normal, p2.normal)
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                direction = direction / norm

                # 求交线上一点: 最小二乘解两平面方程
                A = np.vstack([p1.normal, p2.normal])
                b = np.array([-p1.d, -p2.d], dtype=float)
                try:
                    point = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    continue

                # 如果有点云，在交线附近提取边缘点
                edge_points = np.zeros((0, 3))
                if source_pcd is not None and not source_pcd.is_empty():
                    edge_points = self._extract_edge_points(
                        source_pcd, p1, p2, edge_width=edge_width
                    )

                lines.append(LineFeature(
                    id=len(lines) + 1,
                    direction=direction,
                    point=point,
                    points=edge_points,
                    source=p1.source,
                    plane_ids=(p1.id, p2.id),
                ))

        return lines

    @staticmethod
    def _extract_edge_points(pcd: o3d.geometry.PointCloud,
                             plane1: PlaneFeature,
                             plane2: PlaneFeature,
                             edge_width: float = 1.0) -> np.ndarray:
        """提取同时靠近两个平面的点作为边缘点"""
        points = np.asarray(pcd.points)
        d1 = np.abs(points @ plane1.normal + plane1.d)
        d2 = np.abs(points @ plane2.normal + plane2.d)
        mask = (d1 <= edge_width) & (d2 <= edge_width)
        return points[mask]

    # ------------------------------------------------------------------
    # Point / Vertex features
    # ------------------------------------------------------------------
    def extract_points_from_array(self,
                                  vertices: np.ndarray,
                                  source: str = "scan") -> List[PointFeature]:
        """从顶点数组中提取点特征"""
        points = []
        if vertices is None or len(vertices) == 0:
            return points
        for i, v in enumerate(vertices):
            points.append(PointFeature(
                id=i + 1,
                coord=np.array(v, dtype=float),
                source=source,
            ))
        return points

    def extract_points_from_plane_intersections(self,
                                                 planes: List[PlaneFeature],
                                                 source: str = "cad") -> List[PointFeature]:
        """通过求 CAD 平面的三面交点来恢复顶点"""
        points = []
        n = len(planes)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = planes[i], planes[j], planes[k]
                    A = np.vstack([p1.normal, p2.normal, p3.normal])
                    b = np.array([-p1.d, -p2.d, -p3.d], dtype=float)
                    if np.abs(np.linalg.det(A)) < 1e-5:
                        continue
                    try:
                        v = np.linalg.solve(A, b)
                        points.append(PointFeature(
                            id=len(points) + 1,
                            coord=v,
                            source=source,
                        ))
                    except Exception:
                        continue
        return points

    # ------------------------------------------------------------------
    # Feature matching
    # ------------------------------------------------------------------
    @staticmethod
    def match_planes(scan_planes: List[PlaneFeature],
                     cad_planes: List[PlaneFeature],
                     max_angle_deg: float = 15.0,
                     max_dist_mm: float = 100.0) -> List[Tuple[PlaneFeature, PlaneFeature]]:
        """
        根据法向方向分组匹配扫描平面与 CAD 平面。
        对每一组平行平面，按质心在法向上的投影排序后一一对应匹配，
        避免跨组误匹配，对立方体/棱柱类零件更稳健。
        """
        matched = []
        used_cad = set()

        def _group_by_normal(planes: List[PlaneFeature]) -> List[List[PlaneFeature]]:
            """将法向近似平行的平面归为一组"""
            groups: List[List[PlaneFeature]] = []
            for p in planes:
                found = False
                for g in groups:
                    ref = g[0].normal
                    angle = np.degrees(np.arccos(np.clip(np.abs(p.normal @ ref), -1.0, 1.0)))
                    if angle <= max_angle_deg:
                        g.append(p)
                        found = True
                        break
                if not found:
                    groups.append([p])
            return groups

        scan_groups = _group_by_normal(scan_planes)
        cad_groups = _group_by_normal(cad_planes)

        for sg in scan_groups:
            ref_normal = sg[0].normal
            # 找到最相似的 CAD 组
            best_cg = None
            best_angle = float("inf")
            for cg in cad_groups:
                if cg[0].id in used_cad:
                    continue
                angle = np.degrees(np.arccos(np.clip(np.abs(ref_normal @ cg[0].normal), -1.0, 1.0)))
                if angle < best_angle:
                    best_angle = angle
                    best_cg = cg

            if best_cg is None or best_angle > max_angle_deg:
                continue

            # 按法向投影排序后一一匹配
            sg_sorted = sorted(sg, key=lambda p: float(p.centroid @ ref_normal))
            cg_sorted = sorted(best_cg, key=lambda p: float(p.centroid @ p.normal))

            for sp, cp in zip(sg_sorted, cg_sorted):
                dist = float(np.linalg.norm(sp.centroid - cp.centroid))
                if dist > max_dist_mm:
                    continue
                matched.append((sp, cp))
                used_cad.add(cp.id)
                sp.matched_id = cp.id

        return matched

    @staticmethod
    def match_cylinders(scan_cylinders: List[CylinderFeature],
                        cad_cylinders: List[CylinderFeature],
                        max_angle_deg: float = 15.0,
                        max_dist_mm: float = 50.0,
                        max_radius_diff_mm: float = 10.0) -> List[Tuple[CylinderFeature, CylinderFeature]]:
        """根据轴向、中心位置和半径匹配圆柱"""
        matched = []
        used_cad = set()

        for sc in scan_cylinders:
            best_match: Optional[CylinderFeature] = None
            best_score = float("inf")
            for cc in cad_cylinders:
                if cc.id in used_cad:
                    continue
                angle = np.degrees(np.arccos(np.clip(np.abs(sc.axis @ cc.axis), -1.0, 1.0)))
                dist = float(np.linalg.norm(sc.center - cc.center))
                rad_diff = abs(sc.radius - cc.radius)
                if angle > max_angle_deg or dist > max_dist_mm or rad_diff > max_radius_diff_mm:
                    continue
                score = angle + dist / 20.0 + rad_diff / 2.0
                if score < best_score:
                    best_score = score
                    best_match = cc

            if best_match is not None:
                matched.append((sc, best_match))
                used_cad.add(best_match.id)
                sc.matched_id = best_match.id

        return matched

    @staticmethod
    def match_points(scan_points: List[PointFeature],
                     cad_points: List[PointFeature],
                     max_dist_mm: float = 10.0) -> List[Tuple[PointFeature, PointFeature]]:
        """根据欧氏距离最近邻匹配点特征"""
        matched = []
        used_cad = set()

        for sp in scan_points:
            best_match: Optional[PointFeature] = None
            best_dist = float("inf")
            for cp in cad_points:
                if cp.id in used_cad:
                    continue
                dist = float(np.linalg.norm(sp.coord - cp.coord))
                if dist < best_dist and dist <= max_dist_mm:
                    best_dist = dist
                    best_match = cp

            if best_match is not None:
                matched.append((sp, best_match))
                used_cad.add(best_match.id)
                sp.matched_id = best_match.id

        return matched
