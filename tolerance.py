"""
tolerance.py
几何公差计算模块 (GD&T)
职责: 根据已匹配的扫描特征与 CAD 名义特征，按 ISO/ASME 标准计算形状、方向、位置、轮廓公差。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import open3d as o3d

from features import PlaneFeature, CylinderFeature, LineFeature, PointFeature


@dataclass
class ToleranceResult:
    """
    单条公差结果

    threshold 记录该项实际使用的判定阈值，使 tolerance.json 自描述——避免下游读者
    误以为所有项都按同一个 ±2.0 mm 判定 (方向公差按角度、形状/位置/轮廓公差按长度，
    两者量纲不同)。algorithm 标注该项的算法性质 (简化实现须显式说明)。
    """
    name: str
    value: float
    nominal: Optional[float] = None
    unit: str = "mm"
    status: str = "N/A"
    features: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    threshold: Optional[float] = None
    algorithm: Optional[str] = None

    def to_dict(self) -> Dict:
        out = {
            "name": self.name,
            "value": round(float(self.value), 6),
            "nominal": round(float(self.nominal), 6) if self.nominal is not None else None,
            "unit": self.unit,
            "status": self.status,
            "features": self.features,
            "details": self.details,
        }
        if self.threshold is not None:
            out["threshold"] = round(float(self.threshold), 6)
        if self.algorithm is not None:
            out["algorithm"] = self.algorithm
        return out


class ToleranceAnalyzer:
    """
    几何公差分析器

    阈值分纲:
      - 方向公差 (垂直度/平行度/倾斜度) 单位为度，使用 angle_threshold_deg;
      - 形状/位置/轮廓公差单位为 mm，使用 tolerance_threshold_mm。
    旧实现把 angle_threshold_deg 硬编码为 2.0 且不可配置，与长度阈值混在同一个
    --tolerance 参数下，读者无法从报告判断每项实际用的是哪个阈值。
    """

    def __init__(self, tolerance_threshold_mm: float = 2.0,
                 angle_threshold_deg: Optional[float] = None):
        self.logger = logging.getLogger("PointToCAD_System.Tolerance")
        self.tolerance_threshold_mm = self._validate(tolerance_threshold_mm,
                                                     "tolerance_threshold_mm")
        # 默认 2.0° 与原硬编码值一致，但现在可通过 --angle_tolerance 配置
        self.angle_threshold_deg = self._validate(
            angle_threshold_deg if angle_threshold_deg is not None else 2.0,
            "angle_threshold_deg",
        )

    @staticmethod
    def _validate(value: float, name: str) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为正有限值，收到: {value!r}")
        return float(value)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def analyze(self,
                scan_planes: List[PlaneFeature],
                cad_planes: List[PlaneFeature],
                scan_cylinders: List[CylinderFeature],
                cad_cylinders: List[CylinderFeature],
                scan_lines: List[LineFeature],
                scan_points: List[PointFeature],
                cad_points: List[PointFeature],
                deviation_signed: Optional[np.ndarray] = None) -> Dict[str, List[ToleranceResult]]:
        """
        计算所有可适用的几何公差。
        返回按类别分组的字典: shape / orientation / location / profile
        """
        results = {
            "shape": [],
            "orientation": [],
            "location": [],
            "profile": [],
        }

        matched_planes = self.match_planes(scan_planes, cad_planes)
        matched_cylinders = self.match_cylinders(scan_cylinders, cad_cylinders)
        matched_points = self.match_points(scan_points, cad_points)

        # ---------------- 形状公差 ----------------
        for sp, cp in matched_planes:
            results["shape"].append(self._flatness(sp, cp))

        for line in scan_lines:
            results["shape"].append(self._straightness(line))

        for sc, cc in matched_cylinders:
            results["shape"].append(self._circularity(sc, cc))
            results["shape"].append(self._cylindricity(sc, cc))

        # ---------------- 方向公差 ----------------
        for i in range(len(matched_planes)):
            for j in range(i + 1, len(matched_planes)):
                sp1, cp1 = matched_planes[i]
                sp2, cp2 = matched_planes[j]

                dot = float(np.clip(np.abs(sp1.normal @ sp2.normal), -1.0, 1.0))
                measured_angle = np.degrees(np.arccos(dot))

                # 近似垂直 (80°~100°): 计算垂直度
                if 80.0 <= measured_angle <= 100.0:
                    results["orientation"].append(self._perpendicularity(sp1, cp1, sp2, cp2))
                # 近似平行 (0°~20° 或 160°~180°): 计算平行度
                if measured_angle <= 20.0 or measured_angle >= 160.0:
                    results["orientation"].append(self._parallelism(sp1, cp1, sp2, cp2))
                # 倾斜关系 (20°~80° 或 100°~160°): 计算倾斜度
                if (20.0 < measured_angle < 80.0) or (100.0 < measured_angle < 160.0):
                    results["orientation"].append(self._angularity(sp1, cp1, sp2, cp2))

        # ---------------- 位置公差 ----------------
        for sc, cc in matched_cylinders:
            results["location"].append(self._position_cylinder(sc, cc))

        for sp, cp in matched_points:
            results["location"].append(self._position_point(sp, cp))

        # 同轴度：多圆柱之间
        if len(matched_cylinders) >= 2:
            for i in range(len(matched_cylinders)):
                for j in range(i + 1, len(matched_cylinders)):
                    sc1, cc1 = matched_cylinders[i]
                    sc2, cc2 = matched_cylinders[j]
                    results["location"].append(self._concentricity(sc1, cc1, sc2, cc2))

        # 对称度：对每一对平行平面，计算中分面对称偏差
        parallel_pairs = self._find_parallel_plane_pairs(matched_planes)
        for (sp1, cp1), (sp2, cp2) in parallel_pairs:
            results["location"].append(self._symmetry(sp1, cp1, sp2, cp2))

        # ---------------- 轮廓公差 ----------------
        results["profile"].append(self._profile_of_surface(deviation_signed))

        # 汇总统计
        total = sum(len(v) for v in results.values())
        self.logger.info(f"几何公差计算完成: 共 {total} 项 (shape={len(results['shape'])}, "
                         f"orientation={len(results['orientation'])}, location={len(results['location'])}, "
                         f"profile={len(results['profile'])}).")

        return results

    # ------------------------------------------------------------------
    # Feature matching wrappers
    # ------------------------------------------------------------------
    @staticmethod
    def match_planes(scan_planes: List[PlaneFeature],
                     cad_planes: List[PlaneFeature]) -> List[Tuple[PlaneFeature, PlaneFeature]]:
        from features import FeatureExtractor
        return FeatureExtractor.match_planes(scan_planes, cad_planes)

    @staticmethod
    def match_cylinders(scan_cylinders: List[CylinderFeature],
                        cad_cylinders: List[CylinderFeature]) -> List[Tuple[CylinderFeature, CylinderFeature]]:
        from features import FeatureExtractor
        return FeatureExtractor.match_cylinders(scan_cylinders, cad_cylinders)

    @staticmethod
    def match_points(scan_points: List[PointFeature],
                     cad_points: List[PointFeature]) -> List[Tuple[PointFeature, PointFeature]]:
        from features import FeatureExtractor
        return FeatureExtractor.match_points(scan_points, cad_points)

    @staticmethod
    def _find_parallel_plane_pairs(matched_planes: List[Tuple[PlaneFeature, PlaneFeature]]
                                   ) -> List[Tuple[Tuple[PlaneFeature, PlaneFeature],
                                                   Tuple[PlaneFeature, PlaneFeature]]]:
        """从已匹配平面中找出所有平行平面对"""
        pairs = []
        n = len(matched_planes)
        for i in range(n):
            for j in range(i + 1, n):
                sp1, _ = matched_planes[i]
                sp2, _ = matched_planes[j]
                dot = float(np.clip(np.abs(sp1.normal @ sp2.normal), -1.0, 1.0))
                angle = np.degrees(np.arccos(dot))
                if angle <= 5.0:
                    pairs.append((matched_planes[i], matched_planes[j]))
        return pairs

    # ------------------------------------------------------------------
    # Shape tolerances
    # ------------------------------------------------------------------
    def _flatness(self, scan_plane: PlaneFeature, cad_plane: PlaneFeature) -> ToleranceResult:
        """
        平面度 (Flatness): 被测实际表面对其理想平面的允许变动量。
        工程常用算法: 最小包容区域宽度 = max(signed_dist) - min(signed_dist)
        """
        points = scan_plane.points
        if len(points) < 3:
            return ToleranceResult(
                name=f"Flatness_P{scan_plane.id}", value=0.0,
                status="N/A", features=[f"Plane{scan_plane.id}"]
            )

        # 最佳拟合平面 (PCA 最小特征值方向)
        centered = points - np.mean(points, axis=0)
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]
        normal = normal / np.linalg.norm(normal)

        # 与 CAD 名义法向保持一致
        if normal @ cad_plane.normal < 0:
            normal = -normal

        d_fit = -float(np.mean(points @ normal))
        signed_dists = points @ normal + d_fit

        flatness = float(np.max(signed_dists) - np.min(signed_dists))
        max_abs_dev = float(np.max(np.abs(signed_dists)))

        return ToleranceResult(
            name=f"Flatness_P{scan_plane.id}",
            value=flatness,
            nominal=0.0,
            unit="mm",
            status="PASS" if flatness <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="max-min range of best-fit plane (simplified; ISO 最小包容区域需迭代求解放置平面)",
            features=[f"Scan_Plane{scan_plane.id}", f"CAD_Plane{cad_plane.id}"],
            details={
                "max_signed_dev": float(np.max(signed_dists)),
                "min_signed_dev": float(np.min(signed_dists)),
                "max_abs_dev": max_abs_dev,
                "point_count": int(len(points)),
            }
        )

    def _straightness(self, line: LineFeature) -> ToleranceResult:
        """
        直线度 (Straightness): 被测实际线对其理想直线的允许变动量。
        算法: 将边缘点投影到垂直于理想直线的平面，计算最小包容圆直径。
        简化实现: 用最大径向偏差的两倍作为直线度。
        """
        points = line.points
        if len(points) < 3:
            return ToleranceResult(
                name=f"Straightness_E{line.id}", value=0.0,
                status="N/A", features=[f"Edge_{line.plane_ids}"],
                details={"note": "边缘点不足"}
            )

        direction = line.direction / np.linalg.norm(line.direction)
        centered = points - line.point
        perp = centered - np.outer(centered @ direction, direction)

        # 建立垂直平面内的二维坐标
        ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(direction, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(direction, u)
        coords_2d = np.column_stack((perp @ u, perp @ v))

        # 最小包容圆直径近似 = 2 * max(径向距离)
        radial_dists = np.linalg.norm(coords_2d, axis=1)
        straightness = float(2.0 * np.max(radial_dists))

        return ToleranceResult(
            name=f"Straightness_E{line.id}",
            value=straightness,
            nominal=0.0,
            unit="mm",
            status="PASS" if straightness <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="2 x max radial deviation (simplified; ISO 最小包容圆柱需迭代求解放置圆柱)",
            features=[f"Edge_P{line.plane_ids[0]}_P{line.plane_ids[1]}"],
            details={
                "point_count": len(points),
                "max_radial_dev": float(np.max(radial_dists)),
            }
        )

    def _circularity(self, scan_cyl: CylinderFeature, cad_cyl: CylinderFeature) -> ToleranceResult:
        """
        圆度 (Circularity): 被测圆柱同一正截面上实际轮廓对理想圆的允许变动量。
        简化: 在圆柱中部取一个正截面，拟合圆并计算径向极差。
        """
        if len(scan_cyl.points) < 6:
            return ToleranceResult(
                name=f"Circularity_C{scan_cyl.id}", value=0.0,
                status="N/A", features=[f"Cylinder{scan_cyl.id}"]
            )

        axis = scan_cyl.axis
        center = scan_cyl.center
        points = scan_cyl.points

        # 投影到圆柱中部截面
        projections = (points - center) @ axis
        mid_proj = float(np.median(projections))
        mid_plane_center = center + axis * mid_proj
        radial_vecs = (points - mid_plane_center) - np.outer((points - mid_plane_center) @ axis, axis)
        radii = np.linalg.norm(radial_vecs, axis=1)

        circularity = float(np.max(radii) - np.min(radii))

        return ToleranceResult(
            name=f"Circularity_C{scan_cyl.id}",
            value=circularity,
            nominal=0.0,
            unit="mm",
            status="PASS" if circularity <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="radial range at mid-height section (simplified; ISO 圆度需最小包容圆)",
            features=[f"Scan_Cylinder{scan_cyl.id}", f"CAD_Cylinder{cad_cyl.id}"],
            details={
                "mean_radius": float(np.mean(radii)),
                "nominal_radius": cad_cyl.radius,
            }
        )

    def _cylindricity(self, scan_cyl: CylinderFeature, cad_cyl: CylinderFeature) -> ToleranceResult:
        """
        圆柱度 (Cylindricity): 被测实际圆柱面对其理想圆柱面的允许变动量。
        算法: 以检测到的圆柱轴线和半径为理想圆柱，计算所有点到该圆柱的径向偏差极差。
        """
        if len(scan_cyl.points) < 6:
            return ToleranceResult(
                name=f"Cylindricity_C{scan_cyl.id}", value=0.0,
                status="N/A", features=[f"Cylinder{scan_cyl.id}"]
            )

        axis = scan_cyl.axis
        center = scan_cyl.center
        points = scan_cyl.points

        radial_vecs = (points - center) - np.outer((points - center) @ axis, axis)
        radii = np.linalg.norm(radial_vecs, axis=1)

        cylindricity = float(np.max(radii) - np.min(radii))

        return ToleranceResult(
            name=f"Cylindricity_C{scan_cyl.id}",
            value=cylindricity,
            nominal=0.0,
            unit="mm",
            status="PASS" if cylindricity <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="radial range about fitted axis (simplified; ISO 圆柱度需最小包容圆柱)",
            features=[f"Scan_Cylinder{scan_cyl.id}", f"CAD_Cylinder{cad_cyl.id}"],
            details={
                "mean_radius": float(np.mean(radii)),
                "nominal_radius": cad_cyl.radius,
            }
        )

    # ------------------------------------------------------------------
    # Orientation tolerances
    # ------------------------------------------------------------------
    def _perpendicularity(self,
                          sp1: PlaneFeature, cp1: PlaneFeature,
                          sp2: PlaneFeature, cp2: PlaneFeature) -> ToleranceResult:
        """垂直度 (Perpendicularity): 两平面夹角偏离 90° 的程度。"""
        dot = float(np.clip(np.abs(sp1.normal @ sp2.normal), -1.0, 1.0))
        angle = np.degrees(np.arccos(dot))
        deviation = abs(angle - 90.0)

        return ToleranceResult(
            name=f"Perpendicularity_P{sp1.id}_P{sp2.id}",
            value=deviation,
            nominal=90.0,
            unit="deg",
            status="PASS" if deviation <= self.angle_threshold_deg else "FAIL",
            threshold=self.angle_threshold_deg,
            features=[f"Plane{sp1.id}", f"Plane{sp2.id}"],
            details={"measured_angle": round(angle, 4)}
        )

    def _parallelism(self,
                     sp1: PlaneFeature, cp1: PlaneFeature,
                     sp2: PlaneFeature, cp2: PlaneFeature) -> ToleranceResult:
        """平行度 (Parallelism): 两平面夹角偏离 0° 的程度。"""
        dot = float(np.clip(np.abs(sp1.normal @ sp2.normal), -1.0, 1.0))
        angle = np.degrees(np.arccos(dot))

        return ToleranceResult(
            name=f"Parallelism_P{sp1.id}_P{sp2.id}",
            value=angle,
            nominal=0.0,
            unit="deg",
            status="PASS" if angle <= self.angle_threshold_deg else "FAIL",
            threshold=self.angle_threshold_deg,
            features=[f"Plane{sp1.id}", f"Plane{sp2.id}"],
            details={"measured_angle": round(angle, 4)}
        )

    def _angularity(self,
                    sp1: PlaneFeature, cp1: PlaneFeature,
                    sp2: PlaneFeature, cp2: PlaneFeature) -> ToleranceResult:
        """
        倾斜度 (Angularity): 两平面实际夹角相对于 CAD 名义夹角的偏离。
        对非 0°/90° 的斜面配合尤为重要。
        """
        dot_s = float(np.clip(np.abs(sp1.normal @ sp2.normal), -1.0, 1.0))
        angle_s = np.degrees(np.arccos(dot_s))

        dot_c = float(np.clip(np.abs(cp1.normal @ cp2.normal), -1.0, 1.0))
        angle_c = np.degrees(np.arccos(dot_c))

        deviation = abs(angle_s - angle_c)

        return ToleranceResult(
            name=f"Angularity_P{sp1.id}_P{sp2.id}",
            value=deviation,
            nominal=round(float(angle_c), 4),
            unit="deg",
            status="PASS" if deviation <= self.angle_threshold_deg else "FAIL",
            threshold=self.angle_threshold_deg,
            features=[f"Plane{sp1.id}", f"Plane{sp2.id}"],
            details={
                "measured_angle": round(float(angle_s), 4),
                "nominal_angle": round(float(angle_c), 4),
            }
        )

    # ------------------------------------------------------------------
    # Location tolerances
    # ------------------------------------------------------------------
    def _position_cylinder(self, scan_cyl: CylinderFeature, cad_cyl: CylinderFeature) -> ToleranceResult:
        """位置度 (Position): 圆柱轴线/中心到名义位置的偏差。
        GD&T 位置度为直径公差带 (diametric zone)，数值 = 2 × 中心偏移量。"""
        dist = 2.0 * float(np.linalg.norm(scan_cyl.center - cad_cyl.center))

        return ToleranceResult(
            name=f"Position_C{scan_cyl.id}",
            value=dist,
            nominal=0.0,
            unit="mm",
            status="PASS" if dist <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            features=[f"Scan_Cylinder{scan_cyl.id}", f"CAD_Cylinder{cad_cyl.id}"],
            details={
                "scan_center": scan_cyl.center.tolist(),
                "cad_center": cad_cyl.center.tolist(),
                "zone": "diametric (2x center offset)",
            }
        )

    def _position_point(self, scan_pt: PointFeature, cad_pt: PointFeature) -> ToleranceResult:
        """位置度 (Position): 顶点到名义位置的偏差。
        GD&T 位置度为直径公差带 (diametric zone)，数值 = 2 × 中心偏移量。"""
        dist = 2.0 * float(np.linalg.norm(scan_pt.coord - cad_pt.coord))

        return ToleranceResult(
            name=f"Position_V{scan_pt.id}",
            value=dist,
            nominal=0.0,
            unit="mm",
            status="PASS" if dist <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            features=[f"Scan_Vertex{scan_pt.id}", f"CAD_Vertex{cad_pt.id}"],
            details={
                "scan_coord": scan_pt.coord.tolist(),
                "cad_coord": cad_pt.coord.tolist(),
                "zone": "diametric (2x center offset)",
            }
        )

    def _concentricity(self,
                       sc1: CylinderFeature, cc1: CylinderFeature,
                       sc2: CylinderFeature, cc2: CylinderFeature) -> ToleranceResult:
        """同轴度 (Concentricity): 两圆柱轴线之间的距离。"""
        # 简化: 当两轴近似平行时，取两轴上最近点距离
        axis1, center1 = sc1.axis, sc1.center
        axis2, center2 = sc2.axis, sc2.center

        # 计算两空间直线的最短距离
        diff = center2 - center1
        cross = np.cross(axis1, axis2)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-8:
            # 两轴平行
            dist = float(np.linalg.norm(diff - np.dot(diff, axis1) * axis1))
        else:
            dist = float(abs(np.dot(diff, cross)) / cross_norm)

        return ToleranceResult(
            name=f"Concentricity_C{sc1.id}_C{sc2.id}",
            value=dist,
            nominal=0.0,
            unit="mm",
            status="PASS" if dist <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="shortest distance between two axes (simplified; 轴不平行时 ISO 需按公差带宽度定义)",
            features=[f"Cylinder{sc1.id}", f"Cylinder{sc2.id}"],
            details={
                "axis1": axis1.tolist(),
                "axis2": axis2.tolist(),
            }
        )

    def _symmetry(self,
                  sp1: PlaneFeature, cp1: PlaneFeature,
                  sp2: PlaneFeature, cp2: PlaneFeature) -> ToleranceResult:
        """
        对称度 (Symmetry): 一对平行平面的中分面相对于名义中分面的偏移。
        简化: 计算两平面中点连线在法向方向上的中点偏移。
        """
        # 理想中分面应在两 CAD 平面正中间
        mid_cad = (cp1.centroid + cp2.centroid) / 2.0
        mid_scan = (sp1.centroid + sp2.centroid) / 2.0

        # 中分面法向取平均
        normal = (sp1.normal + sp2.normal) / 2.0
        normal = normal / np.linalg.norm(normal)

        symmetry = float(abs(np.dot(mid_scan - mid_cad, normal)))

        return ToleranceResult(
            name=f"Symmetry_P{sp1.id}_P{sp2.id}",
            value=symmetry,
            nominal=0.0,
            unit="mm",
            status="PASS" if symmetry <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="mid-plane offset along normal (simplified; ISO 对称度需最小包容平行面间距)",
            features=[f"Plane{sp1.id}", f"Plane{sp2.id}"],
            details={
                "nominal_midplane": mid_cad.tolist(),
                "measured_midplane": mid_scan.tolist(),
            }
        )

    # ------------------------------------------------------------------
    # Profile tolerance
    # ------------------------------------------------------------------
    def _profile_of_surface(self, deviation_signed: Optional[np.ndarray]) -> ToleranceResult:
        """
        面轮廓度 (Profile of a Surface): 实际表面轮廓相对于理想轮廓的允许变动量。
        使用已有的 scan→cad signed deviation 统计量。
        """
        if deviation_signed is None or len(deviation_signed) == 0:
            return ToleranceResult(
                name="Profile_of_Surface",
                value=0.0,
                nominal=0.0,
                unit="mm",
                status="N/A",
                features=["scan_surface"],
                details={"note": "无偏差数据"}
            )

        max_dev = float(np.max(deviation_signed))
        min_dev = float(np.min(deviation_signed))
        profile = max_dev - min_dev  # 轮廓度常用最大-最小偏差

        return ToleranceResult(
            name="Profile_of_Surface",
            value=profile,
            nominal=0.0,
            unit="mm",
            status="PASS" if profile <= self.tolerance_threshold_mm else "FAIL",
            threshold=self.tolerance_threshold_mm,
            algorithm="max-min of signed deviation (simplified; ISO 面轮廓度需双向公差带)",
            features=["scan_surface"],
            details={
                "max_deviation": max_dev,
                "min_deviation": min_dev,
                "mean_deviation": float(np.mean(deviation_signed)),
            }
        )
